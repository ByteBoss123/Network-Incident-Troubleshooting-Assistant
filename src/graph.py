"""Topology graph built from real telemetry.

HDFS: (:Host)-[:CO_REPLICA {blocks}]-(:Host) between hosts holding replicas of the same block,
      (:Host)-[:REREPLICATED]->(:Host) from "Transmitted block" lines,
      (:Block)-[:STORED_ON]->(:Host) from addStoredBlock lines, (:Host)-[:IN_SUBNET]->(:Subnet /16).
BGL:  (:Rack)-[:HAS_MIDPLANE]->(:Midplane)-[:HAS_NODE]->(:Node) parsed from node ids, with alert counts.

The same graph is (a) exported as CSV for Neo4j LOAD CSV / the driver loader in neo4j_load.py and
(b) analysed in-process with NetworkX so results are reproducible without a running Neo4j server.
Analysis: which hosts carry a statistically elevated share of anomalous blocks (one-sided binomial
test against the overall anomaly rate, Bonferroni-corrected), and graph structure stats.
"""
import json
from pathlib import Path

import duckdb
import networkx as nx
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "netincident.duckdb"
EXPORT = ROOT / "data" / "neo4j_import"
OUT = ROOT / "results"


def export_tables(con):
    EXPORT.mkdir(parents=True, exist_ok=True)
    q = {
        "hosts": """
            WITH ips AS (SELECT src_ip ip FROM hdfs_transfers UNION SELECT dst_ip FROM hdfs_transfers
                         UNION SELECT host_ip FROM hdfs_replicas)
            SELECT ip, split_part(ip,'.',1)||'.'||split_part(ip,'.',2)||'.0.0/16' AS subnet FROM ips""",
        # In the Loghub HDFS release the "Receiving block ... src: dest:" lines always carry src == dest,
        # so host-to-host edges come from (a) replicas of the same block (write-pipeline peers) and
        # (b) explicit re-replication transfers ("<host>:Transmitted block <blk> to /<host>").
        "co_replica": """
            SELECT a.host_ip AS src_ip, b.host_ip AS dst_ip, COUNT(*) AS blocks,
                   SUM(k.is_anomaly::INT) AS anomalous_blocks
            FROM hdfs_replicas a JOIN hdfs_replicas b ON a.block_id = b.block_id AND a.host_ip < b.host_ip
            JOIN hdfs_blocks k ON k.block_id = a.block_id GROUP BY 1,2""",
        "rereplication": """
            SELECT regexp_extract(content, '^(\\d+\\.\\d+\\.\\d+\\.\\d+)', 1) AS src_ip,
                   regexp_extract(content, 'to /(\\d+\\.\\d+\\.\\d+\\.\\d+)', 1) AS dst_ip,
                   block_id FROM hdfs_events WHERE event_id = 'E16'""",
        "blocks": """
            SELECT b.block_id, b.is_anomaly, s.pca_flag, s.pca_residual, s.split
            FROM hdfs_blocks b JOIN read_csv_auto('{scores}') s USING(block_id)""".format(
            scores=OUT / "test_scores.csv"),
        "stored_on": "SELECT block_id, host_ip FROM hdfs_replicas",
        "received_by": "SELECT DISTINCT block_id, dst_ip AS host_ip FROM hdfs_transfers",
        "bgl_nodes": """
            SELECT node, rack, midplane, node_card, COUNT(*) AS lines, SUM(is_alert::INT) AS alerts,
                   string_agg(DISTINCT CASE WHEN is_alert THEN alert_type END, ',') AS alert_types
            FROM bgl_events WHERE rack IS NOT NULL GROUP BY 1,2,3,4""",
    }
    frames = {}
    for name, sql in q.items():
        df = con.execute(sql).df()
        df.to_csv(EXPORT / f"{name}.csv", index=False)
        frames[name] = df
    return frames


def host_anomaly_concentration(con, alpha=0.05):
    """Per host: distinct blocks that touched it (as transfer src/dst or stored replica)."""
    df = con.execute("""
        WITH touch AS (
            SELECT src_ip ip, block_id FROM hdfs_transfers UNION
            SELECT dst_ip, block_id FROM hdfs_transfers UNION
            SELECT host_ip, block_id FROM hdfs_replicas)
        SELECT ip, COUNT(*) AS blocks, SUM(b.is_anomaly::INT) AS anomalous
        FROM touch JOIN hdfs_blocks b USING(block_id) GROUP BY ip""").df()
    base = con.execute("SELECT AVG(is_anomaly::INT) FROM hdfs_blocks").fetchone()[0]
    m = len(df)
    df["rate"] = df["anomalous"] / df["blocks"]
    df["p_value"] = [binomtest(int(a), int(n), base, alternative="greater").pvalue
                     for a, n in zip(df["anomalous"], df["blocks"])]
    df["significant"] = df["p_value"] < alpha / m
    return df.sort_values("p_value"), base, m


def build_nx(frames):
    G = nx.Graph()
    for r in frames["hosts"].itertuples():
        G.add_node(r.ip, kind="Host", subnet=r.subnet)
    for r in frames["co_replica"].itertuples():
        G.add_edge(r.src_ip, r.dst_ip, blocks=int(r.blocks), anomalous=int(r.anomalous_blocks))
    return G


def hotspot_neighbors(G, conc, k=5):
    """For the most anomaly-concentrated host, its heaviest co-replica peers (blast radius)."""
    top = conc.iloc[0]["ip"]
    nb = sorted(G[top].items(), key=lambda t: (-t[1]["anomalous"], -t[1]["blocks"], t[0]))[:k]  # deterministic ties
    return {"host": top, "degree": G.degree(top),
            "top_peers": [[n, d["blocks"], d["anomalous"]] for n, d in nb]}


def run():
    con = duckdb.connect(str(DB), read_only=True)
    frames = export_tables(con)
    conc, base, m = host_anomaly_concentration(con)
    con_stats = {"same": con.execute("SELECT COUNT(*) FROM hdfs_transfers WHERE src_ip = dst_ip").fetchone()[0]}
    con.close()
    conc.to_csv(OUT / "host_anomaly_concentration.csv", index=False)

    G = build_nx(frames)
    U = G
    deg = sorted(G.degree(weight="blocks"), key=lambda t: -t[1])[:5]
    bgl = frames["bgl_nodes"]
    rack_alerts = bgl.groupby("rack")["alerts"].sum().sort_values(ascending=False)
    res = {
        "hosts": G.number_of_nodes(),
        "co_replica_edges": G.number_of_edges(),
        "rereplication_transfers": len(frames["rereplication"]),
        "rereplication_cross_subnet": int((frames["rereplication"]["src_ip"].str.split(".").str[1]
                                           != frames["rereplication"]["dst_ip"].str.split(".").str[1]).sum()),
        "receiving_lines_src_eq_dest": int(con_stats["same"]),
        "subnets": int(frames["hosts"]["subnet"].nunique()),
        "connected_components": nx.number_connected_components(U),
        "density": round(nx.density(G), 4),
        "top_hosts_by_weighted_degree": [[h, int(d)] for h, d in deg],
        "anomaly_hotspot_neighbors": hotspot_neighbors(G, conc),
        "overall_block_anomaly_rate": round(float(base), 4),
        "hosts_tested": m,
        "hosts_significant_bonferroni": int(conc["significant"].sum()),
        "top_concentration_hosts": conc.head(5)[["ip", "blocks", "anomalous", "rate", "p_value"]]
            .round({"rate": 4}).assign(p_value=lambda d: d.p_value.map(lambda x: float(f"{x:.3g}")))
            .to_dict("records"),
        # expected output of the Neo4j "host_anomaly_rate" query, for parity checking
        "expected_neo4j_host_anomaly_rate": conc[conc.blocks >= 20].assign(rate=lambda d: d.rate.round(4))
            .sort_values(["rate", "blocks"], ascending=False).head(5)[["ip", "blocks", "anomalous", "rate"]]
            .astype({"anomalous": int}).to_dict("records"),
        "bgl_racks": int(bgl["rack"].nunique()),
        "bgl_nodes": len(bgl),
        "bgl_racks_with_alerts": int((rack_alerts > 0).sum()),
        "bgl_top_alert_racks": [[r, int(a)] for r, a in rack_alerts.head(3).items()],
    }
    (OUT / "graph_metrics.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
