"""Check the from-scratch algorithms against NetworkX / sorting on the project's real data and time them.

Writes results/algorithms_benchmark.json.
"""
import json
import sys
import time
from pathlib import Path

import duckdb
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from algorithms import (correlate_incidents, correlate_incidents_naive, dijkstra, k_hop_neighbors, path_to,  # noqa: E402
                        top_k)

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "netincident.duckdb"
OUT = ROOT / "results"
WINDOW_S = 1800


def timed(fn, *a, repeat=3, **kw):
    best, out = float("inf"), None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn(*a, **kw)
        best = min(best, time.perf_counter() - t)
    return out, round(best * 1000, 2)


def partition(groups):
    return sorted(sorted(g) for g in groups)


def load_events(con, table, second_key):
    # every alert in this capture targets the same sensor host, so destination IP links everything;
    # alerts are joined on the same source IP or the same signature instead (flows: same destination port)
    rows = con.execute(f"SELECT epoch(timestamp), src_ip, {second_key} FROM {table} ORDER BY timestamp, flow_id").fetchall()
    return [{"t": float(t), "src": s, "k2": k} for t, s, k in rows]


def correlation_check(events, keys=("src", "k2")):
    groups, ms_fast = timed(correlate_incidents, events, WINDOW_S, keys)
    edges, ms_naive = timed(correlate_incidents_naive, events, WINDOW_S, keys, repeat=1)
    G = nx.Graph()
    G.add_nodes_from(range(len(events)))
    G.add_edges_from(edges)
    same = partition(groups) == partition(nx.connected_components(G))
    sizes = sorted((len(g) for g in groups), reverse=True)
    return {"events": len(events), "incidents": len(groups), "largest_incident": sizes[0],
            "singletons": sum(1 for s in sizes if s == 1), "matches_networkx_components": same,
            "sweep_union_find_ms": ms_fast, "all_pairs_ms": ms_naive,
            "speedup": round(ms_naive / max(ms_fast, 1e-3), 1)}


def run():
    con = duckdb.connect(str(DB), read_only=True)
    res = {"window_seconds": WINDOW_S,
           "incident_correlation": {
               "ids_alerts_by_source_or_signature": correlation_check(load_events(con, "sec_alerts", "signature_id")),
               "ids_flows_by_source_or_port": correlation_check(load_events(con, "sec_flows", "dest_port"))}}

    # host topology (co-replica graph), weight = 1 / shared blocks: heavily shared peers are "closer"
    edges = con.execute("""
        SELECT a.host_ip, b.host_ip, COUNT(*) FROM hdfs_replicas a JOIN hdfs_replicas b
        ON a.block_id = b.block_id AND a.host_ip < b.host_ip GROUP BY 1, 2""").fetchall()
    hotspot = con.execute("""
        SELECT r.host_ip FROM hdfs_replicas r JOIN hdfs_blocks b USING(block_id)
        GROUP BY 1 HAVING COUNT(*) >= 20 ORDER BY AVG(b.is_anomaly::INT) DESC, COUNT(*) DESC, 1 LIMIT 1""").fetchone()[0]
    G = nx.Graph()
    adj = {}
    for u, v, n in edges:
        G.add_edge(u, v, weight=1.0 / n)
        adj.setdefault(u, {})[v] = 1.0 / n
        adj.setdefault(v, {})[u] = 1.0 / n

    bfs = {}
    for k in (1, 2):
        mine, ms = timed(k_hop_neighbors, adj, hotspot, k)
        ref = nx.single_source_shortest_path_length(G, hotspot, cutoff=k)
        bfs[f"{k}_hop"] = {"hosts": len(mine) - 1, "matches_networkx": mine == dict(ref), "ms": ms}

    (dist, prev), ms_dij = timed(dijkstra, adj, hotspot)
    ref = nx.single_source_dijkstra_path_length(G, hotspot)
    far = max(dist, key=lambda h: (dist[h], h))
    dij = {"reachable_hosts": len(dist) - 1,
           "matches_networkx": set(dist) == set(ref) and all(abs(dist[h] - ref[h]) < 1e-9 for h in ref),
           "ms": ms_dij, "farthest_host_hops": len(path_to(prev, far)) - 1}

    flows = con.execute("SELECT src_ip, COUNT(DISTINCT dest_port) FROM sec_flows GROUP BY 1").fetchall()
    con.close()
    mine = top_k(flows, 5, key=lambda r: r[1])
    ref = sorted(flows, key=lambda r: -r[1])[:5]
    topk = {"sources": len(flows), "top5_by_distinct_ports": [[s, int(n)] for s, n in mine],
            "matches_full_sort": [r[1] for r in mine] == [r[1] for r in ref]}

    res |= {"topology": {"hotspot_host": hotspot, "hosts": G.number_of_nodes(), "edges": G.number_of_edges(),
                         "bfs_blast_radius": bfs, "dijkstra": dij},
            "top_k_scanners": topk}
    OUT.mkdir(exist_ok=True)
    (OUT / "algorithms_benchmark.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
