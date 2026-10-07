"""Load the exported topology graph into Neo4j and run the incident-analysis Cypher queries.

Usage (Neo4j 5.x, e.g. Neo4j Desktop/Docker locally or Neo4j AuraDB Free):
    export NEO4J_URI=neo4j+s://<id>.databases.neo4j.io   # or bolt://localhost:7687
    export NEO4J_USER=neo4j NEO4J_PASSWORD=...
    python src/neo4j_load.py

Results are written to results/neo4j_query_results.json so they can be diffed against the
in-process NetworkX/DuckDB results in results/graph_metrics.json.
"""
import json
import os
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
IMP = ROOT / "data" / "neo4j_import"
OUT = ROOT / "results"
BATCH = 2000

SCHEMA = [
    "CREATE CONSTRAINT host_ip IF NOT EXISTS FOR (h:Host) REQUIRE h.ip IS UNIQUE",
    "CREATE CONSTRAINT block_id IF NOT EXISTS FOR (b:Block) REQUIRE b.id IS UNIQUE",
    "CREATE CONSTRAINT subnet_cidr IF NOT EXISTS FOR (s:Subnet) REQUIRE s.cidr IS UNIQUE",
    "CREATE CONSTRAINT bgl_node IF NOT EXISTS FOR (n:ComputeNode) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT rack_id IF NOT EXISTS FOR (r:Rack) REQUIRE r.id IS UNIQUE",
]

LOADS = {
    "hosts": """UNWIND $rows AS r
        MERGE (h:Host {ip: r.ip}) MERGE (s:Subnet {cidr: r.subnet}) MERGE (h)-[:IN_SUBNET]->(s)""",
    "blocks": """UNWIND $rows AS r
        MERGE (b:Block {id: r.block_id})
        SET b.is_anomaly = r.is_anomaly, b.pca_flag = r.pca_flag,
            b.pca_residual = r.pca_residual, b.split = r.split""",
    "stored_on": """UNWIND $rows AS r
        MATCH (b:Block {id: r.block_id}) MATCH (h:Host {ip: r.host_ip}) MERGE (b)-[:STORED_ON]->(h)""",
    "received_by": """UNWIND $rows AS r
        MATCH (b:Block {id: r.block_id}) MATCH (h:Host {ip: r.host_ip}) MERGE (b)-[:RECEIVED_BY]->(h)""",
    "co_replica": """UNWIND $rows AS r
        MATCH (a:Host {ip: r.src_ip}) MATCH (b:Host {ip: r.dst_ip})
        MERGE (a)-[e:CO_REPLICA]-(b) SET e.blocks = r.blocks, e.anomalous = r.anomalous_blocks""",
    "rereplication": """UNWIND $rows AS r
        MATCH (a:Host {ip: r.src_ip}) MATCH (b:Host {ip: r.dst_ip})
        MERGE (a)-[:REREPLICATED {block: r.block_id}]->(b)""",
    "bgl_nodes": """UNWIND $rows AS r
        MERGE (k:Rack {id: r.rack})
        MERGE (m:Midplane {id: r.rack + '-' + r.midplane}) MERGE (k)-[:HAS_MIDPLANE]->(m)
        MERGE (n:ComputeNode {id: r.node})
        SET n.lines = r.lines, n.alerts = r.alerts, n.alert_types = r.alert_types
        MERGE (m)-[:HAS_NODE]->(n)""",
}

QUERIES = {
    # hosts whose stored blocks are disproportionately anomalous
    "host_anomaly_rate": """
        MATCH (b:Block)-[:STORED_ON|RECEIVED_BY]->(h:Host)
        WITH h, b WITH DISTINCT h, b
        WITH h, count(b) AS blocks, sum(CASE WHEN b.is_anomaly THEN 1 ELSE 0 END) AS anomalous
        WHERE blocks >= 20
        RETURN h.ip AS ip, blocks, anomalous, round(toFloat(anomalous)/blocks, 4) AS rate
        ORDER BY rate DESC, blocks DESC LIMIT 5""",
    # blast radius: co-replica peers of the worst host, ranked by shared anomalous blocks
    "blast_radius": """
        MATCH (h:Host {ip: $hotspot})-[e:CO_REPLICA]-(p:Host)
        RETURN p.ip AS peer, e.blocks AS shared_blocks, e.anomalous AS shared_anomalous
        ORDER BY shared_anomalous DESC, shared_blocks DESC LIMIT 5""",
    "graph_counts": """
        MATCH (h:Host) WITH count(h) AS hosts
        MATCH ()-[e:CO_REPLICA]-() WITH hosts, count(e)/2 AS co_replica_edges
        MATCH (b:Block) RETURN hosts, co_replica_edges, count(b) AS blocks""",
    "rack_alerts": """
        MATCH (k:Rack)-[:HAS_MIDPLANE]->()-[:HAS_NODE]->(n:ComputeNode)
        RETURN k.id AS rack, sum(n.alerts) AS alerts ORDER BY alerts DESC LIMIT 3""",
    "shortest_rereplication_path": """
        MATCH (a:Host)-[:REREPLICATED]->(b:Host) WITH a, b LIMIT 1
        MATCH p = shortestPath((a)-[:CO_REPLICA*..4]-(b))
        RETURN a.ip AS src, b.ip AS dst, length(p) AS hops""",
}


def _rows(name):
    df = pd.read_csv(IMP / f"{name}.csv")
    return df.where(pd.notnull(df), None).to_dict("records")


def main():
    uri = os.environ["NEO4J_URI"]
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"])
    with GraphDatabase.driver(uri, auth=auth) as drv:
        drv.verify_connectivity()
        with drv.session() as s:
            for q in SCHEMA:
                s.run(q)
            for name, cypher in LOADS.items():
                rows = _rows(name)
                for i in range(0, len(rows), BATCH):
                    s.execute_write(lambda tx, b=rows[i:i + BATCH], c=cypher: tx.run(c, rows=b).consume())
                print(f"loaded {name}: {len(rows)} rows")
            hotspot = s.run(QUERIES["host_anomaly_rate"]).data()[0]["ip"]
            out = {k: s.run(q, hotspot=hotspot).data() for k, q in QUERIES.items()}
    (OUT / "neo4j_query_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
