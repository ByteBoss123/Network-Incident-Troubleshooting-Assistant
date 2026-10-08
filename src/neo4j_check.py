"""Compare the live Neo4j query results with the expected values computed in DuckDB/NetworkX."""
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    got = json.loads((RES / "neo4j_query_results.json").read_text())
    g = json.loads((RES / "graph_metrics.json").read_text())
    s = json.loads((RES / "security_metrics.json").read_text())["graph"]
    checks = {
        "host_anomaly_rate": ([(r["ip"], r["blocks"], r["anomalous"]) for r in got["host_anomaly_rate"]],
                              [(r["ip"], r["blocks"], r["anomalous"]) for r in g["expected_neo4j_host_anomaly_rate"]]),
        "co_replica_edges": (got["graph_counts"][0]["co_replica_edges"], g["co_replica_edges"]),
        "hosts": (got["graph_counts"][0]["hosts"], g["hosts"]),
        "top_rack": (got["rack_alerts"][0]["rack"], g["bgl_top_alert_racks"][0][0]),
        "top_scanner": ((got["top_scanners"][0]["src"], got["top_scanners"][0]["ports"]),
                        (s["top_fanout_sources"][0]["src_ip"], s["top_fanout_sources"][0]["ports"])),
        "top_signature": ((got["top_signatures"][0]["sid"], got["top_signatures"][0]["hits"]),
                          (s["expected_top_signatures"][0]["sid"], s["expected_top_signatures"][0]["hits"])),
    }
    ok = True
    for name, (a, b) in checks.items():
        match = a == b
        ok &= match
        print(f"{'PASS' if match else 'FAIL'} {name}: neo4j={a} expected={b}")
    (RES / "neo4j_parity.json").write_text(json.dumps({k: a == b for k, (a, b) in checks.items()}, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
