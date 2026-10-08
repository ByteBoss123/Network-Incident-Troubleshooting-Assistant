"""Snapshot the real pipeline outputs into one JSON blob for the React dashboard."""
import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
con = duckdb.connect(str(ROOT / "data" / "netincident.duckdb"), read_only=True)

det = json.loads((RES / "detection_metrics.json").read_text())
gf = json.loads((RES / "graph_feature_metrics.json").read_text())
gm = json.loads((RES / "graph_metrics.json").read_text())
sec = json.loads((RES / "security_metrics.json").read_text())
ev = json.loads((RES / "rag_eval_metrics.json").read_text())
llm = {k: json.loads((RES / f"rag_eval_metrics_{k}.json").read_text()) for k in ("nova_pro", "llama3_70b")}

sc = pd.read_csv(RES / "test_scores.csv")
seq = con.execute("""SELECT block_id, string_agg(event_id || 'x' || cnt, ' ' ORDER BY event_id) s
                     FROM hdfs_block_event GROUP BY 1""").df().set_index("block_id")["s"]
exc = {"E7", "E10", "E14", "E27", "E8", "E13"}
t = sc[sc.split == "test"].copy()
t["seq"] = t.block_id.map(seq)
t["exc"] = t.seq.map(lambda s: any(tok.split("x")[0] in exc for tok in s.split()))
thr = gf["risk_threshold_chosen_on_train"]
t["verdict"] = t.exc | t.pca_flag | (t.host_risk >= thr)
blocks = {r.block_id: [int(r.y), int(r.verdict), round(float(r.pca_residual), 2), round(float(r.host_risk), 3), r.seq]
          for r in t.itertuples()}

src = con.execute("""
    WITH f AS (SELECT src_ip, COUNT(*) flows, COUNT(DISTINCT dest_port) ports, SUM(alerted::INT) alerted
               FROM sec_flows WHERE external_src GROUP BY 1),
         a AS (SELECT src_ip, COUNT(*) alerts, mode(signature) sig FROM sec_alerts GROUP BY 1)
    SELECT COALESCE(f.src_ip, a.src_ip) ip, COALESCE(flows,0), COALESCE(ports,0), COALESCE(alerted,0),
           COALESCE(alerts,0), sig FROM f FULL JOIN a ON f.src_ip = a.src_ip""").fetchall()
sources = {ip: [int(f), int(p), int(al), int(a), s] for ip, f, p, al, a, s in src}

hourly = pd.read_csv(RES / "sec_hourly.csv")
hc = pd.read_csv(RES / "host_anomaly_concentration.csv").head(8)

data = {
    "detectors": [{"name": n, **{k: m[k] for k in ("precision", "recall", "f1")}} for n, m in [
        ("Exception-template rule", det["models"]["rule_exception_templates"]),
        ("Isolation Forest", det["models"]["isolation_forest"]),
        ("Logistic regression", det["models"]["logistic_regression"]),
        ("PCA residual", det["models"]["pca_residual"]),
        ("PCA + replica-host risk", gf["test_pca_or_host_risk"])]],
    "test": {"blocks": det["n_test"], "anomalies": det["test_anomalies"], "ceiling": det["count_feature_recall_ceiling"],
             "rescued": gf["test_anomalies_rescued_by_graph"], "added_fp": gf["test_false_alarms_added_by_graph"],
             "threshold": thr},
    "hosts": hc[["ip", "blocks", "anomalous", "rate", "p_value", "significant"]].to_dict("records"),
    "hosts_total": gm["hosts"], "co_replica_edges": gm["co_replica_edges"],
    "peers": gm["anomaly_hotspot_neighbors"],
    "sec": {"etl": sec["etl"], "scanners": sec["graph"]["top_fanout_sources"], "triage": sec["alert_triage"],
            "top_sigs": sec["graph"]["expected_top_signatures"]},
    "hourly": [{"h": str(r.hour)[11:16], "d": str(r.hour)[5:10], "flows": int(r.flows), "alerts": int(r.alerts)}
               for r in hourly.itertuples()],
    "eval": {"det": {"n": ev["n_items"], "acc": ev["overall_accuracy"], "p": ev["block_precision"], "r": ev["block_recall"],
                     "halluc": ev["ungrounded_citations"]},
             **{k: {"n": v["n_items"], "acc": v["overall_accuracy"], "p": v["block_precision"], "r": v["block_recall"],
                    "halluc": v["hallucinated_ids"], "filtered": v["content_filtered_answers"],
                    "p50": v["latency_ms_p50"], "by": {t: x["accuracy"] for t, x in v["by_type"].items()}}
                for k, v in llm.items()}},
    "blocks": blocks, "sources": sources,
}
out = json.dumps(data, separators=(",", ":"), default=str)
(ROOT / "dashboard" / "data.json").write_text(out)
print(len(out), len(blocks), len(sources))
