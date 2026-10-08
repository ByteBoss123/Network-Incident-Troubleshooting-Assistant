"""Network security telemetry: real Suricata IDS output (Stratosphere Lab sensor, 24 h, 5,000 EVE events).

ETL   : EVE JSON -> DuckDB tables sec_flows (NetFlow-style records), sec_alerts (IDS signatures),
        sec_dns, sec_anomaly.
Graph : external source -> internal destination:port communication edges (exported for Neo4j).
Series: hourly flow / alert counts with a robust z-score burst detector (median/MAD).
Model : alert triage from flow metadata only. Predict whether an inbound flow trips any Suricata
        signature (flow.alerted) from ports, protocol, packets, bytes, duration and state, with no
        IP addresses or reputation lists as features. Chronological 70/30 split by flow start time.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "netsec" / "test6-malicious.suricata.json"
DB = ROOT / "data" / "netincident.duckdb"
RES = ROOT / "results"
EXPORT = ROOT / "data" / "neo4j_import"
SEED = 42
SENSITIVE_PORTS = {22, 23, 445, 1433, 3389, 2375, 5900}


def load_events():
    ev = [json.loads(line) for line in RAW.open()]
    flows, alerts, dns, anom = [], [], [], []
    for e in ev:
        base = {k: e.get(k) for k in ("timestamp", "flow_id", "src_ip", "src_port", "dest_ip", "dest_port", "proto")}
        t = e["event_type"]
        if t == "flow":
            f = e["flow"]
            flows.append(base | {"app_proto": e.get("app_proto"), "pkts_toserver": f["pkts_toserver"],
                                 "pkts_toclient": f["pkts_toclient"], "bytes_toserver": f["bytes_toserver"],
                                 "bytes_toclient": f["bytes_toclient"], "start": f["start"], "end": f["end"],
                                 "state": f["state"], "reason": f.get("reason"), "alerted": bool(f["alerted"])})
        elif t == "alert":
            a = e["alert"]
            alerts.append(base | {"signature_id": a["signature_id"], "signature": a["signature"],
                                  "category": a["category"], "severity": a["severity"]})
        elif t == "dns":
            dns.append(base | {"dns_type": e["dns"].get("type"), "rrname": e["dns"].get("rrname")})
        elif t == "anomaly":
            anom.append(base | {"anomaly_event": e["anomaly"].get("event")})
    return len(ev), pd.DataFrame(flows), pd.DataFrame(alerts), pd.DataFrame(dns), pd.DataFrame(anom)


def etl(con):
    n, flows, alerts, dns, anom = load_events()
    for c in ("timestamp", "start", "end"):
        flows[c] = pd.to_datetime(flows[c], utc=True)
    for df in (alerts, dns, anom):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    flows["external_src"] = ~flows.src_ip.str.startswith(("192.168.", "10.", "172.16."))
    for name, df in {"sec_flows": flows, "sec_alerts": alerts, "sec_dns": dns, "sec_anomaly": anom}.items():
        con.register("tmp_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
    return {
        "eve_events": n, "flows": len(flows), "alerts": len(alerts), "dns": len(dns), "anomalies": len(anom),
        "alerted_flows": int(flows.alerted.sum()),
        "external_source_ips": int(flows.loc[flows.external_src, "src_ip"].nunique()),
        "alert_source_ips": int(alerts.src_ip.nunique()),
        "distinct_signatures": int(alerts.signature_id.nunique()),
        "window_start_utc": str(flows.start.min()), "window_end_utc": str(flows.start.max()),
        "dup_flow_ids": int(flows.flow_id.duplicated().sum()),
    }


def graph_export(con):
    EXPORT.mkdir(parents=True, exist_ok=True)
    edges = con.execute("""
        SELECT src_ip, dest_ip, dest_port, proto, COUNT(*) AS flows, SUM(alerted::INT) AS alerted_flows,
               SUM(pkts_toserver) AS pkts, SUM(bytes_toserver) AS bytes
        FROM sec_flows GROUP BY 1,2,3,4""").df()
    edges.to_csv(EXPORT / "sec_flow_edges.csv", index=False)
    sig = con.execute("""
        SELECT src_ip, signature_id, any_value(signature) AS signature, any_value(category) AS category,
               COUNT(*) AS hits FROM sec_alerts GROUP BY 1,2""").df()
    sig.to_csv(EXPORT / "sec_alert_edges.csv", index=False)
    fan = con.execute("""
        SELECT src_ip, COUNT(DISTINCT dest_port) AS ports, COUNT(DISTINCT dest_ip) AS dests, COUNT(*) AS flows
        FROM sec_flows WHERE external_src GROUP BY 1 ORDER BY ports DESC, flows DESC, src_ip LIMIT 5""").df()
    top_sig = (sig.groupby("signature_id").agg(signature=("signature", "first"), sources=("src_ip", "count"),
                                              hits=("hits", "sum"))
               .sort_values("hits", ascending=False).head(5).reset_index())
    return {"flow_edges": len(edges), "alert_edges": len(sig),
            "expected_top_signatures": top_sig.rename(columns={"signature_id": "sid"}).to_dict("records"),
            "top_fanout_sources": fan.to_dict("records")}


def time_series(con, z=3.5):
    s = con.execute("""
        WITH f AS (SELECT date_trunc('hour', start) h, COUNT(*) flows FROM sec_flows GROUP BY 1),
             a AS (SELECT date_trunc('hour', timestamp) h, COUNT(*) alerts FROM sec_alerts GROUP BY 1)
        SELECT COALESCE(f.h, a.h) AS hour, COALESCE(flows, 0) AS flows, COALESCE(alerts, 0) AS alerts
        FROM f FULL JOIN a ON f.h = a.h ORDER BY 1""").df()
    med = s.alerts.median()
    mad = (s.alerts - med).abs().median() or 1.0
    s["robust_z"] = 0.6745 * (s.alerts - med) / mad
    s["burst"] = s.robust_z > z
    s.to_csv(RES / "sec_hourly.csv", index=False)
    return {"hours": len(s), "alerts_per_hour_median": float(med), "burst_hours": int(s.burst.sum()),
            "bursts": s[s.burst].assign(hour=lambda d: d.hour.astype(str))[["hour", "alerts", "robust_z"]]
            .round(2).to_dict("records")}


def features(df):
    X = pd.DataFrame({
        "dest_port": df.dest_port, "src_port": df.src_port,
        "sensitive_port": df.dest_port.isin(SENSITIVE_PORTS).astype(int),
        "tcp": (df.proto == "TCP").astype(int),
        "pkts_toserver": df.pkts_toserver, "pkts_toclient": df.pkts_toclient,
        "bytes_toserver": df.bytes_toserver, "bytes_toclient": df.bytes_toclient,
        "duration_s": (df.end - df.start).dt.total_seconds(),
        "no_reply": (df.pkts_toclient == 0).astype(int),
    })
    for st in ("new", "established", "closed"):
        X[f"state_{st}"] = (df.state == st).astype(int)
    for ap in ("failed", "http", "dns"):
        X[f"app_{ap}"] = (df.app_proto == ap).astype(int)
    return X


def score(y, s, flag):
    p, r, f, _ = precision_recall_fscore_support(y, flag, average="binary", zero_division=0)
    return {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4),
            "roc_auc": round(float(roc_auc_score(y, s)), 4), "pr_auc": round(float(average_precision_score(y, s)), 4)}


def model(con):
    df = con.execute("SELECT * FROM sec_flows WHERE external_src ORDER BY start").df()
    X, y = features(df), df.alerted.astype(int).values
    cut = int(len(df) * 0.7)
    Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:]
    out = {"n_flows": len(df), "n_train": cut, "n_test": len(df) - cut,
           "train_alert_rate": round(float(ytr.mean()), 4), "test_alert_rate": round(float(yte.mean()), 4),
           "models": {}}
    rule = Xte.sensitive_port.values
    out["models"]["rule_sensitive_port"] = score(yte, rule + 1e-9 * Xte.pkts_toserver.values, rule == 1)
    lr = LogisticRegression(max_iter=3000, class_weight="balanced")
    lr.fit(np.log1p(Xtr.clip(lower=0)), ytr)
    p = lr.predict_proba(np.log1p(Xte.clip(lower=0)))[:, 1]
    out["models"]["logistic_regression"] = score(yte, p, p >= 0.5)
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED,
                                        categorical_features=None)
    gb.fit(Xtr, ytr)
    g = gb.predict_proba(Xte)[:, 1]
    # operating threshold chosen on the last 20% of TRAIN (validation), not on test
    vcut = int(cut * 0.8)
    gb_v = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)
    gb_v.fit(Xtr.iloc[:vcut], ytr[:vcut])
    pv = gb_v.predict_proba(Xtr.iloc[vcut:])[:, 1]
    grid = np.linspace(0.05, 0.95, 19)
    thr = float(max(grid, key=lambda t: precision_recall_fscore_support(
        ytr[vcut:], pv >= t, average="binary", zero_division=0)[2]))
    out["models"]["gradient_boosting"] = score(yte, g, g >= thr) | {"threshold_from_validation": round(thr, 2)}
    # what does the model miss? break test misses down by signature category
    test = df.iloc[cut:].assign(p=g, flag=g >= thr)
    missed = test[(test.alerted) & (~test.flag)]
    cats = con.execute("""SELECT flow_id, any_value(category) AS category FROM sec_alerts GROUP BY 1""").df()
    m = missed.merge(cats, on="flow_id", how="left")
    # Reputation-list signatures fire on WHO the source is, not on behaviour visible in flow metadata.
    rep = con.execute("""SELECT DISTINCT flow_id FROM sec_alerts
        WHERE signature ILIKE '%reputation%' OR signature ILIKE '%CINS%' OR signature ILIKE '%DROP%'
           OR signature ILIKE '%Dshield%' OR signature ILIKE '%Compromised%'""").df().flow_id
    al = test[test.alerted]
    is_rep = al.flow_id.isin(rep)
    out["test_alerted_reputation_only_share"] = round(float(is_rep.mean()), 4)
    out["recall_on_reputation_alerts"] = round(float(al[is_rep].flag.mean()), 4) if is_rep.any() else None
    out["recall_on_behavioural_alerts"] = round(float(al[~is_rep].flag.mean()), 4) if (~is_rep).any() else None
    out["test_missed_alerted_flows"] = len(missed)
    out["missed_by_category"] = m.category.fillna("no linked alert record").value_counts().to_dict()
    test[["flow_id", "start", "src_ip", "dest_port", "alerted", "p", "flag"]].assign(
        start=lambda d: d.start.astype(str)).to_csv(RES / "sec_test_scores.csv", index=False)
    return out


def run():
    con = duckdb.connect(str(DB))
    res = {"etl": etl(con), "graph": graph_export(con), "time_series": time_series(con),
           "alert_triage": model(con)}
    con.close()
    (RES / "security_metrics.json").write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
