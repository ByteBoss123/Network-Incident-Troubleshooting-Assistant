"""Does topology context catch anomalies that look healthy in the logs?

Leakage-free host risk: each host's anomalous-block rate is computed from TRAIN-split blocks only
(chronologically earlier). For a test block, host_risk = max train rate over the hosts storing its
replicas (hosts with < MIN_BLOCKS train blocks get the global train rate). Combined detector:
flag = PCA flag OR host_risk >= RISK_THRESHOLD, where RISK_THRESHOLD is chosen on the train split
(maximising train F1 of the combined rule), never on test.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "netincident.duckdb"
RES = ROOT / "results"
MIN_BLOCKS = 10


def prf(y, f):
    p, r, f1, _ = precision_recall_fscore_support(y, f, average="binary", zero_division=0)
    return {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f1), 4),
            "flagged": int(np.sum(f))}


def run():
    sc = pd.read_csv(RES / "test_scores.csv")
    con = duckdb.connect(str(DB), read_only=True)
    rep = con.execute("SELECT block_id, host_ip FROM hdfs_replicas").df()
    con.close()
    rep = rep.merge(sc[["block_id", "split", "y"]], on="block_id")
    tr = rep[rep.split == "train"]
    g = tr.groupby("host_ip")["y"].agg(["sum", "count"])
    glob = sc.loc[sc.split == "train", "y"].mean()
    g["rate"] = np.where(g["count"] >= MIN_BLOCKS, g["sum"] / g["count"], glob)
    rep["host_rate"] = rep["host_ip"].map(g["rate"]).fillna(glob)
    risk = rep.groupby("block_id")["host_rate"].max()
    sc["host_risk"] = sc["block_id"].map(risk).fillna(glob)  # blocks with no stored replica: global rate

    train, test = sc[sc.split == "train"], sc[sc.split == "test"]
    # NB: host rates were fitted on train labels, so the train-side choice is optimistic; that only
    # affects how the threshold is picked, test blocks never contribute to host rates.
    grid = np.round(np.arange(0.05, 0.95, 0.05), 2)
    best = max(grid, key=lambda t: prf(train.y, train.pca_flag | (train.host_risk >= t))["f1"])
    out = {
        "min_blocks_per_host": MIN_BLOCKS,
        "global_train_anomaly_rate": round(float(glob), 4),
        "risk_threshold_chosen_on_train": float(best),
        "hosts_above_threshold": g.index[(g["rate"] >= best) & (g["count"] >= MIN_BLOCKS)].tolist(),
        "test_pca_only": prf(test.y, test.pca_flag),
        "test_host_risk_only": prf(test.y, test.host_risk >= best),
        "test_pca_or_host_risk": prf(test.y, test.pca_flag | (test.host_risk >= best)),
        "test_anomalies_rescued_by_graph": int(((test.y == 1) & ~test.pca_flag & (test.host_risk >= best)).sum()),
        "test_false_alarms_added_by_graph": int(((test.y == 0) & ~test.pca_flag & (test.host_risk >= best)).sum()),
    }
    sc.to_csv(RES / "test_scores.csv", index=False)
    (RES / "graph_feature_metrics.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
