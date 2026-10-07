"""Block-level log anomaly detection on real HDFS sessions.

Features: per-block event-template count vector (19 templates) built in SQL.
Split: chronological by block first-seen time (first 70% train, last 30% test),
so the test set is "future" traffic, not a random shuffle.

Models
  rule      : flag a block if it logged any exception/warning template (E7,E10,E14,E27,E8,E13)
  pca       : PCA residual (Xu et al. 2009 style), unsupervised, threshold = train 96th pct
  iforest   : Isolation Forest, unsupervised, threshold = train 96th pct
  logreg    : supervised logistic regression on tf-idf-weighted counts (class_weight balanced)
The 96th percentile is fixed in advance from the ~4% anomaly base rate, not tuned on test labels.
"""
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "netincident.duckdb"
OUT = ROOT / "results"
MODELS = ROOT / "models"
EXCEPTION_EVENTS = ["E7", "E10", "E14", "E27", "E8", "E13"]
SEED = 42


def load_matrix(db=DB):
    con = duckdb.connect(str(db), read_only=True)
    long = con.execute("""
        SELECT b.block_id, b.first_ts, b.is_anomaly, be.event_id, be.cnt
        FROM hdfs_blocks b JOIN hdfs_block_event be USING(block_id)""").df()
    con.close()
    X = long.pivot_table(index="block_id", columns="event_id", values="cnt", fill_value=0, aggfunc="sum")
    meta = long.groupby("block_id").agg(first_ts=("first_ts", "first"), y=("is_anomaly", "first"))
    meta = meta.loc[X.index].sort_values("first_ts")
    X = X.loc[meta.index]
    return X, meta["y"].astype(int).values, meta


def chrono_split(n, frac=0.7):
    cut = int(n * frac)
    return np.arange(cut), np.arange(cut, n)


def metrics(y, score, flag):
    p, r, f, _ = precision_recall_fscore_support(y, flag, average="binary", zero_division=0)
    out = {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4),
           "flagged": int(flag.sum())}
    if len(np.unique(score)) > 2:
        out["roc_auc"] = round(float(roc_auc_score(y, score)), 4)
        out["pr_auc"] = round(float(average_precision_score(y, score)), 4)
    return out


def run():
    X, y, _meta = load_matrix()
    tr, te = chrono_split(len(y))
    Xtr, Xte, ytr, yte = X.values[tr], X.values[te], y[tr], y[te]
    res = {"n_blocks": len(y), "n_train": len(tr), "n_test": len(te),
           "train_anomaly_rate": round(float(ytr.mean()), 4), "test_anomaly_rate": round(float(yte.mean()), 4),
           "test_anomalies": int(yte.sum()), "features": list(X.columns), "models": {}}

    # rule baseline
    cols = [c for c in EXCEPTION_EVENTS if c in X.columns]
    rule_score = X[cols].values[te].sum(1)
    res["models"]["rule_exception_templates"] = metrics(yte, rule_score, rule_score > 0)

    # unsupervised: PCA residual
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    pca = PCA(n_components=0.95, random_state=SEED).fit(Ztr)

    def resid(Z):
        return ((Z - pca.inverse_transform(pca.transform(Z))) ** 2).sum(1)

    thr = np.percentile(resid(Ztr), 96)
    s = resid(Zte)
    res["models"]["pca_residual"] = metrics(yte, s, s > thr) | {"n_components": int(pca.n_components_)}

    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(Xtr)
    s_tr, s_te = -iso.score_samples(Xtr), -iso.score_samples(Xte)
    res["models"]["isolation_forest"] = metrics(yte, s_te, s_te > np.percentile(s_tr, 96))

    # supervised
    idf = np.log((1 + len(Xtr)) / (1 + (Xtr > 0).sum(0))) + 1
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    lr.fit(np.log1p(Xtr) * idf, ytr)
    p = lr.predict_proba(np.log1p(Xte) * idf)[:, 1]
    res["models"]["logistic_regression"] = metrics(yte, p, p >= 0.5)
    coef = sorted(zip(X.columns, lr.coef_[0]), key=lambda t: -abs(t[1]))[:6]
    res["logreg_top_features"] = [[c, round(float(w), 3)] for c, w in coef]

    # error analysis: missed anomalies by the supervised model
    test_ids = X.index[te]
    missed = test_ids[(yte == 1) & (p < 0.5)]
    res["logreg_missed_anomalies"] = len(missed)
    # Recall ceiling: anomalies whose whole count vector equals the most common healthy-block vector
    # cannot be separated by any count-based detector without flagging that healthy majority too.
    key = X.astype(int).astype(str).agg("|".join, axis=1)
    vc = key[y == 0].value_counts()
    modal = vc.index[0]
    te_anom_keys = key.iloc[te][yte == 1]
    res["modal_healthy_vector_share_of_normal"] = round(float(vc.iloc[0] / (y == 0).sum()), 4)
    res["test_anomalies_identical_to_modal_healthy"] = int((te_anom_keys == modal).sum())
    res["count_feature_recall_ceiling"] = round(1 - res["test_anomalies_identical_to_modal_healthy"] / len(te_anom_keys), 4)
    res["logreg_missed_identical_to_modal_healthy"] = int((key.loc[missed] == modal).sum())

    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"lr": lr, "idf": idf, "columns": list(X.columns)}, MODELS / "logreg.joblib")
    joblib.dump({"scaler": sc, "pca": pca, "threshold": float(thr), "columns": list(X.columns)},
                MODELS / "pca_detector.joblib")
    all_resid = resid(sc.transform(X.values))
    scores = pd.DataFrame({"block_id": X.index, "split": np.where(np.isin(np.arange(len(y)), te), "test", "train"),
                           "y": y, "pca_residual": all_resid.round(4), "pca_flag": all_resid > thr})
    scores.to_csv(OUT / "test_scores.csv", index=False)
    (OUT / "detection_metrics.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
