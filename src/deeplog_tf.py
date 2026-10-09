"""DeepLog-style LSTM log anomaly detector in TensorFlow/Keras (Du et al., CCS 2017).

An LSTM learns the next log event of *normal* HDFS block sessions from the previous `WINDOW` events.
At inference a block is anomalous if any of its events falls outside the model's top-g predictions
(or was never seen in training). Same chronological 70/30 block split as the PCA detector
(results/test_scores.csv), so the test F1 is directly comparable to PCA's 0.760.

Leakage control: the model trains only on normal blocks from the first 80% of the training window;
g is chosen by F1 on the last 20% of the training window (validation), never on test.
"""
import json
import os
import random
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "netincident.duckdb"
RES = ROOT / "results"
WINDOW = 10
SEED = 42


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.experimental.enable_op_determinism()


def load_sequences():
    con = duckdb.connect(str(DB), read_only=True)
    ev = con.execute("SELECT block_id, event_id FROM hdfs_events ORDER BY block_id, line_id").df()
    con.close()
    seqs = ev.groupby("block_id", sort=False)["event_id"].apply(list)
    split = pd.read_csv(RES / "test_scores.csv")  # block order + split from the PCA run
    return seqs, split


def windows(seq_ids):
    """(history, next) pairs; history left-padded with 0."""
    X, y = [], []
    padded = [0] * WINDOW + seq_ids
    for k in range(len(seq_ids)):
        X.append(padded[k:k + WINDOW])
        y.append(seq_ids[k])
    return X, y


def block_scores(model, seqs_ids, vocab_size):
    """Per block: worst rank of the true next event (0 = top-1) and min probability; unseen event -> inf rank."""
    allX, owner, truth = [], [], []
    for b, s in enumerate(seqs_ids):
        X, y = windows(s)
        allX += X
        owner += [b] * len(X)
        truth += y
    probs = model.predict(np.array(allX, dtype="int32"), batch_size=4096, verbose=0)
    truth = np.array(truth)
    p_true = np.where(truth < vocab_size, probs[np.arange(len(truth)), np.minimum(truth, vocab_size - 1)], 0.0)
    rank = (probs > p_true[:, None]).sum(axis=1)
    rank = np.where(truth == vocab_size, 10**6, rank)  # unknown event
    df = pd.DataFrame({"b": owner, "rank": rank, "p": p_true})
    agg = df.groupby("b").agg(max_rank=("rank", "max"), min_p=("p", "min"))
    return agg.reindex(range(len(seqs_ids))).fillna({"max_rank": 0, "min_p": 1.0})


def prf(y, flag):
    p, r, f, _ = precision_recall_fscore_support(y, flag, average="binary", zero_division=0)
    return round(float(p), 4), round(float(r), 4), round(float(f), 4)


def run():
    set_seeds()
    t0 = time.time()
    seqs, split = load_sequences()
    split = split[split.block_id.isin(seqs.index)].reset_index(drop=True)
    tr = split[split.split == "train"].reset_index(drop=True)
    te = split[split.split == "test"].reset_index(drop=True)
    vcut = int(len(tr) * 0.8)
    fit_blocks, val_blocks = tr.iloc[:vcut], tr.iloc[vcut:]

    # vocabulary from normal training blocks only; id 0 = padding, unseen events -> vocab_size
    normal_fit = fit_blocks[fit_blocks.y == 0].block_id
    events = sorted({e for b in normal_fit for e in seqs[b]})
    vocab = {e: i + 1 for i, e in enumerate(events)}
    V = len(vocab) + 1

    def ids(block_ids):
        return [[vocab.get(e, V) for e in seqs[b]] for b in block_ids]

    X, Y = [], []
    for s in ids(normal_fit):
        x, y = windows(s)
        X += x
        Y += y
    X, Y = np.array(X, dtype="int32"), np.array(Y, dtype="int32")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW,)),
        tf.keras.layers.Embedding(V + 1, 32),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(V, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    hist = model.fit(X, Y, epochs=8, batch_size=512, validation_split=0.1, verbose=0, shuffle=True)

    val_ids, te_ids = ids(val_blocks.block_id), ids(te.block_id)
    sv, st = block_scores(model, val_ids, V), block_scores(model, te_ids, V)
    yv, yt = val_blocks.y.values, te.y.values

    # choose g (top-g candidates) on validation only
    grid = range(1, min(V, 12))
    best_g = max(grid, key=lambda g: prf(yv, sv.max_rank.values >= g)[2])
    flag = st.max_rank.values >= best_g
    p, r, f = prf(yt, flag)
    score = -np.log(np.clip(st.min_p.values, 1e-12, 1))
    pca = json.loads((RES / "detection_metrics.json").read_text())
    pca_test = pca.get("models", pca).get("pca_residual", {}) if isinstance(pca, dict) else {}

    # overlap with PCA on test anomalies
    pca_flag = te.block_id.map(pd.read_csv(RES / "test_scores.csv").set_index("block_id")["pca_flag"]).values \
        if "pca_flag" in pd.read_csv(RES / "test_scores.csv").columns else None
    out = {
        "model": "DeepLog-style 2-layer LSTM (TensorFlow/Keras), window 10, embedding 32, 64 units",
        "tensorflow": tf.__version__,
        "train_windows_normal_only": int(len(X)), "vocab_events": len(vocab),
        "final_train_next_event_acc": round(float(hist.history["accuracy"][-1]), 4),
        "final_holdout_next_event_acc": round(float(hist.history["val_accuracy"][-1]), 4),
        "g_chosen_on_validation": int(best_g),
        "n_test_blocks": int(len(te)), "n_test_anomalies": int(yt.sum()),
        "test": {"precision": p, "recall": r, "f1": f,
                 "roc_auc": round(float(roc_auc_score(yt, score)), 4),
                 "pr_auc": round(float(average_precision_score(yt, score)), 4)},
        "pca_reference_test": pca_test,
        "seconds": round(time.time() - t0, 1),
    }
    if pca_flag is not None:
        pf = pca_flag.astype(bool)
        an = yt == 1
        out["test_anomalies_caught"] = {"lstm": int((flag & an).sum()), "pca": int((pf & an).sum()),
                                        "both": int((flag & pf & an).sum()),
                                        "lstm_or_pca": int(((flag | pf) & an).sum())}
        out["test_false_alarms"] = {"lstm": int((flag & ~an).sum()), "pca": int((pf & ~an).sum())}
        p2, r2, f2 = prf(yt, flag | pf)
        out["test_lstm_or_pca"] = {"precision": p2, "recall": r2, "f1": f2}
    (RES / "deeplog_tf_metrics.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
