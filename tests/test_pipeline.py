import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"

import rag


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect(str(ROOT / "data" / "netincident.duckdb"), read_only=True)
    yield c
    c.close()


def test_every_event_line_has_a_labelled_block(con):
    orphans = con.execute("SELECT COUNT(*) FROM hdfs_events e LEFT JOIN hdfs_blocks b USING(block_id) "
                          "WHERE b.block_id IS NULL").fetchone()[0]
    assert orphans == 0


def test_block_ids_unique(con):
    n, d = con.execute("SELECT COUNT(*), COUNT(DISTINCT block_id) FROM hdfs_blocks").fetchone()
    assert n == d == 7940


def test_split_is_chronological(con):
    sc = pd.read_csv(RES / "test_scores.csv")
    ts = con.execute("SELECT block_id, first_ts FROM hdfs_blocks").df().set_index("block_id")["first_ts"]
    sc["ts"] = sc.block_id.map(ts)
    assert sc[sc.split == "train"].ts.max() <= sc[sc.split == "test"].ts.min()


def test_retrieval_corpus_has_no_test_blocks():
    sc = pd.read_csv(RES / "test_scores.csv")
    test_ids = set(sc[sc.split == "test"].block_id)
    inc = {d.metadata["id"] for d in rag.knowledge_base() if d.metadata["kind"] == "incident"}
    assert inc and not (inc & test_ids)


def test_host_rates_use_train_only():
    m = json.loads((RES / "graph_feature_metrics.json").read_text())
    assert m["risk_threshold_chosen_on_train"] > 0
    assert m["test_pca_or_host_risk"]["recall"] >= m["test_pca_only"]["recall"]


def test_exception_block_flagged():
    sc = pd.read_csv(RES / "test_scores.csv")
    con = duckdb.connect(str(ROOT / "data" / "netincident.duckdb"), read_only=True)
    b = con.execute("SELECT block_id FROM hdfs_events WHERE event_id = 'E7' LIMIT 1").fetchone()[0]
    con.close()
    out = rag.ask(f"Why did block {b} fail?")
    assert "LIKELY ANOMALOUS" in out["answer"] and "[E7]" in out["answer"]
    assert b in set(sc.block_id)


def test_unknown_block_says_not_found():
    assert "does not appear" in rag.ask("Why did block blk_42 fail?")["answer"]


def test_answers_only_cite_context():
    from evaluate import grounded
    out = rag.ask("Which hosts have elevated anomaly rates?")
    n, bad = grounded(out["answer"], out["context"], out["retrieved"])
    assert n > 0 and bad == []


def test_api():
    from fastapi.testclient import TestClient

    import api
    c = TestClient(api.app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/ask", json={"question": "Which rack has the most BGL alerts?"}).json()
    assert "[R30]" in r["answer"]
    assert c.get("/block/blk_42").status_code == 404


def test_security_etl_counts(con):
    flows, alerted, alerts = con.execute(
        "SELECT (SELECT COUNT(*) FROM sec_flows), (SELECT SUM(alerted::INT) FROM sec_flows), "
        "(SELECT COUNT(*) FROM sec_alerts)").fetchone()
    assert (flows, alerted, alerts) == (3512, 366, 745)


def test_security_split_chronological_and_no_ip_features():
    import security
    m = json.loads((RES / "security_metrics.json").read_text())["alert_triage"]
    assert m["n_train"] + m["n_test"] == m["n_flows"]
    cols = security.features(pd.DataFrame({
        "dest_port": [1], "src_port": [2], "proto": ["TCP"], "pkts_toserver": [1], "pkts_toclient": [0],
        "bytes_toserver": [1], "bytes_toclient": [0], "start": pd.to_datetime(["2021-01-01"]),
        "end": pd.to_datetime(["2021-01-01"]), "state": ["new"], "app_proto": [None]})).columns
    assert not any("ip" in c for c in cols)


def test_source_question_routes_to_ids():
    out = rag.ask("Is 193.46.255.92 malicious?")
    assert out["answer"].startswith("Verdict: source [193.46.255.92] was FLAGGED")


def test_bedrock_llm_answers_rescore_reproducibly():
    import llm_eval
    for tag, path in [("nova_pro", "llm_answers_nova-pro.jsonl"), ("llama3_70b", "llm_answers_llama3.jsonl"),
                      ("local_qwen1_5b", "llm_answers_local_qwen.jsonl")]:
        saved = json.loads((RES / f"rag_eval_metrics_{tag}.json").read_text())
        again = llm_eval.score(RES / path, tag=tag, prompts_path=RES / "llm_prompts_bedrock.jsonl")
        assert again["overall_accuracy"] == saved["overall_accuracy"]
        assert again["hallucinated_ids"] == 0
