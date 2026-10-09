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


def _snap(rules, enis, lbs=()):
    return {"regions": [{"region": "r1", "vpcs": [{"default": True}],
                         "security_groups": [{"id": "sg-a", "name": "web", "rules": rules},
                                             {"id": "sg-old", "name": "stale", "rules": []}],
                         "enis": enis, "load_balancers": list(lbs)}]}


def test_netconfig_audit_flags_exposed_probed_port_and_plaintext_lb():
    import netconfig_audit as na
    rules = [{"proto": "tcp", "from": 22, "to": 22, "cidrs": ["0.0.0.0/0"], "sg_refs": [], "prefix_lists": []}]
    enis = [{"id": "e1", "type": "interface", "public_ip": True, "sgs": ["sg-a"]},
            {"id": "e2", "type": "ecs_task", "public_ip": True, "sgs": ["sg-internal"]}]
    lbs = [{"name": "lb", "scheme": "internet-facing", "listeners": [{"protocol": "HTTP", "port": 80}]}]
    out = na.audit(_snap(rules, enis, lbs), {22: {"flows": 5, "alerted": 3, "sources": 2},
                                             3389: {"flows": 4, "alerted": 2, "sources": 2}})
    checks = [f["check"] for f in out["findings"]]
    assert "probed_port_exposed" in checks and "plaintext_listener" in checks
    assert [f["eni"] for f in out["findings"] if f["check"] == "unneeded_public_ip"] == ["e2"]
    assert any(f["check"] == "unattached_sg" and "sg-old" in f["sg"] for f in out["findings"])
    assert out["summary"]["top10_alerted_ports_exposed"] == [22]


def test_netconfig_audit_sg_reference_only_rule_is_not_internet_exposed():
    import netconfig_audit as na
    rules = [{"proto": "tcp", "from": 8000, "to": 8000, "cidrs": [], "sg_refs": ["sg-lb"], "prefix_lists": []}]
    out = na.audit(_snap(rules, [{"id": "e1", "type": "interface", "public_ip": False, "sgs": ["sg-a"]}]),
                   {8000: {"flows": 1, "alerted": 1, "sources": 1}})
    assert out["summary"]["internet_exposed_rules"] == 0


def test_local_quantization_results_rescore():
    import llm_eval
    saved = json.loads((RES / "local_inference_optimization.json").read_text())
    for q in ("fp16", "q8_0", "q4_k_m"):
        m = llm_eval.score(RES / "local_opt_greedy" / f"answers_gguf_{q}.jsonl", tag=f"local_gguf_{q}",
                           prompts_path=RES / "llm_prompts_bedrock.jsonl")
        assert m["overall_accuracy"] == saved[f"gguf_{q}"]["accuracy"]
    assert saved["gguf_q8_0"]["items_differing_from_fp16"] == 0


def test_judge_handles_negated_answers():
    import llm_eval
    row = {"type": "source", "truth": False}
    assert llm_eval.judge(row, "No, there is no evidence of traffic from 1.2.3.4 being malicious.")[0]
    assert llm_eval.judge({"type": "source", "truth": True}, "Verdict: 1.2.3.4 is FLAGGED")[0]
    assert llm_eval.judge({"type": "block", "truth": 0}, "Verdict: No anomaly detected.")[0]


# ---- from-scratch algorithms and the detector interface ----
import random

import networkx as nx

from algorithms import UnionFind, correlate_incidents, correlate_incidents_naive, dijkstra, k_hop_neighbors, top_k


def test_union_find_basic():
    uf = UnionFind(5)
    assert uf.union(0, 1) and uf.union(3, 4) and not uf.union(1, 0)
    assert uf.sets == 3 and uf.find(0) == uf.find(1) and uf.find(2) != uf.find(3)


def test_incident_correlation_matches_all_pairs_on_random_events():
    rng = random.Random(0)
    for _ in range(20):
        ev = [{"t": rng.uniform(0, 5000), "src": rng.choice("abcde"), "k2": rng.randrange(6)} for _ in range(120)]
        G = nx.Graph()
        G.add_nodes_from(range(len(ev)))
        G.add_edges_from(correlate_incidents_naive(ev, 300, ("src", "k2")))
        fast = sorted(sorted(g) for g in correlate_incidents(ev, 300, ("src", "k2")))
        assert fast == sorted(sorted(c) for c in nx.connected_components(G))


def test_graph_algorithms_match_networkx_on_random_graphs():
    rng = random.Random(1)
    for _ in range(10):
        G = nx.gnm_random_graph(60, 150, seed=rng.randrange(10**6))
        for u, v in G.edges:
            G[u][v]["weight"] = rng.uniform(0.1, 5)
        adj = {n: {m: G[n][m]["weight"] for m in G[n]} for n in G}
        assert k_hop_neighbors(adj, 0, 2) == dict(nx.single_source_shortest_path_length(G, 0, cutoff=2))
        dist, _ = dijkstra(adj, 0)
        ref = nx.single_source_dijkstra_path_length(G, 0)
        assert set(dist) == set(ref) and all(abs(dist[n] - ref[n]) < 1e-9 for n in ref)


def test_top_k_matches_sort():
    rng = random.Random(2)
    xs = [rng.randrange(1000) for _ in range(500)]
    assert top_k(xs, 7, key=lambda x: x) == sorted(xs, reverse=True)[:7]


def test_algorithm_benchmark_results_on_real_data():
    r = json.loads((RES / "algorithms_benchmark.json").read_text())
    for v in r["incident_correlation"].values():
        assert v["matches_networkx_components"]
    assert r["incident_correlation"]["ids_alerts_by_source_or_signature"]["incidents"] == 388
    assert all(v["matches_networkx"] for v in r["topology"]["bfs_blast_radius"].values())
    assert r["topology"]["dijkstra"]["matches_networkx"] and r["top_k_scanners"]["matches_full_sort"]


def test_detector_interface_reproduces_published_metrics():
    from detectors import evaluate
    ref = json.loads((RES / "detection_metrics.json").read_text())["models"]
    for name, m in evaluate().items():
        assert {k: ref[name][k] for k in m} == m, name
