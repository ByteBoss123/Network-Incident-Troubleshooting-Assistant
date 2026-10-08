"""Golden-set evaluation of the troubleshooting assistant.

Golden set (built from held-out data and analytic ground truth, seeded):
  block   : 40 anomalous + 40 normal TEST-split blocks, 4 phrasings; truth = Loghub label
  host    : host-concentration questions; truth = Bonferroni-significant hosts
  rack    : BGL rack questions; truth = rack with most alert lines
  template: event-meaning questions; truth = template id must be retrieved
  missing : block ids absent from the logs; truth = answer must say not found
Metrics: block verdict precision/recall/accuracy, task accuracy per type, citation grounding
(every [id] cited in the answer must exist in the structured context or retrieved docs), latency.
"""
import json
import random
import re
import statistics
import time
from pathlib import Path

import pandas as pd

import rag

RES = Path(__file__).resolve().parents[1] / "results"
SEED = 7
PHRASINGS = ["Why did block {b} fail?", "Is {b} healthy?", "Troubleshoot {b}",
             "Investigate an incident on block {b}; is it anomalous?"]
TEMPLATES = {"E7": "What does a writeBlock received exception mean?",
             "E13": "What does receiving an empty packet for a block indicate?",
             "E14": "Exception in receiveBlock - is that serious?",
             "E21": "What happens when a block file is deleted?"}
CITE = re.compile(r"\[([^\]]+)\]")


def golden():
    rnd = random.Random(SEED)
    sc = pd.read_csv(RES / "test_scores.csv")
    t = sc[sc.split == "test"]
    a = rnd.sample(sorted(t[t.y == 1].block_id), 40)
    n = rnd.sample(sorted(t[t.y == 0].block_id), 40)
    items = [{"type": "block", "q": rnd.choice(PHRASINGS).format(b=b), "block": b, "truth": 1} for b in a]
    items += [{"type": "block", "q": rnd.choice(PHRASINGS).format(b=b), "block": b, "truth": 0} for b in n]
    hc = pd.read_csv(RES / "host_anomaly_concentration.csv")
    sig = sorted(hc[hc.significant].ip)
    items += [{"type": "host", "q": q, "truth": sig} for q in
              ["Which hosts have elevated anomaly rates?", "Which datanodes are anomaly hotspots?",
               "Are any hosts responsible for a disproportionate share of failures?"]]
    items += [{"type": "host_one", "q": f"How risky is host {ip}?", "ip": ip,
               "truth": bool(hc.set_index("ip").loc[ip, "significant"])}
              for ip in sig + rnd.sample(sorted(hc[~hc.significant].ip), 3)]
    items += [{"type": "rack", "q": q, "truth": "R30"} for q in
              ["Which rack has the most BGL alerts?", "Where are the BGL hardware alerts concentrated?"]]
    items += [{"type": "template", "q": q, "truth": e} for e, q in TEMPLATES.items()]
    items += [{"type": "missing", "q": f"Why did block {b} fail?", "truth": "not found"}
              for b in ["blk_1234567890123", "blk_-999999999999"]]
    items += security_items()
    rnd.shuffle(items)
    return items


SIGNATURES = {"2010935": "What does the MSSQL port 1433 inbound scan signature mean?",
              "2001219": "Explain the potential SSH scan alert",
              "2017515": "What is the python-requests user-agent alert?"}


def security_items():
    """Suricata questions: per-source IDS verdicts, widest scanner, signature lookups."""
    rnd = random.Random(SEED + 1)
    con = rag._con()
    alerted = [r[0] for r in con.execute("SELECT DISTINCT src_ip FROM sec_alerts ORDER BY 1").fetchall()]
    clean = [r[0] for r in con.execute("""SELECT DISTINCT src_ip FROM sec_flows WHERE external_src
              AND src_ip NOT IN (SELECT src_ip FROM sec_alerts) ORDER BY 1""").fetchall()]
    items = [{"type": "source", "q": f"Is {ip} malicious?", "truth": True} for ip in rnd.sample(alerted, 6)]
    items += [{"type": "source", "q": f"Should I worry about traffic from {ip}?", "truth": False}
              for ip in rnd.sample(clean, 6)]
    top = json.loads((RES / "security_metrics.json").read_text())["graph"]["top_fanout_sources"][0]["src_ip"]
    items += [{"type": "scanner", "q": q, "truth": top} for q in
              ["Which source IPs are scanning us?", "Who are the top attackers probing our ports?"]]
    items += [{"type": "signature", "q": q, "truth": sid} for sid, q in SIGNATURES.items()]
    return items


def grounded(answer, ctx, retrieved):
    allowed = json.dumps(ctx) + " " + " ".join(retrieved)
    cites = CITE.findall(answer)
    bad = [c for c in cites if c not in allowed]
    return len(cites), bad


def score(item, out):
    ans = out["answer"]
    if item["type"] == "block":
        first = ans.strip().split("\n")[0].lower()
        pred = int("anomal" in first and not re.search(r"\bnormal\b|healthy", first))
        return pred == item["truth"], pred
    if item["type"] == "host":
        flagged = set(re.findall(r"\[(\d+\.\d+\.\d+\.\d+)\][^\n]*\(significant\)", ans))
        return flagged == set(item["truth"]), None
    if item["type"] == "host_one":
        said = "(significant)" in ans
        return said == item["truth"], None
    if item["type"] == "rack":
        return f"[{item['truth']}]" in ans.split("\n")[0], None
    if item["type"] == "template":
        return f"template:{item['truth']}" in out["retrieved"], None
    if item["type"] == "source":
        return ("FLAGGED" in ans.split("\n")[0]) == item["truth"], None
    if item["type"] == "scanner":
        return f"[{item['truth']}]" in ans.split("\n")[0], None
    if item["type"] == "signature":
        return f"signature:{item['truth']}" in out["retrieved"], None
    if item["type"] == "missing":
        return "does not appear" in ans or "not contain" in ans, None


def run():
    chain, backend = rag.build_chain()
    items = golden()
    rows, lat = [], []
    for it in items:
        t0 = time.perf_counter()
        out = rag.ask(it["q"], chain)
        lat.append((time.perf_counter() - t0) * 1000)
        ok, pred = score(it, out)
        n_cites, bad = grounded(out["answer"], out["context"], out["retrieved"])
        rows.append({**it, "ok": bool(ok), "pred": pred, "n_citations": n_cites, "ungrounded": bad,
                     "answer": out["answer"]})
    df = pd.DataFrame(rows)
    b = df[df.type == "block"]
    tp = int(((b.pred == 1) & (b.truth == 1)).sum())
    res = {
        "backend": backend,
        "n_items": len(df),
        "by_type": {t: {"n": len(g), "accuracy": round(float(g.ok.mean()), 4)} for t, g in df.groupby("type")},
        "block_precision": round(tp / max(1, int((b.pred == 1).sum())), 4),
        "block_recall": round(tp / max(1, int((b.truth == 1).sum())), 4),
        "overall_accuracy": round(float(df.ok.mean()), 4),
        "citations_total": int(df.n_citations.sum()),
        "ungrounded_citations": int(df.ungrounded.map(len).sum()),
        "latency_ms_p50": round(statistics.median(lat), 1),
        "latency_ms_p95": round(sorted(lat)[int(0.95 * len(lat)) - 1], 1),
    }
    df.to_json(RES / "rag_eval_items.jsonl", orient="records", lines=True)
    (RES / "rag_eval_metrics.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
