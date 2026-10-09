"""Evaluate a real LLM as the assistant's generator on the same golden set.

  python src/llm_eval.py export   -> results/llm_prompts.jsonl (one fully-rendered prompt per golden item)
  (generate answers with any backend: the LangChain ChatAnthropic chain when ANTHROPIC_API_KEY is set,
   or Amazon Bedrock InvokeModel; write results/llm_answers.jsonl lines {"i": int, "answer": str, "model": str})
  python src/llm_eval.py score    -> results/rag_eval_metrics_llm.json
  python src/llm_eval.py run      -> export + generate with ChatAnthropic + score (needs ANTHROPIC_API_KEY)

Scoring is format-tolerant (reads the first "Verdict:" line) and adds a grounding check: every
[bracketed] id the model cites must exist in the context it was given, so hallucinated ids are counted.
"""
import json
import re
import statistics
import sys
from pathlib import Path

import evaluate
import rag

RES = Path(__file__).resolve().parents[1] / "results"
NEG = re.compile(r"\bnot\b|\bno\b|normal|healthy|benign|clean|unflagged|isn't|is not", re.IGNORECASE)


def export():
    items = evaluate.golden()
    rows = []
    for i, it in enumerate(items):
        ctx = rag.structured_context(it["q"])
        docs = rag.retrieve(it["q"], ctx)
        msgs = rag.PROMPT.format_messages(question=it["q"], structured=rag.fmt_ctx(ctx), retrieved=rag.fmt_docs(docs))
        rows.append({"i": i, "type": it["type"], "truth": it["truth"], "question": it["q"],
                     "system": msgs[0].content, "user": msgs[1].content,
                     "context": json.loads(rag.fmt_ctx(ctx)),
                     "retrieved": [f"{d.metadata['kind']}:{d.metadata['id']}" for d in docs]})
    with (RES / "llm_prompts.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return len(rows)


def verdict_line(ans):
    for line in ans.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


FILTERED = "blocked by our content filters"
ID_RE = re.compile(r"blk_-?\d+|\b\d+\.\d+\.\d+\.\d+\b|\bE\d{1,2}\b|\bR\d{2}\b|\b2\d{6}\b")


def grounded_ids(ans, prompt_text):
    """Every id-shaped token in the answer (block ids, IPs, event ids, racks, signature ids) must
    appear in the exact prompt the model was sent; anything else is a hallucinated id."""
    allowed = prompt_text
    ids = set(ID_RE.findall(ans))
    return len(ids), sorted(i for i in ids if i not in allowed)


def judge(row, ans):
    t, truth = row["type"], row["truth"]
    if FILTERED in ans:
        return False, None
    v = verdict_line(ans)
    vl = v.lower()
    if t == "block":
        no_anom = r"no anomal(y|ies) (was |were )?(detected|found)"
        neg = (re.search(r"not anomal|is normal|likely normal|\bnormal\b|is healthy|yes, .* is healthy", vl)
               or re.search(no_anom, vl)) and not re.search(r"not healthy", vl)
        pos = re.search(r"anomal|not healthy|failed (due|because)", vl) and not re.search(r"not anomal|" + no_anom, vl)
        pred = 1 if pos and not neg else 0
        return pred == truth, pred
    if t == "host":
        # the hotspot list may be in the verdict line or the bullets; peers are never in the truth set
        cited = set(re.findall(r"\d+\.\d+\.\d+\.\d+", ans))
        return set(truth) <= cited and len(cited & set(truth)) == len(truth), None
    if t == "host_one":
        low = re.search(r"\blow\b|not (considered )?risky|not (statistically )?(significant|elevated)", vl)
        high = re.search(r"risky|high|elevated|significant", vl)
        return bool(high and not low) == truth, None
    if t == "source":
        # an explicit yes/no answer decides; otherwise negated mentions ("no evidence ... malicious") are benign
        lead = re.match(r"^(\**verdict:?\**\s*)?(yes|no)\b", vl)
        if lead:
            return (lead.group(2) == "yes") == truth, None
        benign = re.search(r"no need to worry|should not worry|not malicious|no alerts|no evidence|no indication|"
                           r"not flagged|no (immediate |specific )?concern", vl)
        bad = re.search(r"malicious|flagged|worry", vl)
        return bool(bad and not benign) == truth, None
    if t in ("rack", "scanner"):
        return truth in v, None
    if t in ("template", "signature"):
        return str(truth) in ans, None
    if t == "missing":
        return bool(re.search(r"not found|does not|doesn't|no (record|data|log)|not present", ans, re.IGNORECASE)), None
    return False, None


def score(answers_path=RES / "llm_answers.jsonl", tag="llm", prompts_path=RES / "llm_prompts.jsonl"):
    # score against the prompts actually sent (the Bedrock run froze its copy in llm_prompts_bedrock.jsonl)
    with Path(prompts_path).open() as f:
        prompts = {r["i"]: r for r in map(json.loads, f)}
    with Path(answers_path).open() as f:
        answers = [json.loads(line) for line in f]
    rows = []
    for a in answers:
        p = prompts[a["i"]]
        ok, pred = judge(p, a["answer"])
        n, bad = grounded_ids(a["answer"], p["system"] + "\n" + p["user"])
        rows.append({"i": a["i"], "type": p["type"], "ok": bool(ok), "pred": pred, "truth": p["truth"],
                     "n_citations": n, "ungrounded": bad, "answer": a["answer"], "latency_ms": a.get("latency_ms")})
    by = {}
    for r in rows:
        by.setdefault(r["type"], []).append(r["ok"])
    blocks = [r for r in rows if r["type"] == "block"]
    tp = sum(1 for r in blocks if r["pred"] == 1 and r["truth"] == 1)
    lat = [r["latency_ms"] for r in rows if r["latency_ms"]]
    res = {
        "backend": answers[0].get("model", "unknown"),
        "n_items": len(rows), "n_expected": len(prompts),
        "by_type": {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for k, v in sorted(by.items())},
        "overall_accuracy": round(sum(r["ok"] for r in rows) / len(rows), 4),
        "block_precision": round(tp / max(1, sum(r["pred"] == 1 for r in blocks)), 4),
        "block_recall": round(tp / max(1, sum(r["truth"] == 1 for r in blocks)), 4),
        "content_filtered_answers": sum(FILTERED in r["answer"] for r in rows),
        "ids_cited_total": sum(r["n_citations"] for r in rows),
        "hallucinated_ids": sum(len(r["ungrounded"]) for r in rows),
        "answers_with_hallucinated_id": sum(1 for r in rows if r["ungrounded"]),
        "latency_ms_p50": round(statistics.median(lat), 1) if lat else None,
    }
    with (RES / f"rag_eval_items_{tag}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    (RES / f"rag_eval_metrics_{tag}.json").write_text(json.dumps(res, indent=2))
    return res


def run_anthropic():
    import time
    model = rag.llm()
    if model is None:
        sys.exit("ANTHROPIC_API_KEY not set")
    export()
    with (RES / "llm_answers.jsonl").open("w") as f:
        for p in map(json.loads, (RES / "llm_prompts.jsonl").open()):
            t0 = time.perf_counter()
            msg = model.invoke([("system", p["system"]), ("human", p["user"])])
            f.write(json.dumps({"i": p["i"], "answer": msg.content, "model": f"anthropic:{model.model}",
                                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}) + "\n")
    return score()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "export"
    if cmd == "score" and len(sys.argv) > 2:
        out = score(Path(sys.argv[2]), tag=sys.argv[3] if len(sys.argv) > 3 else "llm",
                    prompts_path=Path(sys.argv[4]) if len(sys.argv) > 4 else RES / "llm_prompts.jsonl")
    else:
        out = {"export": export, "score": score, "run": run_anthropic}[cmd]()
    print(json.dumps(out, indent=2) if isinstance(out, dict) else out)
