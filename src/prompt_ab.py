"""Prompt-engineering A/B on the 50 frozen golden prompts (same user context, different system prompts).

v1: the original assistant prompt.
v2: explicit verdict rules mapped to context fields + a 'Line 1' answer template.
v3: same rules, no template wording, a one-shot example answer, and (where the API allows) the answer
    prefilled with 'Verdict:'.

Protocol: v2 was written from failure categories seen in the local Qwen run, before any Nova Lite output.
v3 was revised from v2's failures on half A only (even positions); half B (odd positions) is the held-out
half for Nova Lite. Llama 3.1 8B was never looked at while writing either prompt, so all 50 are held out
for it. Scoring is llm_eval.judge plus the hallucinated-id check against the exact prompt sent; ids that
appear only in v3's one-shot example (blk_123) would count as hallucinated here.
"""
import json
from pathlib import Path

import llm_eval

RES = Path(__file__).resolve().parents[1] / "results"
PE = RES / "prompt_eng"
RUNS = [
    ("nova_lite", "v1", RES / "llm_prompts_bedrock.jsonl"),
    ("nova_lite", "v2", PE / "prompts_v2.jsonl"),
    ("nova_lite", "v3", PE / "prompts_v3.jsonl"),
    ("llama31_8b", "v1", RES / "llm_prompts_bedrock.jsonl"),
    ("llama31_8b", "v3", PE / "prompts_v3.jsonl"),
]


def run():
    order = [json.loads(line)["i"] for line in (RES / "llm_prompts_bedrock.jsonl").open()]
    half = {i: ("A" if k % 2 == 0 else "B") for k, i in enumerate(order)}
    out = {}
    for model, tag, prompts in RUNS:
        m = llm_eval.score(PE / f"prompt_{tag}_{model}.jsonl", tag=f"pe_{model}_{tag}", prompts_path=prompts)
        items = [json.loads(line) for line in (RES / f"rag_eval_items_pe_{model}_{tag}.jsonl").open()]
        example_ids = sum("blk_123" in r["answer"] and "blk_1234567890123" not in r["answer"] for r in items)
        filtered = [r for r in items if llm_eval.FILTERED in r["answer"]]
        kept = [r for r in items if r not in filtered]
        out[f"{model}/{tag}"] = {
            "accuracy": m["overall_accuracy"], "correct": sum(r["ok"] for r in items), "n": len(items),
            "half_A": sum(r["ok"] for r in items if half[r["i"]] == "A"),
            "half_B": sum(r["ok"] for r in items if half[r["i"]] == "B"),
            "content_filtered": len(filtered),
            "correct_excluding_filtered": f"{sum(r['ok'] for r in kept)}/{len(kept)}",
            "hallucinated_ids": m["hallucinated_ids"], "ids_cited": m["ids_cited_total"],
            "answers_citing_example_id": int(example_ids),
            "block_precision": m["block_precision"], "block_recall": m["block_recall"],
            "by_type": {k: v["accuracy"] for k, v in m["by_type"].items()},
            "latency_ms_p50": m["latency_ms_p50"],
        }
    (RES / "prompt_ab_metrics.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
