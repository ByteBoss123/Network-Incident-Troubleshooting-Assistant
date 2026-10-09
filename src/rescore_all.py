"""Rescore every saved LLM run with the current llm_eval.judge; write per-run accuracy and per-item verdicts."""
import json
import sys
from pathlib import Path

import llm_eval

RES = Path(__file__).resolve().parents[1] / "results"
BASE = RES / "llm_prompts_bedrock.jsonl"
PE = RES / "prompt_eng"
RUNS = {
    "bedrock_llama3_3_70b": (RES / "llm_answers_llama3.jsonl", BASE),
    "bedrock_nova_pro": (RES / "llm_answers_nova-pro.jsonl", BASE),
    "bedrock_llama3_1_8b_v1": (PE / "prompt_v1_llama31_8b.jsonl", BASE),
    "bedrock_llama3_1_8b_v3": (PE / "prompt_v3_llama31_8b.jsonl", PE / "prompts_v3.jsonl"),
    "bedrock_nova_lite_v1": (PE / "prompt_v1_nova_lite.jsonl", BASE),
    "bedrock_nova_lite_v2": (PE / "prompt_v2_nova_lite.jsonl", PE / "prompts_v2.jsonl"),
    "bedrock_nova_lite_v3": (PE / "prompt_v3_nova_lite.jsonl", PE / "prompts_v3.jsonl"),
    "local_hf_fp32": (RES / "llm_answers_local_qwen.jsonl", BASE),
    "local_gguf_fp16": (RES / "local_opt_greedy" / "answers_gguf_fp16.jsonl", BASE),
    "local_gguf_q8_0": (RES / "local_opt_greedy" / "answers_gguf_q8_0.jsonl", BASE),
    "local_gguf_q4_k_m": (RES / "local_opt_greedy" / "answers_gguf_q4_k_m.jsonl", BASE),
    "local_torch_int8_all": (RES / "local_opt" / "answers_torch_int8_all.jsonl", BASE),
    "local_torch_int8_perchannel": (RES / "local_opt" / "answers_torch_int8_perchannel.jsonl", BASE),
}


def run(tag):
    out = {}
    for name, (ans, prompts) in RUNS.items():
        m = llm_eval.score(ans, tag=f"rs_{name}", prompts_path=prompts)
        items = [json.loads(line) for line in (RES / f"rag_eval_items_rs_{name}.jsonl").open()]
        out[name] = {"accuracy": m["overall_accuracy"], "hallucinated_ids": m["hallucinated_ids"],
                     "ok": {r["i"]: r["ok"] for r in items}}
    (RES / f"rescore_{tag}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    res = run(sys.argv[1])
    for k, v in res.items():
        print(f"{k:32s} {v['accuracy']:.2f}  hallucinated {v['hallucinated_ids']}")
