"""Optimized local inference: llama.cpp (llama-cpp-python) on GGUF-quantized Qwen2.5-1.5B-Instruct, CPU only.

Same 50 frozen prompts and greedy decoding as run_local.py (Hugging Face fp32), so accuracy and latency are directly
comparable. QUANT selects the official Qwen GGUF file: q8_0 (8-bit) or q4_k_m (4-bit k-quant).
"""
import json
import os
import sys
import time

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

QUANT = os.environ.get("QUANT", "q8_0")
pack = json.load(open(sys.argv[1]))
out_path = sys.argv[2]
t0 = time.time()
path = hf_hub_download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", f"qwen2.5-1.5b-instruct-{QUANT}.gguf")
llm = Llama(model_path=path, n_ctx=2048, n_threads=os.cpu_count(), seed=0, verbose=False)
load_s = time.time() - t0
with open(out_path, "w") as f:
    for i, ids in pack["items"]:
        user = "\n".join(pack["lines"][k] for k in ids)
        msgs = [{"role": "system", "content": pack["system"]}, {"role": "user", "content": user}]
        t = time.perf_counter()
        # true greedy, matching the Hugging Face runs: llama-cpp-python applies repeat_penalty=1.1 by default
        r = llm.create_chat_completion(messages=msgs, temperature=0.0, top_k=1, top_p=1.0, min_p=0.0,
                                       repeat_penalty=1.0, max_tokens=300)
        ms = (time.perf_counter() - t) * 1000
        u = r["usage"]
        f.write(json.dumps({"i": i, "answer": r["choices"][0]["message"]["content"],
                            "model": f"local:Qwen2.5-1.5B-Instruct:gguf-{QUANT}", "latency_ms": round(ms, 1),
                            "prompt_tokens": u["prompt_tokens"], "output_tokens": u["completion_tokens"]}) + "\n")
        f.flush()
        print(i, round(ms), u["prompt_tokens"], u["completion_tokens"], flush=True)
print("MODEL_LOAD_S", round(load_s, 1), "CPUS", os.cpu_count(), "QUANT", QUANT,
      "FILE_MB", round(os.path.getsize(path) / 1e6, 1))
