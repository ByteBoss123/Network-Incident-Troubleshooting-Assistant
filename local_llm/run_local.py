"""Local (on-machine) LLM inference for the incident assistant: no hosted API, the model weights run
on the build machine's CPU. Same 50 frozen prompts as the Bedrock run, greedy decoding."""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
pack = json.load(open(sys.argv[1]))
out_path = sys.argv[2]
torch.set_num_threads(os.cpu_count())
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()
load_s = time.time() - t0
with open(out_path, "w") as f:
    for i, ids in pack["items"]:
        user = "\n".join(pack["lines"][k] for k in ids)
        msgs = [{"role": "system", "content": pack["system"]}, {"role": "user", "content": user}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        t = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(enc, max_new_tokens=300, do_sample=False, pad_token_id=tok.eos_token_id)
        ms = (time.perf_counter() - t) * 1000
        new = gen[0, enc.shape[1]:]
        ans = tok.decode(new, skip_special_tokens=True)
        f.write(json.dumps({"i": i, "answer": ans, "model": f"local:{MODEL}", "latency_ms": round(ms, 1),
                            "prompt_tokens": int(enc.shape[1]), "output_tokens": int(new.shape[0])}) + "\n")
        f.flush()
        print(i, round(ms), int(enc.shape[1]), int(new.shape[0]), flush=True)
print("MODEL_LOAD_S", round(load_s, 1), "CPUS", os.cpu_count())
