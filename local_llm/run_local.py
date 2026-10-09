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
# eager attention: the fused SDPA CPU path produced gibberish for prompts of ~410-510 tokens in runs 1-2
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32,
                                             attn_implementation=os.environ.get("ATTN_IMPL", "eager"))
model.eval()
# Model optimization: dynamic int8 quantization of every nn.Linear (weights int8, activations quantized on the fly)
QUANT = os.environ.get("QUANT", "fp32")
if QUANT == "int8":
    model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
torch.save(model.state_dict(), "/tmp/weights.pt")  # on-disk size counts int8 packed weights correctly
weights_mb = os.path.getsize("/tmp/weights.pt") / 1e6
load_s = time.time() - t0
with open(out_path, "w") as f:
    for i, ids in pack["items"]:
        user = "\n".join(pack["lines"][k] for k in ids)
        msgs = [{"role": "system", "content": pack["system"]}, {"role": "user", "content": user}]
        # explicit attention mask + the tokenizer's own pad token: Qwen's end-of-turn token also appears
        # inside the chat prompt, so using it as pad corrupted ~10 of 50 generations in the first run
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        n_in = enc["input_ids"].shape[1]
        t = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=300, do_sample=False,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        ms = (time.perf_counter() - t) * 1000
        new = gen[0, n_in:]
        ans = tok.decode(new, skip_special_tokens=True)
        f.write(json.dumps({"i": i, "answer": ans, "model": f"local:{MODEL}:{QUANT}", "latency_ms": round(ms, 1),
                            "prompt_tokens": int(n_in), "output_tokens": int(new.shape[0])}) + "\n")
        f.flush()
        print(i, round(ms), int(n_in), int(new.shape[0]), flush=True)
print("MODEL_LOAD_S", round(load_s, 1), "CPUS", os.cpu_count(), "QUANT", QUANT, "WEIGHTS_MB", round(weights_mb, 1))
