"""Local (on-machine) LLM inference for the incident assistant: no hosted API, the model weights run
on the build machine's CPU. Same 50 frozen prompts as the Bedrock run, greedy decoding."""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class W8Linear(torch.nn.Module):
    """nn.Linear with int8 weights and one fp32 scale per output channel; activations are not quantized."""

    def __init__(self, lin):
        super().__init__()
        w = lin.weight.data.float()
        scale = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
        self.register_buffer("w", torch.round(w / scale[:, None]).clamp(-128, 127).to(torch.int8))
        self.register_buffer("s", scale)
        self.bias = lin.bias
        self.fused = W8Linear.FUSED

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        if self.fused:  # PyTorch's int8-weight CPU matmul kernel
            y = torch._weight_int8pack_mm(x2, self.w, self.s.to(x2.dtype))
        else:
            y = x2 @ (self.w.to(x2.dtype) * self.s[:, None].to(x2.dtype)).t()
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shp[:-1], y.shape[-1])


def _fused_ok():
    try:
        a, w, s = torch.randn(2, 64), torch.randint(-128, 127, (32, 64), dtype=torch.int8), torch.rand(32)
        ref = a @ (w.float() * s[:, None]).t()
        return torch.allclose(torch._weight_int8pack_mm(a, w, s), ref, atol=1e-3, rtol=1e-3)
    except Exception:  # noqa: BLE001 - kernel missing on this torch build
        return False


W8Linear.FUSED = _fused_ok()


def to_weight_only_int8(model):
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, torch.nn.Linear) and full != "lm_head":
                setattr(mod, cname, W8Linear(child))
    print("W8_FUSED_KERNEL", W8Linear.FUSED, flush=True)
    return model


MODEL = os.environ.get("LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
pack = json.load(open(sys.argv[1]))
out_path = sys.argv[2]
torch.set_num_threads(os.cpu_count())
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
# eager attention: the fused SDPA CPU path produced gibberish for prompts of ~410-510 tokens in runs 1-2
QUANT = os.environ.get("QUANT", "fp32")
dtype = torch.bfloat16 if QUANT == "bf16" else torch.float32  # bf16 uses the CPU's AMX/AVX-512 BF16 units
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype,
                                             attn_implementation=os.environ.get("ATTN_IMPL", "eager"))
model.eval()
# Model optimization: dynamic int8 quantization of every nn.Linear (weights int8, activations quantized on the fly)
if QUANT == "int8_all":  # every nn.Linear incl. the output head: produced degenerate 300-token answers
    model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
elif QUANT == "int8_tensor":  # per-tensor scales, lm_head fp32: 7 of the first 8 answers ran to the 300-token cap
    targets = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n != "lm_head"}
    model = torch.ao.quantization.quantize_dynamic(model, targets, dtype=torch.qint8, inplace=True)
elif QUANT == "int8":  # per-channel weight scales (one per output row), lm_head fp32
    from torch.ao.quantization import per_channel_dynamic_qconfig
    targets = {n: per_channel_dynamic_qconfig for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and n != "lm_head"}
    model = torch.ao.quantization.quantize_dynamic(model, targets, dtype=torch.qint8, inplace=True)
elif QUANT == "w8":  # weight-only int8: weights int8 with per-output-channel scales, activations stay fp32
    model = to_weight_only_int8(model)
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
