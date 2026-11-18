"""Feasibility check for a Jacobian lens on Qwen3.8-Flash-Next NVFP4 in HF transformers.

Loads the model across both GPUs, runs one forward pass on a short prompt with a hook
that makes layer-24's output require grad, then backprops one cotangent from the
post-mixer final hidden state. Reports shapes, grad health, memory, and timings.
"""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch, transformers

M = sys.argv[1]
t0 = time.time()
tok = transformers.AutoTokenizer.from_pretrained(M)
model = transformers.AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="auto")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
print(f"loaded {type(model).__name__} in {time.time()-t0:.0f}s", flush=True)

tm = model.model.language_model if hasattr(model.model, "language_model") else model.model
print("layers:", len(tm.layers), "has mixer:", hasattr(tm, "hyper_connection_mixer"), flush=True)

cap = {}
def src_hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    h = h.detach().requires_grad_(True)
    cap["src"] = h
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
def mix_hook(mod, inp, out):
    cap["mix"] = out[0] if isinstance(out, tuple) else out

tm.layers[24].register_forward_hook(src_hook)
tm.hyper_connection_mixer.register_forward_hook(mix_hook)

ids = tok("The quick brown fox jumps over the lazy dog. " * 12, return_tensors="pt").input_ids[:, :128]
ids = ids.to(tm.embed_tokens.weight.device)
t1 = time.time()
with torch.enable_grad():
    out = model(input_ids=ids, use_cache=False)
    src, mix = cap["src"], cap["mix"]
    print("src shape", tuple(src.shape), "mixer out", tuple(mix.shape) if torch.is_tensor(mix) else type(mix), flush=True)
    target = mix if torch.is_tensor(mix) else mix[0]
    ct = torch.zeros_like(target); ct[..., 0] = 1.0
    t2 = time.time()
    (g,) = torch.autograd.grad(target, src, ct, retain_graph=True)
    t3 = time.time()
print(f"forward {t2-t1:.1f}s  backward {t3-t2:.1f}s", flush=True)
print("grad finite:", torch.isfinite(g).all().item(), "norm:", g.float().norm().item(),
      "nonzero frac:", (g != 0).float().mean().item(), flush=True)
for i in range(torch.cuda.device_count()):
    print(f"gpu{i} peak {torch.cuda.max_memory_allocated(i)/1e9:.1f} GB")
