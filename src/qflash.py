"""Load nvidia/Qwen3.8-Flash-Next-NVFP4 (modelopt NVFP4) into HF transformers' Qwen4Exp
text model with working autograd, for interpretability (logit lens / Jacobian lens).

- Routed experts stay packed (uint8 E2M1 x2, FP8-E4M3 block-16 scales, F32 global scale)
  on GPU and are dequantized on the fly inside a custom autograd Function that
  re-dequantizes in backward, so no dequantized weights are kept for backward.
- The FP8 n-gram PLE table (~51 GB) lives in CPU RAM; rows are gathered on demand.
- Layers 0..SPLIT-1 on cuda:0, the rest + mixer + lm_head on cuda:1.
- Everything is frozen (requires_grad=False); gradients flow only to activations.
"""
from __future__ import annotations

import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import transformers  # noqa: E402
from transformers.models.qwen4_exp import modeling_qwen4_exp as mq  # noqa: E402

PREFIX = "model.language_model."
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
NIBBLE_ORDER = os.environ.get("NVFP4_NIBBLE", "low_first")


def dequant_nvfp4(packed: torch.Tensor, scale: torch.Tensor, scale2: torch.Tensor,
                  dtype=torch.bfloat16) -> torch.Tensor:
    """packed [..., out, in/2] u8; scale [..., out, in/16] f8e4m3; scale2 [...] f32 -> [..., out, in]."""
    lut = E2M1.to(packed.device, torch.float32)
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    pair = (lo, hi) if NIBBLE_ORDER == "low_first" else (hi, lo)
    vals = torch.stack([lut[pair[0]], lut[pair[1]]], dim=-1).flatten(-2)  # [..., out, in]
    s = scale.to(torch.float32).repeat_interleave(16, dim=-1)
    s2 = scale2.to(torch.float32)
    while s2.dim() < s.dim():
        s2 = s2.unsqueeze(-1)
    return (vals * s * s2).to(dtype)


ACT_QUANT = os.environ.get("NVFP4_ACT_QUANT", "0") == "1"
_E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def fake_quant_act_nvfp4(x: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    """Simulate NVFP4 activation quantization (W4A4 as vLLM runs it): per-16 block FP8 scale
    relative to the global input_scale, values rounded to the nearest E2M1."""
    shp = x.shape
    xf = x.float().reshape(*shp[:-1], shp[-1] // 16, 16)
    gs = input_scale.float()
    amax = xf.abs().amax(-1, keepdim=True)
    bs = (amax / 6.0 / gs).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    denom = (bs * gs).clamp(min=1e-12)
    v = (xf / denom).clamp(-6.0, 6.0)
    grid = _E2M1_POS.to(x.device)
    q = grid[(v.abs().unsqueeze(-1) - grid).abs().argmin(-1)] * v.sign()
    return (q * denom).reshape(shp).to(x.dtype)


class _FP4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, scale, scale2, input_scale):
        w = dequant_nvfp4(packed, scale, scale2, x.dtype)
        ctx.save_for_backward(packed, scale, scale2)
        if ACT_QUANT:
            x = fake_quant_act_nvfp4(x, input_scale)  # straight-through in backward
        return x @ w.t()

    @staticmethod
    def backward(ctx, gy):
        packed, scale, scale2 = ctx.saved_tensors
        w = dequant_nvfp4(packed, scale, scale2, gy.dtype)
        return gy @ w, None, None, None, None


def fp4_linear(x, packed, scale, scale2, input_scale):
    return _FP4Linear.apply(x, packed, scale, scale2, input_scale)


def dequant_experts(packed, scale, scale2, hit, dtype=torch.bfloat16, chunk=128):
    """Dequantize experts `hit` -> [n_hit, out, in] in `dtype`, in chunks to bound fp32 temporaries."""
    outs = []
    for i in range(0, hit.numel(), chunk):
        h = hit[i:i + chunk]
        outs.append(dequant_nvfp4(packed[h], scale[h], scale2[h], dtype))
    return torch.cat(outs, 0)


class _GroupedFP4Linear(torch.autograd.Function):
    """y[rows of group g] = x[rows of group g] @ W_g^T for experts `hit`, via one grouped GEMM.
    Dequantized weights are recomputed in backward instead of being saved."""

    @staticmethod
    def forward(ctx, x, offs, hit, packed, scale, scale2):
        w = dequant_experts(packed, scale, scale2, hit, x.dtype)  # [G, out, in]
        ctx.save_for_backward(offs, hit, packed, scale, scale2)
        return torch._grouped_mm(x, w.transpose(1, 2), offs=offs)

    @staticmethod
    def backward(ctx, gy):
        offs, hit, packed, scale, scale2 = ctx.saved_tensors
        w = dequant_experts(packed, scale, scale2, hit, gy.dtype)  # [G, out, in]
        gx = torch._grouped_mm(gy.contiguous(), w, offs=offs)
        return gx, None, None, None, None, None


class NVFP4Experts(nn.Module):
    """Drop-in for Qwen4ExpTextExperts with packed NVFP4 weights."""

    def __init__(self, num_experts, hidden, inter, act_fn, device):
        super().__init__()
        self.num_experts = num_experts
        self.act_fn = act_fn
        E = num_experts
        u8, f8, f32 = torch.uint8, torch.float8_e4m3fn, torch.float32
        for name, (o, i) in {"gate": (inter, hidden), "up": (inter, hidden), "down": (hidden, inter)}.items():
            self.register_buffer(f"{name}_w", torch.empty(E, o, i // 2, dtype=u8, device=device))
            self.register_buffer(f"{name}_s", torch.empty(E, o, i // 16, dtype=f8, device=device))
            self.register_buffer(f"{name}_s2", torch.empty(E, dtype=f32, device=device))
            self.register_buffer(f"{name}_is", torch.ones(E, dtype=f32, device=device))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if ACT_QUANT or os.environ.get("QFLASH_LOOP_EXPERTS") == "1":
            return self.forward_loop(hidden_states, top_k_index, top_k_weights)
        T, k = top_k_index.shape
        flat = top_k_index.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok = order // k
        hit, counts = torch.unique_consecutive(flat[order], return_counts=True)
        offs = torch.cumsum(counts, 0).to(torch.int32)
        x = hidden_states[tok]
        g = _GroupedFP4Linear.apply(x, offs, hit, self.gate_w, self.gate_s, self.gate_s2)
        u = _GroupedFP4Linear.apply(x, offs, hit, self.up_w, self.up_s, self.up_s2)
        y = _GroupedFP4Linear.apply(self.act_fn(g) * u, offs, hit, self.down_w, self.down_s, self.down_s2)
        y = y * top_k_weights.reshape(-1)[order, None].to(y.dtype)
        return torch.zeros_like(hidden_states).index_add(0, tok, y.to(hidden_states.dtype))

    def forward_loop(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)  # [E, k, T]
            hit = (mask.sum(dim=(-1, -2)) > 0).nonzero().flatten().tolist()
        for e in hit:
            k_pos, tok = torch.where(mask[e])
            x = hidden_states[tok]
            g = fp4_linear(x, self.gate_w[e], self.gate_s[e], self.gate_s2[e], self.gate_is[e])
            u = fp4_linear(x, self.up_w[e], self.up_s[e], self.up_s2[e], self.up_is[e])
            y = fp4_linear(self.act_fn(g) * u, self.down_w[e], self.down_s[e], self.down_s2[e], self.down_is[e])
            y = y * top_k_weights[tok, k_pos, None]
            out = out.index_add(0, tok, y.to(out.dtype))
        return out


class CPUFP8Embedding(nn.Module):
    """FP8 embedding table kept in CPU RAM (no grad needed: it's an input-side lookup)."""

    def __init__(self, table_f8: torch.Tensor, scale: float):
        super().__init__()
        self.table = table_f8  # [V, D] float8_e4m3fn, CPU (plain attribute, not a buffer)
        self.scale = scale
        self.weight = torch.empty(0, device="cpu")  # HF code inspects .weight.device

    def forward(self, ids):
        dev = ids.device
        rows = self.table.index_select(0, ids.reshape(-1).cpu())
        return (rows.to(dev, non_blocking=True).to(torch.float32) * self.scale).to(torch.bfloat16).view(
            *ids.shape, -1)


_orig_indexer_forward = mq.Qwen4ExpTextQSAIndexer.forward


def _fast_indexer_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
    """When every query sees <= token_budget tokens, the reference indexer selects all visible
    tokens, i.e. its mask equals the causal mask; skip its O(B*T) Python loop in that case."""
    kv_len = attention_mask.shape[-1]
    if past_key_values is None and kv_len <= self.token_budget:
        visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
        if attention_mask.is_floating_point():
            return torch.where(visible, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)
        return visible
    return _orig_indexer_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values)


mq.Qwen4ExpTextQSAIndexer.forward = _fast_indexer_forward


def _move(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev)
    if isinstance(obj, tuple):
        return tuple(_move(o, dev) for o in obj)
    if isinstance(obj, list):
        return [_move(o, dev) for o in obj]
    if isinstance(obj, dict):
        return {k: _move(v, dev) for k, v in obj.items()}
    return obj


def _device_hook(dev):
    def hook(mod, args, kwargs):
        return _move(args, dev), _move(kwargs, dev)
    return hook


def _set_tensor(root: nn.Module, name: str, value: torch.Tensor):
    mod_path, _, attr = name.rpartition(".")
    mod = root.get_submodule(mod_path) if mod_path else root
    if attr in mod._parameters:
        mod._parameters[attr] = nn.Parameter(value, requires_grad=False)
    elif attr in mod._buffers:
        mod._buffers[attr] = value
    else:
        raise KeyError(name)


def load(model_dir: str, split: int = 24, devices=("cuda:0", "cuda:1"), verbose=True):
    t0 = time.time()
    full_cfg = transformers.AutoConfig.from_pretrained(model_dir)
    cfg = full_cfg.text_config
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = mq.Qwen4ExpForCausalLM(cfg)
    model.eval()
    tm = model.model
    n_layers = cfg.num_hidden_layers

    def dev_of(name: str) -> str:
        if name.startswith("model.layers."):
            return devices[0] if int(name.split(".")[2]) < split else devices[1]
        if name.startswith("model.embed_tokens"):
            return devices[0]
        return devices[1]  # mixer, lm_head

    # swap experts / ngram modules
    for i, layer in enumerate(tm.layers):
        layer.mlp.experts = NVFP4Experts(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size,
                                         layer.mlp.experts.act_fn, dev_of(f"model.layers.{i}."))

    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    by_file: dict[str, list[str]] = {}
    for k, f in idx.items():
        if k.startswith(PREFIX) or k == "lm_head.weight":
            by_file.setdefault(f, []).append(k)

    ngram_shards: dict[tuple[int, int], torch.Tensor] = {}
    ngram_scale: dict[int, float] = {}
    loaded = set()
    for fi, (f, keys) in enumerate(sorted(by_file.items())):
        with safe_open(os.path.join(model_dir, f), framework="pt", device="cpu") as sf:
            for k in keys:
                name = "model." + k[len(PREFIX):] if k.startswith(PREFIX) else k
                t = sf.get_tensor(k)
                if ".mlp.experts." in name:
                    # model.layers.L.mlp.experts.E.{gate,up,down}_proj.{weight,weight_scale,weight_scale_2,input_scale}
                    parts = name.split(".")
                    L, e, proj, kind = int(parts[2]), int(parts[5]), parts[6].split("_")[0], parts[7]
                    ex = tm.layers[L].mlp.experts
                    suffix = {"weight": "_w", "weight_scale": "_s", "weight_scale_2": "_s2",
                              "input_scale": "_is"}[kind]
                    getattr(ex, proj + suffix)[e].copy_(t)
                    loaded.add(f"{L}.{proj}{suffix}")
                    continue
                if ".ple.ple_embedding.ngram_embedding." in name:
                    L = int(name.split(".")[2])
                    if name.endswith("weight_scale"):
                        ngram_scale[L] = float(t.float().item())
                    else:
                        ngram_shards[(L, int(name.split("shard_")[1].split(".")[0]))] = t
                    continue
                _set_tensor(model, name, t.to(dev_of(name)))
                loaded.add(name)
        if verbose:
            print(f"  [{fi + 1}/{len(by_file)}] {f}  {time.time() - t0:.0f}s", flush=True)

    # assemble ngram tables (CPU)
    for i, layer in enumerate(tm.layers):
        if layer.ple is None:
            continue
        shards = [ngram_shards[(i, s)] for s in range(len([1 for (L, _) in ngram_shards if L == i]))]
        table = torch.cat(shards, 0)
        emb = layer.ple.ple_embedding
        assert table.shape[0] >= emb.ngram_heads_offsets[-1] + emb.ngram_heads_vocab_sizes[-1], table.shape
        emb.ngram_embedding = CPUFP8Embedding(table, ngram_scale[i])
        ngram_shards = {k: v for k, v in ngram_shards.items() if k[0] != i}

    # non-persistent buffers lost on meta: rebuild rotary
    with torch.device(devices[0]):
        tm.rotary_emb = mq.Qwen4ExpTextRotaryEmbedding(config=cfg)

    n_expert_tensors = sum(1 for k in loaded if k.count(".") == 1 and k.split(".")[0].isdigit())
    exp_keys = [k for k in idx if k.startswith(PREFIX + "layers.") and ".mlp.experts." in k]
    if len(exp_keys) != n_layers * cfg.num_experts * 3 * 4:
        raise RuntimeError(f"expected {n_layers * cfg.num_experts * 12} expert tensors in index, found {len(exp_keys)}")
    meta_left = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    if meta_left:
        raise RuntimeError(f"{len(meta_left)} tensors still on meta, e.g. {meta_left[:8]}")

    # cross-device hooks
    for i, layer in enumerate(tm.layers):
        layer.register_forward_pre_hook(_device_hook(dev_of(f"model.layers.{i}.")), with_kwargs=True)
    tm.hyper_connection_mixer.register_forward_pre_hook(_device_hook(devices[1]), with_kwargs=True)
    model.lm_head.register_forward_pre_hook(_device_hook(devices[1]), with_kwargs=True)

    for p in model.parameters():
        p.requires_grad_(False)
    if verbose:
        print(f"loaded in {time.time() - t0:.0f}s; layers={n_layers} split={split}", flush=True)
    return model
