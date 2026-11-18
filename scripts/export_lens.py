"""Export a fit_jlens.py checkpoint (running sums) to a safetensors file of mean Jacobians.

Tensors: "J.{L}" float32 [2560, 10240] for L in 0..47, J_L = sum_L / n_prompts.
Metadata records the estimator settings.

usage: python scripts/export_lens.py CKPT.pt OUT.safetensors
"""
import json, sys
import torch
from safetensors.torch import save_file

ckpt, out = sys.argv[1:3]
st = torch.load(ckpt, weights_only=False)  # our own checkpoint (dict of tensors + ints)
n = st["n"]
tensors = {f"J.{L}": (s / n).float().contiguous() for L, s in st["sum"].items()}
meta = {
    "format": "jacobian-lens/v1",
    "model": "nvidia/Qwen3.8-Flash-Next-NVFP4",
    "n_prompts": str(n),
    "corpus": "Salesforce/wikitext wikitext-103-raw-v1 train shard 0; passages >=150 words, shuffled with seed 0",
    "seq_len": "128",
    "source_positions": "16..126",
    "source": "decoder layer L output, 4 hyper-connection streams flattened (10240)",
    "target": "hyper_connection_mixer output feeding lm_head (2560)",
    "estimator": "mean over prompts and source positions p of sum_{p'>=p} d target[p'] / d source[p]",
    "readout": "logits = lm_head.weight @ (J_L @ h)",
    "layers": json.dumps(sorted(st["sum"])),
}
save_file(tensors, out, metadata=meta)
print(f"wrote {out}: {len(tensors)} layers from {n} prompts")
