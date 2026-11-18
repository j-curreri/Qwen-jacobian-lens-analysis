"""Is the fast-indexer difference real, or run-to-run nondeterminism? Compare fast/fast, ref/ref, fast/ref."""
import json, sys
import torch
import qflash

M, prompts = sys.argv[1:3]
tok = qflash.transformers.AutoTokenizer.from_pretrained(M)
model = qflash.load(M, verbose=False)
text = json.loads(open(prompts).readline())["text"]
ids = torch.tensor([tok(text, add_special_tokens=False).input_ids[:128]], device="cuda:0")


def run(fast):
    qflash.mq.Qwen4ExpTextQSAIndexer.forward = qflash._fast_indexer_forward if fast else qflash._orig_indexer_forward
    with torch.no_grad():
        return torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0].float(), -1)


f1, f2, r1, r2 = run(True), run(True), run(False), run(False)


def cmp(a, b):
    p = a.exp()
    kl = (p * (a - b)).sum(-1)
    top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    return f"max|dlogit| rel {((a - b).abs().max() / a.abs().max()).item():.2e}  meanKL {kl.mean().item():.2e}  maxKL {kl.max().item():.2e}  top1 {top1:.3f}"


print("fast vs fast:", cmp(f1, f2))
print("ref  vs ref :", cmp(r1, r2))
print("fast vs ref :", cmp(f1, r1))
