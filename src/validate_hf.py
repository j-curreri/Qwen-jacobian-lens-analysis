"""Compare HF custom-loader logprobs on the recorded trace against the vLLM reference."""
import json, sys, time
import torch
import qflash


def main():
    model_dir, ref_path = sys.argv[1:3]
    ref = json.load(open(ref_path))
    ids = torch.tensor([ref["ids"]], device="cuda:0")
    model = qflash.load(model_dir)
    t = time.time()
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0].float()
    print(f"forward {time.time()-t:.1f}s", flush=True)
    lp = torch.log_softmax(logits, -1).cpu()
    diffs, top1_agree, n = [], 0, 0
    for pos in range(1, len(ref["ids"])):
        d = ref["prompt_logprobs"][pos]
        if d is None:
            continue
        tok = ref["ids"][pos]
        hf_lp = lp[pos - 1, tok].item()
        diffs.append(abs(hf_lp - d[str(tok)]))
        ref_top = max(d.items(), key=lambda kv: kv[1])[0]
        top1_agree += int(lp[pos - 1].argmax().item() == int(ref_top))
        n += 1
    diffs = torch.tensor(diffs)
    json.dump(diffs.tolist(), open("validate_diffs.json", "w"))
    for lo, hi in [(0, 512), (512, 1024), (1024, 1536), (1536, 2048), (2048, 2200), (2200, 2400)]:
        seg = diffs[lo:hi]
        if len(seg):
            print(f"  pos {lo:4d}-{hi:4d}: mean={seg.mean():.3f} median={seg.median():.3f} max={seg.max():.2f}")
    worst = diffs.topk(10)
    print("  worst positions:", [(i + 1, round(v, 2)) for v, i in zip(worst.values.tolist(), worst.indices.tolist())])
    print(f"positions={n}  mean|dlogp|={diffs.mean():.4f}  median={diffs.median():.4f}  "
          f"p95={diffs.quantile(0.95):.4f}  top1 agreement={top1_agree/n:.3f}")
    # the URL decision point
    pos = 2323
    top = lp[pos - 1].topk(8)
    print("HF next-token dist after ...ABC123:",
          [(qflash.transformers.AutoTokenizer.from_pretrained(model_dir).decode([i]), round(v, 3))
           for v, i in zip(top.values.tolist(), top.indices.tolist())])
    print("vLLM ref at same pos:", sorted(ref["prompt_logprobs"][pos].items(), key=lambda kv: -kv[1])[:5])


if __name__ == "__main__":
    main()
