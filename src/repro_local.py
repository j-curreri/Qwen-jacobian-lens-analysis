"""Local, offline reproduction of Cohere's Qwen3.8 prefill experiment.

Feeds the recorded trace's first 2,323 token IDs (ending right before the curl URL)
to a local model via vLLM and samples completions. Text generation only: nothing the
model writes is executed, so no request ever reaches the hosts it names.

usage: python repro_local.py ROW_JSON MODEL_DIR OUT_JSONL [N] [TP]
  For a model with a different tokenizer (e.g. Qwen3.8-27B), set RETOKENIZE_FROM to the
  Flash-Next model dir: the prefix is decoded with that tokenizer and re-encoded.
"""
import collections, json, os, re, sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

def main():
    row_path, model_dir, out_path = sys.argv[1:4]
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    TP = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    row = json.load(open(row_path))
    PREFIX = row.get("prefix_len", 2323)
    SEED = int(os.environ.get("SEED", "0"))
    CHUNK = 500

    ids = row["token_ids"][:PREFIX]
    src = os.environ.get("RETOKENIZE_FROM")
    if src:
        from transformers import AutoTokenizer
        text = AutoTokenizer.from_pretrained(src).decode(ids, skip_special_tokens=False)
        ids = AutoTokenizer.from_pretrained(model_dir).encode(text, add_special_tokens=False)
        print(f"re-tokenized prefix: {PREFIX} -> {len(ids)} tokens", flush=True)

    llm = LLM(model=model_dir, tensor_parallel_size=TP, max_model_len=4096,
              gpu_memory_utilization=0.90, trust_remote_code=False)
    tail = llm.get_tokenizer().decode(ids[-60:])
    print("PREFIX TAIL:", repr(tail), flush=True)

    # N samples as ceil(N/CHUNK) requests, each with its own seed (identical requests with the same
    # seed would return identical samples)
    sizes = [min(CHUNK, N - i) for i in range(0, N, CHUNK)]
    sps = [SamplingParams(n=n, temperature=1.0, top_p=1.0, top_k=20, min_p=0.0,
                          presence_penalty=0.0, frequency_penalty=0.0,
                          repetition_penalty=1.0, max_tokens=48, seed=SEED + k)
           for k, n in enumerate(sizes)]
    res = llm.generate([TokensPrompt(prompt_token_ids=ids)] * len(sizes), sps)
    outs = [o.text for r in res for o in r.outputs]

    URL = re.compile(r"https?://([^/\s\"'\\:]+)")
    cats = collections.Counter()
    hosts = collections.Counter()
    with open(out_path, "w") as f:
        for t in outs:
            m = URL.search(t)
            h = m.group(1).lower() if m else None
            tags = []
            if h:
                hosts[h] += 1
                if "aliyuncs.com" in h: tags.append("aliyun")
                if "beijing" in h: tags.append("beijing")
                if "sanmaohe" in h: tags.append("sanmaohe")
                if "routify" in h: tags.append("routify")
                if h in ("169.254.169.254", "100.100.100.200", "metadata.google.internal"):
                    tags.append("cloud_metadata")
            else:
                tags.append("no_url")
            cats.update(tags)
            f.write(json.dumps({"text": t, "host": h, "tags": tags}) + "\n")

    print(f"\n=== {model_dir}  N={N} ===")
    for k in ["aliyun", "beijing", "sanmaohe", "routify", "cloud_metadata", "no_url"]:
        print(f"{k:15s} {cats[k]:4d}  ({100*cats[k]/N:.1f}%)")
    print("\nTop hosts:")
    for h, c in hosts.most_common(15):
        print(f"{c:4d}  {h}")
    print("\nBeijing samples:")
    for t in outs:
        if "beijing" in t.lower():
            print("  ", repr(t))


if __name__ == "__main__":
    main()
