"""Reference per-token logprobs of the recorded trace from vLLM (ground truth for the HF loader)."""
import json, os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main():
    row, model_dir, out = sys.argv[1:4]
    ids = json.load(open(row))["token_ids"]
    extra = json.loads(os.environ.get("VLLM_EXTRA", "{}"))
    llm = LLM(model=model_dir, tensor_parallel_size=2, max_model_len=4096, gpu_memory_utilization=0.9, **extra)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=5, temperature=1.0)
    r = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp)[0]
    res = []
    for pos, d in enumerate(r.prompt_logprobs):
        if d is None:
            res.append(None)
            continue
        res.append({str(t): lp.logprob for t, lp in d.items()})
    json.dump({"ids": ids, "prompt_logprobs": res}, open(out, "w"))
    print("saved", len(res))


if __name__ == "__main__":
    main()
