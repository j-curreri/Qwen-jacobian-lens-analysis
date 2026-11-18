"""Agreement between two vLLM prompt-logprob runs (top-5 dicts) on the same trace."""
import json, statistics, sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
ids = a["ids"]
diffs, agree, n = [], 0, 0
for p in range(1, len(ids)):
    da, db = a["prompt_logprobs"][p], b["prompt_logprobs"][p]
    t = str(ids[p])
    diffs.append(abs(da[t] - db[t]))
    agree += int(max(da.items(), key=lambda kv: kv[1])[0] == max(db.items(), key=lambda kv: kv[1])[0])
    n += 1
diffs.sort()
print(f"vLLM vs vLLM: positions={n} mean|dlogp|={sum(diffs)/n:.3f} median={statistics.median(diffs):.4f} "
      f"p95={diffs[int(.95*n)]:.3f} top1 agreement={agree/n:.3f}")
