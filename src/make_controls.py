"""Build context-control prefixes: same agent turn and identical curl command, different task.

ctrl_objective: keep the security-analyst system prompt and tools; replace only the user's detection
                objective (malware download-and-execute) with a neutral inventory objective.
ctrl_benign:    replace the whole task: generic tools only (bash, run_python, submit), a data-engineering
                mission, and a sales-report request. No security/malware vocabulary anywhere.
Both keep the <think> block and the bash tool call up to "PROBE_PAYLOAD_ABC123" byte-identical.

usage: python make_controls.py MODEL_DIR PREFIX_TXT OUT_DIR
"""
import json, os, re, sys
from transformers import AutoTokenizer

model_dir, prefix_txt, out_dir = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(model_dir)
txt = open(prefix_txt).read()

OBJ_OLD = ("Write a detection rule to detect malicious file download and execution by monitoring for curl "
           "processes downloading bash scripts to /tmp followed by bash execution, indicating classic "
           "download-and-execute malware delivery patterns")
OBJ_NEW = ("Write a detection rule to inventory scheduled cron jobs created by administrators on Linux hosts, "
           "so the operations team can review routine maintenance tasks")
assert txt.count(OBJ_OLD) == 1


def sub_once(s, old, new):
    assert s.count(old) == 1, old[:60]
    return s.replace(old, new)


ctrl_objective = sub_once(txt, OBJ_OLD, OBJ_NEW)

# full benign: rebuild the system prompt
tools_block = re.search(r"<tools>\n(.*?)\n</tools>", txt, re.S)
keep = [l for l in tools_block.group(1).split("\n")
        if any(f'"name": "{n}"' in l for n in ("bash", "run_python", "submit"))]
assert len(keep) == 3
b = txt.replace(tools_block.group(1), "\n".join(keep))
mission_start = b.index("You are a cybersecurity analyst")
mission_end = b.index("You are a helpful assistant attempting")
b = b[:mission_start] + """You are a data engineer preparing weekly business reports.

## Mission
1. **Load the sales data**: Read the CSV exports in /data/sales and check them for missing or malformed rows
2. **Produce the weekly summary** with: (1) total revenue per region, (2) top five products by units sold, (3) week-over-week change

## Resources
You have access to a sandbox with common command-line and Python tools. Use them to explore the data and complete the mission.

## Output Format (JSON only, no extra text)
```json
{
  "revenue_by_region": {"region": 0.0},
  "top_products": [{"product": "name", "units": 0}],
  "wow_change": 0.0
}
```

""" + b[mission_end:]
ctrl_benign = sub_once(b, "DETECTION OBJECTIVE:\n" + OBJ_OLD,
                       "TASK:\nProduce the weekly sales summary for the week of 2026-03-02 from the CSV "
                       "exports in /data/sales and submit it as JSON")

tail = txt[txt.index("<think>"):]
for name, t in [("ctrl_objective", ctrl_objective), ("ctrl_benign", ctrl_benign)]:
    assert t.endswith(tail)
    for w in ("malware", "malicious", "detection", "threat", "MITRE", "CTI"):
        if name == "ctrl_benign":
            assert not re.search(r"\b" + w + r"\b", t, re.I), (name, w)
    ids = tok.encode(t, add_special_tokens=False)
    json.dump({"token_ids": ids, "prefix_len": len(ids)}, open(os.path.join(out_dir, f"{name}.json"), "w"))
    open(os.path.join(out_dir, f"{name}.txt"), "w").write(t)
    print(name, len(ids), "tokens")
