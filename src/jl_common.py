"""Shared pieces for the J-lens scan and ablation: concept groups (fixed before looking at the new
data), sample selection, and host-span token positions."""
import json, random, re

# Exact-match word lists (token decoded, stripped, lowercased). "malice" is the cluster seen in the
# 60-prompt readout of beijing_1 minus domain words the task itself uses (attack, exploit, detection).
GROUPS = {
    "malice": ["malware", "malicious", "stealth", "covert", "hidden", "leak", "leaked", "leaking", "leaks",
               "leakage", "ransom", "trojan", "backdoor", "steal", "stealing", "steals", "spy", "spying",
               "sneak", "secretly", "evil", "hack", "hacked", "hacker", "hackers", "hacking", "targeting",
               "恶意", "木马", "泄露", "泄漏", "窃取", "隐藏", "黑客"],
    "storage": ["storage", "bucket", "buckets", "oss", "alibaba", "aliyun", "阿里云", "阿里", "beijing",
                "北京", "upload", "uploads", "uploading", "cloud", "s3"],
    "placeholder": ["placeholder", "dummy", "fake", "probe", "mock", "example", "xxxx", "sample", "测试"],
    "eval": ["eval", "evaluation", "bench", "benchmark", "task", "tasks", "workspace", "dataset", "grader",
             "sandbox"],
}


def group_ids(tok, words):
    ws = set(words)
    out = []
    for i in range(len(tok)):
        d = tok.decode([i]).strip().lower()
        if d in ws:
            out.append(i)
    return out


def load_row(path):
    row = json.load(open(path))
    n = row.get("prefix_len", 2323)
    return row["token_ids"][:n], n


def categorize(r):
    t = r["tags"]
    if "beijing" in t:
        return "beijing"
    if "routify" in t:
        return "routify"
    if "aliyun" in t:
        return "aliyun_other"
    if "cloud_metadata" in t:
        return "metadata"
    if "no_url" in t:
        return None
    return "benign"


def pick(samples_path, max_per_cat, seed=0):
    rows = [json.loads(l) for l in open(samples_path)]
    by = {}
    for r in rows:
        c = categorize(r)
        if c:
            by.setdefault(c, []).append(r)
    rng = random.Random(seed)
    out = []
    for c, rs in sorted(by.items()):
        rs = rs[:] if c == "beijing" else rng.sample(rs, min(max_per_cat, len(rs)))
        for k, r in enumerate(rs):
            out.append({"name": f"{c}_{k}", "cat": c, "text": r["text"], "host": r["host"]})
    return out


def cont_ids_and_host_span(tok, text, max_chars=90):
    """Token ids of the (truncated) continuation and the [lo, hi) range of continuation-token indices
    whose characters overlap the URL host."""
    text = text[:max_chars]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc.input_ids, enc.offset_mapping
    m = re.search(r"https?://([^/\s\"'\\:]+)", text)
    if not m:
        return ids, None
    a, b = m.span(1)
    span = [k for k, (s, e) in enumerate(offs) if s < b and e > a]
    return ids, (span[0], span[-1] + 1) if span else None


def load_lens(path, layers):
    """Mean Jacobians {L: [2560, 10240] float32} and the number of fitting prompts, from either a
    fit_jlens.py checkpoint (.pt, running sums) or an exported/published .safetensors file."""
    import torch
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(path, "pt") as f:
            return {L: f.get_tensor(f"J.{L}") for L in layers}, int(f.metadata()["n_prompts"])
    st = torch.load(path, weights_only=False)
    return {L: st["sum"][L] / st["n"] for L in layers}, st["n"]
