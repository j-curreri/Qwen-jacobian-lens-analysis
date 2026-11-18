"""J-lens readout + J-space sparse decomposition on Flash-Next along real sampled completions.

usage: python jlens_readout.py MODEL_DIR CKPT ROW_JSON SAMPLES_JSONL OUT_JSON
"""
import json, sys
import torch
import qflash
from jl_common import load_lens

PREFIX = 2323
LAYERS = [18, 21, 24, 27, 30, 33, 36, 39, 42, 45]
TOPK = 12
K_SPARSE = 16


def pick(samples):
    rows = [json.loads(l) for l in open(samples)]
    chosen = {}
    for r in rows:
        if "beijing" in r["tags"]:
            chosen[f"beijing_{sum(k.startswith('beijing') for k in chosen)}"] = r["text"]
    rout = [r["text"] for r in rows if "routify" in r["tags"] and r["text"].startswith('" http://')]
    chosen["routify_0"], chosen["routify_1"] = rout[0], rout[5]
    chosen["imds"] = next(r["text"] for r in rows if "cloud_metadata" in r["tags"])
    for name, host in [("localhost", "localhost"), ("loopback", "127.0.0.1"), ("tmpfiles", "tmpfiles.org"),
                       ("invalid", "zzz.invalid")]:
        t = next((r["text"] for r in rows if r["host"] == host), None)
        if t:
            chosen[name] = t
    return chosen


def gradient_pursuit(h, atoms, atom_norms, k):
    """Greedy nonnegative sparse decomposition of h (d,) over atoms (V, d). Returns [(idx, coef)]
    and the fraction of ||h||^2 explained."""
    sel, coef = [], None
    r = h.clone()
    for _ in range(k):
        scores = (atoms @ r) / atom_norms
        if sel:
            scores[torch.tensor(sel, device=scores.device)] = -float("inf")
        j = int(scores.argmax())
        if scores[j] <= 0:
            break
        sel.append(j)
        D = atoms[sel].float()  # [s, d]
        # nonnegative least squares by projected gradient
        G = D @ D.T
        b = D @ h.float()
        c = torch.clamp(torch.linalg.lstsq(G, b.unsqueeze(1)).solution.squeeze(1), min=0)
        step = 1.0 / torch.linalg.matrix_norm(G, ord=2)
        for _ in range(200):
            c = torch.clamp(c - step * (G @ c - b), min=0)
        coef = c
        r = (h.float() - c @ D).to(h.dtype)
    expl = 1 - (r.float().norm() ** 2 / h.float().norm() ** 2).item()
    return [(i, float(v)) for i, v in zip(sel, coef.tolist())], expl


def main():
    model_dir, ckpt, row_path, samples, out_path = sys.argv[1:6]
    tok = qflash.transformers.AutoTokenizer.from_pretrained(model_dir)
    dec = lambda i: tok.decode([i])
    Js, n = load_lens(ckpt, LAYERS)
    print(f"lens fitted on {n} prompts", flush=True)
    prefix_ids = json.load(open(row_path))["token_ids"][:PREFIX]
    model = qflash.load(model_dir, verbose=False)
    tm = model.model
    WU = model.lm_head.weight  # [V, 2560] on cuda:1
    acts = {}
    for L in LAYERS:
        tm.layers[L].register_forward_hook(lambda m, i, o, L=L: acts.__setitem__(L, o.detach()))

    chosen = pick(samples)
    runs = {}
    for name, text in chosen.items():
        cont = tok.encode(text[:90], add_special_tokens=False)
        ids = prefix_ids + cont
        with torch.no_grad():
            tm(input_ids=torch.tensor([ids], device="cuda:0"), use_cache=False)
        runs[name] = {"ids": ids, "n_cont": len(cont), "acts": {L: acts[L][0].to("cpu") for L in LAYERS}}
        acts.clear()
        print("ran", name, flush=True)

    # shared prefix positions: from the start of the bash command to the end of the prefix
    pref_text = tok.decode(prefix_ids[-80:])
    shared_positions = list(range(PREFIX - 80, PREFIX))

    report = {"n_prompts": n, "layers": LAYERS, "shared_prefix": {}, "samples": {}}
    for L in LAYERS:
        J = Js[L].to("cuda:1", torch.float32)  # [2560, 10240]
        atoms = (WU.float() @ J).to(torch.bfloat16)  # [V, 10240]
        atom_norms = atoms.float().norm(dim=1).clamp(min=1e-6)

        def readout(h):
            h = h.to("cuda:1", torch.float32)
            logits = WU.float() @ (J @ h)
            top = logits.topk(TOPK)
            decomp, expl = gradient_pursuit(h.to(torch.bfloat16), atoms, atom_norms.to(torch.bfloat16), K_SPARSE)
            return {"lens": [dec(i) for i in top.indices.tolist()],
                    "jspace": [(dec(i), round(c, 3)) for i, c in sorted(decomp, key=lambda x: -x[1])],
                    "jspace_expl": round(expl, 4)}

        any_run = next(iter(runs.values()))
        for p in shared_positions:
            report["shared_prefix"].setdefault(str(p), {"token": dec(any_run["ids"][p]), "layers": {}})
            report["shared_prefix"][str(p)]["layers"][L] = readout(any_run["acts"][L][p])
        for name, run in runs.items():
            entry = report["samples"].setdefault(name, {"text": chosen[name][:90], "cells": {}})
            for p in range(PREFIX - 1, PREFIX - 1 + min(run["n_cont"], 30)):
                c = entry["cells"].setdefault(str(p), {"token": dec(run["ids"][p]),
                                                       "next": dec(run["ids"][p + 1]) if p + 1 < len(run["ids"]) else None,
                                                       "layers": {}})
                c["layers"][L] = readout(run["acts"][L][p])
        del atoms, atom_norms, J
        torch.cuda.empty_cache()
        print(f"layer {L} done", flush=True)
    json.dump(report, open(out_path, "w"), ensure_ascii=False, indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main()
