"""Causal test: project concept directions (J-space atoms) out of the residual stream and measure how
the teacher-forced log-probability of each sampled completion's URL host changes.

Atoms for token v at layer L: a_v = (W_U J_L)[v]  (the direction the J-lens reads v from).
Ablation of group G at layer L: h <- h - Q Q^T h, Q = orthonormal basis of span{a_v : v in G},
applied to the outputs of layers ABL_LAYERS at positions >= start ("cont": from the last prefix token,
i.e. only while the URL is being written; "all": every position, including the task prompt).

Conditions: two unablated baselines (their difference = run-to-run noise), malice@cont, malice@all,
storage@cont (positive control: should hurt Alibaba/Beijing hosts), placeholder@cont, and three random
word sets of the same size @cont (specificity control).

usage: python ablate.py MODEL_DIR CKPT ROW_JSON SAMPLES_JSONL OUT_JSON [MAX_PER_CAT]
"""
import json, random, statistics, sys
import torch
import qflash
from jl_common import GROUPS, group_ids, load_lens, load_row, pick, cont_ids_and_host_span

ABL_LAYERS = list(range(33, 40))


def main():
    model_dir, ckpt, row_path, samples, out_path = sys.argv[1:6]
    max_per_cat = int(sys.argv[6]) if len(sys.argv) > 6 else 20
    tok = qflash.transformers.AutoTokenizer.from_pretrained(model_dir)
    prefix_ids, PREFIX = load_row(row_path)
    Js, _ = load_lens(ckpt, ABL_LAYERS)
    model = qflash.load(model_dir, verbose=False)
    tm = model.model
    WU = model.lm_head.weight

    sets = {g: group_ids(tok, GROUPS[g]) for g in ("malice", "storage", "placeholder")}
    words = [i for i in range(len(tok)) if (lambda d: d.isascii() and d.isalpha() and len(d) >= 4)(
        tok.decode([i]).strip())]
    excluded = set().union(*sets.values())
    words = [i for i in words if i not in excluded]
    for r in range(3):
        sets[f"random{r}"] = random.Random(100 + r).sample(words, len(sets["malice"]))
    print({k: len(v) for k, v in sets.items()}, flush=True)

    bases = {}
    for name, ids in sets.items():
        for L in ABL_LAYERS:
            dev = next(iter(list(tm.layers[L].parameters()) + list(tm.layers[L].buffers()))).device
            A =WU[torch.tensor(ids, device=WU.device)].float() @ Js[L].to(WU.device)  # [k, 10240]
            U, S, _ = torch.linalg.svd(A.T, full_matrices=False)
            Q = U[:, S > 1e-3 * S[0]]
            bases[(name, L)] = Q.to(dev)
    print("subspace ranks:", {k: bases[(k, ABL_LAYERS[0])].shape[1] for k in sets}, flush=True)

    ablation = {"group": None, "start": 0}
    removed = []

    def hook(L):
        def f(m, i, o):
            g = ablation["group"]
            if g is None:
                return None
            Q = bases[(g, L)].to(o.device)
            o2 = o.clone()
            seg = o2[:, ablation["start"]:].float()
            proj = (seg @ Q) @ Q.T
            removed.append((proj.norm() / seg.norm()).item())
            o2[:, ablation["start"]:] = (seg - proj).to(o.dtype)
            return o2
        return f

    for L in ABL_LAYERS:
        tm.layers[L].register_forward_hook(hook(L))

    conditions = [("base", None, 0), ("base_repeat", None, 0),
                  ("malice@cont", "malice", PREFIX - 1), ("malice@all", "malice", 0),
                  ("storage@cont", "storage", PREFIX - 1), ("placeholder@cont", "placeholder", PREFIX - 1),
                  ("random0@cont", "random0", PREFIX - 1), ("random1@cont", "random1", PREFIX - 1),
                  ("random2@cont", "random2", PREFIX - 1)]

    chosen = [s for s in pick(samples, max_per_cat) if cont_ids_and_host_span(tok, s["text"])[1]]
    print(f"{len(chosen)} samples", flush=True)
    results = []
    for si, s in enumerate(chosen):
        cont, (lo, hi) = cont_ids_and_host_span(tok, s["text"])
        ids = prefix_ids + cont
        x = torch.tensor([ids], device="cuda:0")
        tgt = torch.tensor(cont)
        rec = {"name": s["name"], "cat": s["cat"], "host": s["host"], "span": [lo, hi], "lp": {}}
        for cname, g, start in conditions:
            ablation.update(group=g, start=start)
            removed.clear()
            with torch.no_grad():
                logits = model(input_ids=x, use_cache=False,
                               logits_to_keep=len(cont) + 1).logits[0, :-1].float()  # predicts cont tokens
            lp = torch.log_softmax(logits, -1).cpu().gather(1, tgt[:, None])[:, 0]
            rec["lp"][cname] = {"host": lp[lo:hi].sum().item(), "host_first": lp[lo].item(),
                                "cont": lp.sum().item(),
                                "removed_frac": statistics.mean(removed) if removed else 0.0}
        ablation.update(group=None)
        results.append(rec)
        if si % 10 == 0:
            b = rec["lp"]
            print(f"{si + 1}/{len(chosen)} {s['name']}: host lp base {b['base']['host']:.2f} "
                  f"repeat {b['base_repeat']['host']:.2f} malice@cont {b['malice@cont']['host']:.2f} "
                  f"storage@cont {b['storage@cont']['host']:.2f}", flush=True)

    summary = {}
    cats = sorted({r["cat"] for r in results})
    for cname, _, _ in conditions[1:]:
        row = {}
        for c in cats:
            rs = [r for r in results if r["cat"] == c]
            for metric in ("host", "host_first"):
                d = [r["lp"][cname][metric] - r["lp"]["base"][metric] for r in rs]
                m = statistics.mean(d)
                se = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")
                row.setdefault(c, {"n": len(rs)})[f"d_{metric}"] = [round(m, 3), round(se, 3)]
            row[c]["removed_frac"] = round(statistics.mean(r["lp"][cname]["removed_frac"] for r in rs), 4)
        summary[cname] = row
    json.dump({"abl_layers": ABL_LAYERS, "set_sizes": {k: len(v) for k, v in sets.items()},
               "summary": summary, "results": results}, open(out_path, "w"), indent=1, ensure_ascii=False)
    print("mean change in log-prob vs base, [mean, s.e.] (host span total / first host token):")
    for cname, row in summary.items():
        print(f"  {cname:18s} " + "  ".join(f"{c}(n={v['n']}): {v['d_host'][0]:+.2f}±{v['d_host'][1]:.2f} / "
                                              f"{v['d_host_first'][0]:+.2f}" for c, v in row.items()))
    print("saved", out_path)


if __name__ == "__main__":
    main()
