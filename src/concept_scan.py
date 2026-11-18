"""Quantitative J-lens concept scan over many sampled completions.

For every selected completion, teacher-force prefix + completion, read the J-lens at LAYERS for the
continuation positions, and measure for each concept group (jl_common.GROUPS):
  mass  = J-lens softmax probability on the group's tokens
  hit   = any group token in the lens top-12
summarized per sample over the HOST WINDOW (position predicting the first host token through the last
host token) as max-mass and any-hit. Baselines: other host categories, and equal-length sliding windows
over the last 400 prefix positions.

usage: python concept_scan.py MODEL_DIR CKPT ROW_JSON SAMPLES_JSONL OUT_JSON [MAX_PER_CAT]
"""
import json, sys, statistics
import torch
import qflash
from jl_common import GROUPS, group_ids, load_lens, load_row, pick, cont_ids_and_host_span

LAYERS = [18, 21, 24, 27, 30, 33, 36, 39, 42, 45]
TOPK = 12
PREFIX_BASE = 400


def main():
    model_dir, ckpt, row_path, samples, out_path = sys.argv[1:6]
    max_per_cat = int(sys.argv[6]) if len(sys.argv) > 6 else 40
    tok = qflash.transformers.AutoTokenizer.from_pretrained(model_dir)
    gids = {g: torch.tensor(group_ids(tok, ws)) for g, ws in GROUPS.items()}
    print({g: len(v) for g, v in gids.items()}, flush=True)
    prefix_ids, PREFIX = load_row(row_path)
    Js, n_fit = load_lens(ckpt, LAYERS)
    model = qflash.load(model_dir, verbose=False)
    tm = model.model
    WU = model.lm_head.weight
    dev = WU.device
    acts = {}
    for L in LAYERS:
        tm.layers[L].register_forward_hook(lambda m, i, o, L=L: acts.__setitem__(L, o.detach()))

    chosen = pick(samples, max_per_cat)
    print(f"{len(chosen)} samples:", {c: sum(s['cat'] == c for s in chosen) for c in {s['cat'] for s in chosen}},
          flush=True)
    # cells: (sample index or -1 for prefix, position); vectors stored per layer
    cells, vecs = [], {L: [] for L in LAYERS}
    for si, s in enumerate(chosen):
        cont, span = cont_ids_and_host_span(tok, s["text"])
        s["span"] = span
        ids = prefix_ids + cont
        with torch.no_grad():
            tm(input_ids=torch.tensor([ids], device="cuda:0"), use_cache=False)
        positions = list(range(PREFIX - 1, PREFIX - 1 + min(len(cont), 30)))
        if si == 0:
            positions = list(range(PREFIX - 1 - PREFIX_BASE, PREFIX - 1)) + positions
        s["tokens"] = {p: tok.decode([ids[p]]) for p in positions}
        for p in positions:
            cells.append((si if p >= PREFIX - 1 else -1, p))
        for L in LAYERS:
            vecs[L].append(acts[L][0, positions].to("cpu", torch.bfloat16))
        acts.clear()
        if si % 20 == 0:
            print(f"forward {si + 1}/{len(chosen)}", flush=True)

    # mass[g][cell, layer], hit[g][cell, layer], top tokens for beijing cells
    nc = len(cells)
    mass = {g: torch.zeros(nc, len(LAYERS)) for g in GROUPS}
    hit = {g: torch.zeros(nc, len(LAYERS), dtype=torch.bool) for g in GROUPS}
    tops = {}
    want_top = {k for k, (si, p) in enumerate(cells) if si >= 0 and chosen[si]["cat"] in ("beijing",)}
    WUf = WU.float()
    for li, L in enumerate(LAYERS):
        J = Js[L].to(dev, torch.float32)
        H = torch.cat(vecs[L]).to(dev, torch.float32)
        for a in range(0, nc, 256):
            logits = (H[a:a + 256] @ J.T) @ WUf.T  # [c, V]
            lse = logits.logsumexp(-1)
            top = logits.topk(TOPK).indices
            for g, ids in gids.items():
                ids = ids.to(dev)
                mass[g][a:a + 256, li] = (logits[:, ids].logsumexp(-1) - lse).exp().cpu()
                hit[g][a:a + 256, li] = torch.isin(top, ids).any(-1).cpu()
            for k in range(a, min(a + 256, nc)):
                if k in want_top and L in (30, 33, 36, 39):
                    tops.setdefault(k, {})[L] = [tok.decode([i]) for i in top[k - a].tolist()]
        del J, H
        vecs[L] = None
        torch.cuda.empty_cache()
        print(f"layer {L} done", flush=True)

    # per-sample host-window summaries
    cell_index = {c: k for k, c in enumerate(cells)}
    per_sample, win_lens = [], []
    for si, s in enumerate(chosen):
        if not s["span"]:
            continue
        lo, hi = s["span"]
        ks = [cell_index[(si, p)] for p in range(PREFIX - 1 + lo, PREFIX + hi) if (si, p) in cell_index]
        if not ks:
            continue
        win_lens.append(len(ks))
        rec = {"name": s["name"], "cat": s["cat"], "host": s["host"], "text": s["text"][:90]}
        for g in GROUPS:
            rec[f"{g}_maxmass"] = float(mass[g][ks].max())
            rec[f"{g}_hit"] = bool(hit[g][ks].any())
            rec[f"{g}_hitfrac"] = float(hit[g][ks].float().mean())
        if s["cat"] == "beijing":
            rec["cells"] = [{"pos": cells[k][1], "token": s["tokens"][cells[k][1]],
                             "malice_mass": [round(float(x), 5) for x in mass["malice"][k]],
                             "top": tops.get(k, {})} for k in ks]
        per_sample.append(rec)

    # prefix baseline: sliding windows of the median host-window length
    w = int(statistics.median(win_lens))
    pk = [k for k, (si, p) in enumerate(cells) if si == -1]
    base = {g: {"maxmass": [], "hit": []} for g in GROUPS}
    for a in range(0, len(pk) - w + 1):
        ks = pk[a:a + w]
        for g in GROUPS:
            base[g]["maxmass"].append(float(mass[g][ks].max()))
            base[g]["hit"].append(bool(hit[g][ks].any()))

    def q(xs, f):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(f * len(xs)))]

    summary = {}
    cats = sorted({r["cat"] for r in per_sample})
    for g in GROUPS:
        row = {}
        for c in cats:
            rs = [r for r in per_sample if r["cat"] == c]
            mm = [r[f"{g}_maxmass"] for r in rs]
            row[c] = {"n": len(rs), "hit_rate": sum(r[f"{g}_hit"] for r in rs) / len(rs),
                      "maxmass_median": statistics.median(mm), "maxmass_mean": sum(mm) / len(mm)}
        bm = base[g]["maxmass"]
        row["prefix_windows"] = {"n": len(bm), "hit_rate": sum(base[g]["hit"]) / len(bm),
                                 "maxmass_median": statistics.median(bm), "maxmass_p95": q(bm, 0.95)}
        summary[g] = row

    # rank test: beijing vs each other category on malice max-mass (Mann-Whitney U, normal approx.)
    tests = {}
    for g in GROUPS:
        bj = [r[f"{g}_maxmass"] for r in per_sample if r["cat"] == "beijing"]
        for c in cats:
            if c == "beijing" or not bj:
                continue
            ot = [r[f"{g}_maxmass"] for r in per_sample if r["cat"] == c]
            U = sum((x > y) + 0.5 * (x == y) for x in bj for y in ot)
            n1, n2 = len(bj), len(ot)
            mu, sd = n1 * n2 / 2, (n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5
            z = (U - mu) / sd
            p = 0.5 * torch.erfc(torch.tensor(abs(z) / 2 ** 0.5)).item() * 2
            tests[f"{g}: beijing vs {c}"] = {"AUC": U / (n1 * n2), "z": z, "p_two_sided": p}

    out = {"n_fit_prompts": n_fit, "layers": LAYERS, "window_len_median": w, "group_sizes":
           {g: len(v) for g, v in gids.items()}, "summary": summary, "tests": tests, "samples": per_sample}
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"summary": summary, "tests": tests}, indent=1, ensure_ascii=False))
    print("saved", out_path)


if __name__ == "__main__":
    main()
