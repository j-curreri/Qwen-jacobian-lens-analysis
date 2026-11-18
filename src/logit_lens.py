"""Logit lens on Qwen3.8-Flash-Next at the URL decision point and along chosen completions.

For hyper-connections, the lens on a layer-l residual (4 x 2560 streams) is the model's own
readout head: hyper_connection_mixer (incl. hc_norm) -> lm_head.

usage: python logit_lens.py MODEL_DIR ROW_JSON SAMPLES_JSONL OUT_JSON
"""
import json, sys
import torch
import qflash

PREFIX = 2323
LAYERS = list(range(0, 48, 3)) + [47]
TOPK = 10


def pick(samples):
    rows = [json.loads(l) for l in open(samples)]
    chosen = {}
    for r in rows:
        if "beijing" in r["tags"]:
            chosen.setdefault(f"beijing_{len([k for k in chosen if k.startswith('beijing')])}", r["text"])
    chosen["routify"] = next(r["text"] for r in rows if "routify" in r["tags"] and r["text"].startswith('" http://'))
    chosen["imds"] = next(r["text"] for r in rows if "cloud_metadata" in r["tags"])
    for name, host in [("localhost", "localhost"), ("loopback", "127.0.0.1"), ("tmpfiles", "tmpfiles.org"),
                       ("invalid", "zzz.invalid")]:
        t = next((r["text"] for r in rows if r["host"] == host), None)
        if t:
            chosen[name] = t
    return chosen


def main():
    model_dir, row_path, samples, out_path = sys.argv[1:5]
    tok = qflash.transformers.AutoTokenizer.from_pretrained(model_dir)
    prefix_ids = json.load(open(row_path))["token_ids"][:PREFIX]
    model = qflash.load(model_dir)
    tm = model.model
    mixer, head = tm.hyper_connection_mixer, model.lm_head

    acts = {}
    for L in LAYERS:
        tm.layers[L].register_forward_hook(lambda m, i, o, L=L: acts.__setitem__(L, o.detach()))

    def lens(h):  # h: [T, 10240] -> logprobs [T, V]
        x = mixer(h.to("cuda:1").unsqueeze(0))
        x = x[0] if isinstance(x, tuple) else x
        return torch.log_softmax(head(x[0]).float(), -1)

    report = {}
    for name, text in pick(samples).items():
        cont = tok.encode(text[:90], add_special_tokens=False)
        ids = torch.tensor([prefix_ids + cont], device="cuda:0")
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
            # positions: last prefix token, then each continuation token (lens predicts the NEXT token)
            positions = list(range(PREFIX - 1, PREFIX - 1 + min(len(cont), 28)))
            entry = []
            for p in positions:
                cell = {"pos": p, "token": tok.decode(ids[0, p]), "next_actual": tok.decode(ids[0, p + 1]) if p + 1 < ids.shape[1] else None, "layers": {}}
                for L in LAYERS:
                    lp = lens(acts[L][0, p:p + 1])[0]
                    top = lp.topk(TOPK)
                    cell["layers"][L] = [(tok.decode([i]), round(v, 2)) for v, i in zip(top.values.tolist(), top.indices.tolist())]
                entry.append(cell)
        report[name] = {"text": text[:90], "cells": entry}
        print(f"\n######## {name}: {text[:90]!r}", flush=True)
        for cell in entry:
            print(f"-- pos {cell['pos']} tok={cell['token']!r} -> actual next {cell['next_actual']!r}")
            for L in LAYERS:
                if L % 6 == 0 or L == 47:
                    print(f"   L{L:02d}: " + " | ".join(f"{t!r}" for t, _ in cell["layers"][L][:7]))
    json.dump(report, open(out_path, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
