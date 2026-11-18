"""Fit a Jacobian lens on Qwen3.8-Flash-Next (NVFP4) with the jlens estimator, adapted to
hyper-connections.

Source: output of decoder layer l (4 hyper-connection streams, 10240 dims).
Target: output of the final hyper_connection_mixer (2560 dims), which feeds lm_head directly
(this model has no separate final norm; the mixer contains it).
J_l (2560 x 10240) = mean over prompts and source positions p of sum_{p' >= p} d target[p'] / d h_l[p],
exactly the reduction in jlens.fitting.jacobian_for_prompt (one-hot cotangent at every valid
target position, mean over valid source positions).
Readout: lens_l(h) = lm_head(J_l @ h).

usage: python fit_jlens.py MODEL_DIR PROMPTS_JSONL CKPT_PATH N_PROMPTS [DIM_BATCH]
"""
import json, math, os, sys, time
import torch
import qflash

SEQ = 128
SKIP_FIRST = 16


def self_check_indexer(model, ids):
    """The fast indexer path must agree with the reference indexer as well as the reference agrees
    with itself (forward passes are nondeterministic: atomic index_add + routing sensitivity)."""
    def lsm(fast):
        qflash.mq.Qwen4ExpTextQSAIndexer.forward = qflash._fast_indexer_forward if fast else qflash._orig_indexer_forward
        with torch.no_grad():
            return torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0].float(), -1)
    fast, ref1, ref2 = lsm(True), lsm(False), lsm(False)
    qflash.mq.Qwen4ExpTextQSAIndexer.forward = qflash._fast_indexer_forward
    kl = lambda a, b: (a.exp() * (a - b)).sum(-1).mean().item()
    kl_fr, kl_rr = kl(ref1, fast), kl(ref1, ref2)
    print(f"indexer self-check: meanKL(ref,fast)={kl_fr:.2e}  meanKL(ref,ref)={kl_rr:.2e}", flush=True)
    if kl_fr > 3 * kl_rr + 1e-3:
        raise SystemExit("fast indexer path disagrees with reference beyond run-to-run noise")


def main():
    model_dir, prompts_path, ckpt, n_prompts = sys.argv[1:5]
    n_prompts = int(n_prompts)
    B = int(sys.argv[5]) if len(sys.argv) > 5 else 32
    tok = qflash.transformers.AutoTokenizer.from_pretrained(model_dir)
    model = qflash.load(model_dir, verbose=False)
    tm = model.model
    n_layers, d_in, d_out = len(tm.layers), 4 * tm.config.hidden_size, tm.config.hidden_size

    texts = [json.loads(l)["text"] for l in open(prompts_path)]
    prompt_ids = []
    for t in texts:
        ids = tok(t, add_special_tokens=False).input_ids
        if len(ids) >= SEQ:
            prompt_ids.append(ids[:SEQ])
        if len(prompt_ids) >= n_prompts:
            break
    print(f"{len(prompt_ids)} prompts of {SEQ} tokens; B={B}; layers={n_layers}", flush=True)

    self_check_indexer(model, torch.tensor([prompt_ids[0]], device="cuda:0"))

    state = {"sum": {L: torch.zeros(d_out, d_in) for L in range(n_layers)}, "n": 0, "done": []}
    if os.path.exists(ckpt):
        state = torch.load(ckpt, weights_only=False)
        print(f"resumed from {ckpt}: {state['n']} prompts done", flush=True)

    acts = {}
    tm.embed_tokens.register_forward_hook(lambda m, i, o: o.detach().requires_grad_(True))
    for L in range(n_layers):
        tm.layers[L].register_forward_hook(lambda m, i, o, L=L: acts.__setitem__(L, o))
    tm.hyper_connection_mixer.register_forward_hook(lambda m, i, o: acts.__setitem__("target", o))

    valid = torch.arange(SKIP_FIRST, SEQ - 1)
    n_passes = math.ceil(d_out / B)
    for pi, ids in enumerate(prompt_ids):
        if pi in state["done"]:
            continue
        t0 = time.time()
        with torch.enable_grad():
            tm(input_ids=torch.tensor([ids], device="cuda:0").expand(B, -1), use_cache=False)
            target = acts["target"]
            sources = [acts[L] for L in range(n_layers)]
            t_fwd = time.time() - t0
            vt = valid.to(target.device)
            bidx = torch.arange(B, device=target.device)
            ct = torch.zeros_like(target)
            rows = {L: torch.empty(d_out, d_in) for L in range(n_layers)}
            for p, d0 in enumerate(range(0, d_out, B)):
                n = min(B, d_out - d0)
                ct.zero_()
                ct[bidx[:n, None], vt[None, :], d0 + bidx[:n, None]] = 1.0
                grads = torch.autograd.grad(target, sources, ct, retain_graph=(p < n_passes - 1))
                for L, g in enumerate(grads):
                    rows[L][d0:d0 + n] = g[:n, valid.to(g.device), :].float().mean(1).cpu()
                del grads
        acts.clear()
        for L in range(n_layers):
            state["sum"][L] += rows[L]
        state["n"] += 1
        state["done"].append(pi)
        dt = time.time() - t0
        print(f"prompt {pi + 1}/{len(prompt_ids)}  fwd {t_fwd:.1f}s  total {dt:.0f}s  "
              f"eta {(len(prompt_ids) - len(state['done'])) * dt / 3600:.1f}h", flush=True)
        if state["n"] % 5 == 0 or len(state["done"]) == len(prompt_ids):
            tmp = ckpt + ".tmp"
            torch.save(state, tmp)
            os.replace(tmp, ckpt)
            print(f"  checkpoint saved ({state['n']} prompts)", flush=True)


if __name__ == "__main__":
    main()
