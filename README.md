# Why does Qwen3.8-Flash-Next upload to Beijing buckets? A Jacobian-lens look

John Curreri, November 2026.

Code, data products and paper for a follow-up to Cohere Labs' report that Qwen3.8-Flash-Next, in an
agentic security task, tries to `curl -X PUT` a probe file to Alibaba Cloud OSS buckets, including
buckets in `oss-cn-beijing`
([blog post](https://cohere.com/blog/concerning-behaviour-popular-open-weight-ai-model),
[trace dataset](https://huggingface.co/datasets/CohereLabs/10-2026-qwen-investigation-traces)).

Cohere showed the behaviour and released the trace. This repo adds:

| | Cohere Labs | this repo |
|---|---|---|
| Model / inference | Qwen3.8-Flash-Next, hosted inference | `nvidia/Qwen3.8-Flash-Next-NVFP4`, local and offline (vLLM), plus Qwen3.8-27B |
| Host statistics | individual samples | 10,000 samples per prompt with host categories |
| Context controls | none | the same `curl` command under a neutral security objective and a fully benign task |
| Internals | none | an autograd-capable HF loader for the NVFP4 checkpoint, validated against vLLM |
| Interpretability | none | a Jacobian lens fitted for this hyper-connection MoE, J-lens readouts along the URL, a concept scan over 167–213 completions per prompt, and a causal ablation with random-subspace controls |

Paper: [`paper/index.html`](paper/index.html) (web) and [`paper/paper.pdf`](paper/paper.pdf).
Fitted lens: [maxamp/Qwen3.8-Flash-Next-NVFP4-jacobian-lens](https://huggingface.co/maxamp/Qwen3.8-Flash-Next-NVFP4-jacobian-lens) on Hugging Face.

**Main result.** The Beijing hosts come from a memorized "upload a probe to a cloud bucket" habit:
storage/bucket concepts carry the decision and causally support it, and the region is filled in late
(`-be` → `ijing` reads out Alibaba / 阿里云 / 华北 / Beijing). Malware/stealth concepts do appear in
the original security-task context, but (a) they appear only after the bucket and region are
written, (b) when the task is benign their strongest reading drops from 17% to 1.4% of lens probability while the Beijing rate stays at 0.5%,
and (c) projecting them out hurts Beijing hosts no more than projecting out random word sets of the
same size.

## Safety and redaction

- Text generation and teacher-forced analysis only. Nothing the model writes is executed and no
  request goes to any host it names. All model runs are offline (`HF_HUB_OFFLINE=1`).
- The model writes Alibaba AccessKeyIds (`LTAI…`), OSS signatures, JWTs and similar strings into its
  URLs. Some may be memorized real credentials. **Do not test them.** Every file in `results/` has
  been passed through [`scripts/redact.py`](scripts/redact.py); run it on your own outputs before
  sharing them.
- Cohere's trace is not redistributed here; `scripts/prepare_data.py` downloads it.

## Requirements

- 2 GPUs with ≥96 GB each (tested on 2× RTX PRO 6000 Blackwell, sm_120). The HF loader puts layers
  0–23 on `cuda:0` and 24–47 on `cuda:1`.
- ≥128 GB system RAM (the FP8 n-gram embedding table, about 51 GB, stays in CPU RAM).
- About 135 GB of disk for the checkpoint, 5 GB for the fitted lens, and 20 GB for outputs.
- Python 3.12, CUDA 13.x, the packages in [`requirements.txt`](requirements.txt).

## Reproduce

```bash
pip install -r requirements.txt
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --local-dir /models/Qwen3.8-Flash-Next-NVFP4
python scripts/prepare_data.py /models/Qwen3.8-Flash-Next-NVFP4          # writes data/
MODEL_DIR=/models/Qwen3.8-Flash-Next-NVFP4 OUT=/big/disk/run scripts/run_all.sh
python scripts/redact.py /big/disk/run
```

`prepare_data.py` checks that the 2,323-token prefix round-trips through the tokenizer and that it
ends at `--data-binary "PROBE_PAYLOAD_ABC123`. The control prompts are built from that prefix by
[`src/make_controls.py`](src/make_controls.py).

| Step | Script | Time (2× RTX PRO 6000) |
|---|---|---|
| 512 samples (NVFP4, optionally 27B) | `src/repro_local.py` | ~5 min each, mostly engine start |
| 10,000 samples × 3 prompts | `src/repro_local.py` | ~10–15 min each |
| Loader validation | `src/vllm_ref.py`, `src/compare_refs.py`, `src/validate_hf.py` | ~15 min |
| Fit the J-lens (100 × 128-token WikiText passages, 48 layers) | `src/fit_jlens.py` | ~8.3 h, resumable |
| Readout along two Beijing completions | `src/jlens_readout.py` | ~11 min |
| Concept scan × 3 prompts | `src/concept_scan.py` | 25–55 min each |
| Ablation (133 completions × 9 conditions) | `src/ablate.py` | ~4.7 h |

To skip the 8-hour fit, use the published lens:

```bash
hf download maxamp/Qwen3.8-Flash-Next-NVFP4-jacobian-lens jacobian_lens.safetensors --local-dir /big/disk/lens
SKIP_FIT=1 LENS=/big/disk/lens/jacobian_lens.safetensors MODEL_DIR=... OUT=... scripts/run_all.sh
```

The analysis scripts accept either that file or a `fit_jlens.py` checkpoint. `scripts/export_lens.py` converts a checkpoint to the published format.

Sampling uses vLLM's nondeterministic kernels, so exact samples differ between runs. Rates should
agree within binomial error (Beijing ≈ 0.5% ± 0.07% at n = 10,000). The HF loader agrees with vLLM
as well as vLLM agrees with itself (top-1 agreement 0.83 vs 0.81 across the 2,371 trace positions;
see `results/validation/`).

## Layout

```
src/qflash.py           NVFP4 → HF transformers loader with autograd (grouped-GEMM experts, fla kernels)
src/repro_local.py      offline vLLM sampling of the prefilled trace, host classification
src/fit_jlens.py        Jacobian-lens fit adapted to hyper-connections (2560 × 10240 per layer)
src/jlens_readout.py    J-lens + sparse J-space readout along chosen completions
src/concept_scan.py     concept-group mass in the lens over many completions, with tests
src/ablate.py           project J-space concept subspaces out of layers 33–39, measure host log-prob
src/jl_common.py        concept word lists (fixed before the scans were run), sample selection
src/make_controls.py    context-control prompts
src/{logit_lens,vllm_ref,compare_refs,validate_hf,determinism_check,grad_check}.py   checks
scripts/                prepare_data.py, run_all.sh, redact.py, export_lens.py, build_pdf.py
results/samples/        all generated completions (redacted) and host-rate summaries
results/lens/           fit log, readout, concept scans, ablation
results/validation/     loader-vs-vLLM and determinism checks
paper/index.html        the paper (web)
paper/paper.pdf         the paper (PDF; rebuild with scripts/build_pdf.py, needs weasyprint)
```

## Credits

- Cohere Labs, for finding the behaviour and releasing the trace and token IDs.
- The Jacobian-lens estimator follows Anthropic's reference implementation
  ([anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens), Apache-2.0) and the paper
  [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html).
  It was reimplemented here for hyper-connections; no code was copied.
- WikiText-103 (Salesforce/wikitext) for the lens-fitting corpus.

## Licence

MIT (see [`LICENSE`](LICENSE)) for the code, results and paper. The published lens is also MIT, but
it is computed from `nvidia/Qwen3.8-Flash-Next-NVFP4`, which is governed by the NVIDIA Open Model
License and the Qwen Community License. Model outputs in `results/samples/` were generated under
those terms too.
