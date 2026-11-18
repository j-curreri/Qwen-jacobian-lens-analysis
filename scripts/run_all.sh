#!/bin/bash
# End-to-end reproduction. Text generation and teacher-forced analysis only: nothing the model writes
# is executed, and no request is sent to any host it names.
#
#   MODEL_DIR=/path/to/Qwen3.8-Flash-Next-NVFP4 OUT=/big/disk/run scripts/run_all.sh
#
# Optional: MODEL_27B=/path/to/Qwen3.8-27B (for the 27B comparison).
# To skip the fit: SKIP_FIT=1 LENS=/path/to/jacobian_lens.safetensors (from Hugging Face, see README).
# Approximate times on 2x RTX PRO 6000 Blackwell (96 GB): see README.md.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${MODEL_DIR:?set MODEL_DIR}" "${OUT:?set OUT}"
LENS=${LENS:-$OUT/flashnext_jlens_ckpt.pt}
N_FIT=${N_FIT:-100}
N_SAMPLES=${N_SAMPLES:-10000}
mkdir -p "$OUT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
D=$ROOT/data
cd "$ROOT"

step() { name=$1; shift; echo "[$(date +%T)] START $name"; "$@" > "$OUT/$name.log" 2>&1 \
  && echo "[$(date +%T)] DONE $name" || { echo "[$(date +%T)] FAILED $name, see $OUT/$name.log"; exit 1; }; }
# vLLM's memory profiling breaks with expandable segments; the HF-loader scripts need them.
vllm() { env -u PYTORCH_CUDA_ALLOC_CONF python "$@"; }
hf() { PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python "$@"; }

[ -f "$D/row0.json" ] || { echo "run scripts/prepare_data.py first"; exit 1; }

# 1. Reproduction (vLLM)
step sample_512        vllm src/repro_local.py "$D/row0.json" "$MODEL_DIR" "$OUT/samples_512.jsonl" 512 2
if [ -n "${MODEL_27B:-}" ]; then
  RETOKENIZE_FROM="$MODEL_DIR" step sample_27b vllm src/repro_local.py "$D/row0.json" "$MODEL_27B" "$OUT/samples_27b.jsonl" 512 2
fi
step sample_base       vllm src/repro_local.py "$D/row0.json"           "$MODEL_DIR" "$OUT/samples_base.jsonl" "$N_SAMPLES" 2
step sample_ctrl_objective vllm src/repro_local.py "$D/ctrl_objective.json" "$MODEL_DIR" "$OUT/samples_ctrl_objective.jsonl" "$N_SAMPLES" 2
step sample_ctrl_benign    vllm src/repro_local.py "$D/ctrl_benign.json"    "$MODEL_DIR" "$OUT/samples_ctrl_benign.jsonl" "$N_SAMPLES" 2

# 2. Loader validation: vLLM prompt logprobs twice (self-consistency), then the HF loader against them
step vllm_ref          vllm src/vllm_ref.py "$D/row0.json" "$MODEL_DIR" "$OUT/vllm_ref.json"
VLLM_EXTRA='{"enable_prefix_caching": false}' step vllm_ref_alt vllm src/vllm_ref.py "$D/row0.json" "$MODEL_DIR" "$OUT/vllm_ref_alt.json"
step compare_refs      python src/compare_refs.py "$OUT/vllm_ref.json" "$OUT/vllm_ref_alt.json"
step validate_hf       hf src/validate_hf.py "$MODEL_DIR" "$OUT/vllm_ref.json"

# 3. Fit the Jacobian lens (~8 h for 100 prompts; resumable: rerun to continue from the checkpoint)
if [ -z "${SKIP_FIT:-}" ]; then
  step fit_jlens       hf src/fit_jlens.py "$MODEL_DIR" "$D/wikitext_prompts.jsonl" "$LENS" "$N_FIT" 32
  step export_lens     python scripts/export_lens.py "$LENS" "$OUT/jacobian_lens.safetensors"
fi

# 4. Analyses
step readout           hf src/jlens_readout.py "$MODEL_DIR" "$LENS" "$D/row0.json" "$OUT/samples_512.jsonl" "$OUT/jlens_readout.json"
step scan_base         hf src/concept_scan.py "$MODEL_DIR" "$LENS" "$D/row0.json" "$OUT/samples_base.jsonl" "$OUT/scan_base.json" 40
step scan_ctrl_objective hf src/concept_scan.py "$MODEL_DIR" "$LENS" "$D/ctrl_objective.json" "$OUT/samples_ctrl_objective.jsonl" "$OUT/scan_ctrl_objective.json" 40
step scan_ctrl_benign  hf src/concept_scan.py "$MODEL_DIR" "$LENS" "$D/ctrl_benign.json" "$OUT/samples_ctrl_benign.jsonl" "$OUT/scan_ctrl_benign.json" 40
step ablate_base       hf src/ablate.py "$MODEL_DIR" "$LENS" "$D/row0.json" "$OUT/samples_base.jsonl" "$OUT/ablate_base.json" 20
echo "[$(date +%T)] ALL DONE. Redact before sharing: python scripts/redact.py $OUT"
