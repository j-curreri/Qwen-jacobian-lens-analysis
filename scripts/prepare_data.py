"""Fetch the inputs that this repo does not redistribute, and derive everything else from them.

  data/row0.json              Cohere's trace (token_ids, trace, model) + prefix_len=2323
  data/prefix.txt             the 2,323-token prefix decoded to text (ends at ...PROBE_PAYLOAD_ABC123)
  data/ctrl_objective.json    context control 1 (security setup kept, objective swapped)
  data/ctrl_benign.json       context control 2 (fully benign task)
  data/wikitext_prompts.jsonl 1,200 WikiText-103 passages (>=150 words) used to fit the J-lens

Sources (downloaded from the Hugging Face Hub unless local copies are passed):
  CohereLabs/10-2026-qwen-investigation-traces  data/train-00000-of-00001.parquet
  Salesforce/wikitext  wikitext-103-raw-v1/train-00000-of-00002.parquet

usage: python scripts/prepare_data.py MODEL_DIR [--cohere-parquet P] [--wikitext-parquet P] [--out data]
"""
import argparse, json, os, random, subprocess, sys

import pyarrow.parquet as pq

PREFIX_LEN = 2323  # Cohere's prefill point: the tokens up to and including "PROBE_PAYLOAD_ABC123"

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("--cohere-parquet")
ap.add_argument("--wikitext-parquet")
ap.add_argument("--out", default="data")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

if not a.cohere_parquet or not a.wikitext_parquet:
    from huggingface_hub import hf_hub_download
if not a.cohere_parquet:
    a.cohere_parquet = hf_hub_download("CohereLabs/10-2026-qwen-investigation-traces",
                                       "data/train-00000-of-00001.parquet", repo_type="dataset")
if not a.wikitext_parquet:
    a.wikitext_parquet = hf_hub_download("Salesforce/wikitext",
                                         "wikitext-103-raw-v1/train-00000-of-00002.parquet",
                                         repo_type="dataset")

row = pq.read_table(a.cohere_parquet).to_pylist()[0]
row["prefix_len"] = PREFIX_LEN
json.dump(row, open(os.path.join(a.out, "row0.json"), "w"))
print(f"row0.json: {len(row['token_ids'])} token ids, prefix_len {PREFIX_LEN}")

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model_dir)
prefix = tok.decode(row["token_ids"][:PREFIX_LEN])
assert prefix.endswith('--data-binary "PROBE_PAYLOAD_ABC123'), prefix[-80:]
assert tok.encode(prefix, add_special_tokens=False) == row["token_ids"][:PREFIX_LEN], "prefix does not round-trip"
open(os.path.join(a.out, "prefix.txt"), "w").write(prefix)

here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(here, "..", "src", "make_controls.py"), a.model_dir,
                os.path.join(a.out, "prefix.txt"), a.out], check=True)

lines = pq.read_table(a.wikitext_parquet).column("text").to_pylist()
paras = [l.strip() for l in lines if l.strip() and not l.strip().startswith("=") and len(l.split()) >= 150]
random.seed(0)
random.shuffle(paras)
with open(os.path.join(a.out, "wikitext_prompts.jsonl"), "w") as f:
    for p in paras[:1200]:
        f.write(json.dumps({"text": p}) + "\n")
print(f"wikitext_prompts.jsonl: 1200 of {len(paras)} long passages")
