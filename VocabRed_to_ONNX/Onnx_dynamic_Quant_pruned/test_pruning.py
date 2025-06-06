#!/usr/bin/env python
import os
import sys
import torch
from transformers import MarianTokenizer
from core.marian import MarianOnnx

MODEL_DIR = "/home/njuttu_umass_edu/on-DeviceNLP/outs/pruning/Helsinki-NLP_opus-mt-fr-en"

# 1) Check that the tokenizer files are actually present on disk:
required_files = [
    "config.json",
    "tokenizer_config.json",
    # Either vocab.json (if using JSON-based tokenizer) or SentencePiece files:
    # (At least one of these must exist)
    "vocab.json",
    "source.spm",
    "target.spm",
]
missing = []
for name in required_files:
    # We allow either (vocab.json) OR (source.spm & target.spm)
    full = os.path.join(MODEL_DIR, name)
    if name in ("vocab.json",):
        if not os.path.isfile(full):
            missing.append(name)
    elif name in ("source.spm", "target.spm"):
        if not os.path.isfile(full):
            missing.append(name)
    else:
        if not os.path.isfile(full):
            missing.append(name)

# If vocab.json is missing but both source.spm & target.spm exist, that's okay.
if ("vocab.json" in missing) and ("source.spm" not in missing) and ("target.spm" not in missing):
    missing = [x for x in missing if x != "vocab.json"]

if missing:
    print("ERROR: The following tokenizer/config files are missing from MODEL_DIR:", file=sys.stderr)
    for m in missing:
        print(f"  • {m}", file=sys.stderr)
    print("\nMake sure you copied all of these into:", MODEL_DIR, file=sys.stderr)
    sys.exit(1)

print(f"Loading tokenizer from {MODEL_DIR}", flush=True)
tokenizer = MarianTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

print("Loading pruned ONNX model…", flush=True)
model = MarianOnnx(MODEL_DIR, device="cpu")

# Sample French sentence
src = ["Le chat noir est sur le tapis ."]
print(f"Tokenizing sample: {src}", flush=True)
inputs = tokenizer(src, return_tensors="pt", padding=True)

print("Generating…", flush=True)
try:
    out_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=50
    )
    decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    print("Decoded →", decoded, flush=True)
except Exception as e:
    print("Error during generation:", e, flush=True)
    sys.exit(1)