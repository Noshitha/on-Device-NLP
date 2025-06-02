#!/usr/bin/env python
import os
import torch
from transformers import MarianTokenizer
from core.marian import MarianOnnx

# After pruning, all files (config.json, vocab, ONNX, etc.) should live here:
MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "..", "outs", "pruning", "Helsinki-NLP_opus-mt-fr-en")
)

print(f"Loading tokenizer from {MODEL_DIR}")
tokenizer = MarianTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

print("Loading pruned ONNX model…")
model = MarianOnnx(MODEL_DIR, device="cpu")

# Sample French sentence
src = ["Le chat noir est sur le tapis ."]
inputs = tokenizer(src, return_tensors="pt", padding=True)

print("Generating…")
out_ids = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=50
)

print("Decoded →", tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0])