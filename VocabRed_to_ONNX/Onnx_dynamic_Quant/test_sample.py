#!/usr/bin/env python
import os
import sys

import torch
import numpy as np
from transformers import MarianTokenizer

from core.marian import MarianOnnx

# ──────────────────────────────────────────────────────────────────────────────
# 1) Path to your pruned ONNX + tokenizer folder
MODEL_DIR = "/home/njuttu_umass_edu/on-DeviceNLP/outs/Vocab_Red_NoQuant/checkpoint-4924000"
DEVICE    = "cpu"   # or "cuda"
# ──────────────────────────────────────────────────────────────────────────────

# 2) Sanity-check that required files exist
for fname in ("config.json", "tokenizer_config.json", "encoder.onnx", "decoder.onnx", "lm_weight.bin"):
    if not os.path.isfile(os.path.join(MODEL_DIR, fname)):
        sys.exit(f"ERROR: {fname} not found in {MODEL_DIR}")
if not (os.path.isfile(os.path.join(MODEL_DIR, "vocab.json")) or
        (os.path.isfile(os.path.join(MODEL_DIR, "source.spm")) and
         os.path.isfile(os.path.join(MODEL_DIR, "target.spm")))):
    sys.exit("ERROR: Need either vocab.json or (source.spm & target.spm) in MODEL_DIR")

# 3) Load tokenizer
print(f"Loading tokenizer from {MODEL_DIR}…")
tokenizer = MarianTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

# 4) Subclass MarianOnnx to inject both encoder & decoder embeddings
class MarianOnnxWithEmb(MarianOnnx):
    def __init__(self, path: str, device: str = "cpu"):
        super().__init__(path, device)
        # load the shared embedding weight (same for encoder & decoder input)
        w = torch.load(os.path.join(path, "lm_weight.bin"), map_location="cpu")
        self._shared_emb = w.detach().cpu().numpy()

    def _encoder_forward(self, input_ids, attention_mask):
        ort_inputs = {
            "input_ids":                    input_ids.cpu().numpy(),
            "attention_mask":               attention_mask.cpu().numpy(),
            "encoder.embed_tokens.weight":  self._shared_emb,
        }
        last_hidden = self.encoder_session.run(None, ort_inputs)[0]
        return torch.from_numpy(last_hidden).to(input_ids.device)

    def _decoder_forward(self, input_ids, encoder_output, attention_mask):
        ort_inputs = {
            "input_ids":                     input_ids.cpu().numpy(),
            "encoder_hidden_states":        encoder_output.cpu().numpy(),
            "attention_mask":                attention_mask.cpu().numpy(),
            "decoder.embed_tokens.weight":   self._shared_emb,
        }
        decoder_out = self.decoder_session.run(None, ort_inputs)[0]
        # final LM head
        lm_logits = torch.from_numpy(decoder_out).to(input_ids.device)
        lm_logits = torch.nn.functional.linear(
            lm_logits,
            self.final_logits_weight,
            bias=self.final_logits_bias
        )
        return lm_logits

# 5) Instantiate
print("Loading ONNX model…")
model = MarianOnnxWithEmb(MODEL_DIR, device=DEVICE)

# 6) (Optional) override max_length
model.config.max_length = 50
print(f"Using max_length = {model.config.max_length}")

# 7) Prepare a test sentence
src = ["Le chat noir est sur le tapis ."]
print(f"Tokenizing : {src}")
inputs = tokenizer(src, return_tensors="pt", padding=True)
if DEVICE == "cuda":
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

# 8) Generate
print("Generating…")
output_ids = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
)
decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print("Decoded :", decoded)
