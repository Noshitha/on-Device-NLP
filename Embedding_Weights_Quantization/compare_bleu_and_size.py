import os
import torch
import onnx
import numpy as np
import matplotlib.pyplot as plt

from transformers import MarianTokenizer
from core.marian import MarianOnnx

# Paths
BASE_DIR = "/home/njuttu_umass_edu/on-DeviceNLP-2"
models = {
    "Liberal Quant": os.path.join(BASE_DIR, "outs/Helsinki-NLP_opus-mt-en-de"),
    "Embedding Quant": os.path.join(BASE_DIR, "quantized_embedding_out/Helsinki-NLP_opus-mt-en-de"),
    "Embedding+ONNX Quant": os.path.join(BASE_DIR, "embedding_Quant_out/Helsinki-NLP_opus-mt-en-de"),
}

# Sample input
sample_sentences = ["Hello world!", "How are you?", "I love working on NLP models."]

def compute_bleu(model_dir):
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianOnnx(model_dir)

    refs = []
    hyps = []

    for sentence in sample_sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        output_ids = model.generate(inputs["input_ids"], inputs["attention_mask"])
        output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        refs.append([sentence])  # Simplification: Reference = input
        hyps.append(output_texts[0])

    # Quick BLEU (simple version)
    matches = sum(ref[0] == hyp for ref, hyp in zip(refs, hyps))
    return matches / len(hyps) * 100  # fake BLEU % to see degradation easily

def get_sizes(model_dir):
    encoder_size = os.path.getsize(os.path.join(model_dir, "encoder.onnx")) / 1e6  # MB
    decoder_size = os.path.getsize(os.path.join(model_dir, "decoder.onnx")) / 1e6
    lm_weight_size = os.path.getsize(os.path.join(model_dir, "lm_weight.bin")) / 1e6
    total_size = sum(os.path.getsize(os.path.join(model_dir, f)) for f in os.listdir(model_dir)) / 1e6
    return encoder_size, decoder_size, lm_weight_size, total_size

# Collect data
results = []

for label, path in models.items():
    bleu = compute_bleu(path)
    encoder_size, decoder_size, lm_weight_size, total_size = get_sizes(path)
    results.append((label, bleu, encoder_size, decoder_size, lm_weight_size, total_size))

# Tabulate
import pandas as pd
df = pd.DataFrame(results, columns=["Model", "BLEU", "Encoder (MB)", "Decoder (MB)", "lm_weight (MB)", "Total (MB)"])
print(df)

# Plot
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("ONNX Compression vs BLEU Score Analysis", fontsize=16)

axs[0, 0].bar(df["Model"], df["BLEU"])
axs[0, 0].set_ylabel("BLEU Score")

axs[0, 1].bar(df["Model"], df["Total (MB)"])
axs[0, 1].set_ylabel("Total Folder Size (MB)")

axs[1, 0].bar(df["Model"], df["Encoder (MB)"], label="Encoder")
axs[1, 0].bar(df["Model"], df["Decoder (MB)"], label="Decoder", bottom=df["Encoder (MB)"])
axs[1, 0].legend()
axs[1, 0].set_ylabel("Encoder+Decoder Sizes (MB)")

axs[1, 1].bar(df["Model"], df["lm_weight (MB)"])
axs[1, 1].set_ylabel("lm_weight.bin Size (MB)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
