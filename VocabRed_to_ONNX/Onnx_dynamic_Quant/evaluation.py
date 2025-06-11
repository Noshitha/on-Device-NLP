#!/usr/bin/env python
import argparse
import os
from tqdm import tqdm
import torch
# model & tokenization
from transformers import MarianTokenizer
from core.marian import MarianOnnx
# metrics
from sacrebleu.metrics import BLEU, CHRF, TER
import evaluate

def load_data(src_lang="fr", tgt_lang="en", split="validation[:100]"):
    from datasets import load_dataset
    ds = load_dataset("wmt14", f"{src_lang}-{tgt_lang}", split=split)
    return [(ex["translation"][src_lang], ex["translation"][tgt_lang]) for ex in ds]

def generate_all(src_texts, tokenizer, onnx_model, device="cpu"):
    preds = []
    for txt in tqdm(src_texts, desc="Translating"):
        batch = tokenizer(txt, return_tensors="pt", padding=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        out_ids = onnx_model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        preds.append(tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0])
    return preds

def main(args):
    # 1) load tokenizer & ONNX model (no embedding injection)
    tokenizer = MarianTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    onnx_model = MarianOnnx(args.onnx_dir, device=args.device)

    # 2) load data
    data = load_data(args.src, args.tgt, args.split)
    src_texts, refs = zip(*data)
    refs_list = [[r] for r in refs]

    # 3) generate predictions
    preds = generate_all(src_texts, tokenizer, onnx_model, device=args.device)

    # 4) compute sacreBLEU metrics
    bleu = BLEU().corpus_score(preds, refs_list)
    chrf = CHRF().corpus_score(preds, refs_list)
    ter  = TER().corpus_score(preds, refs_list)

    # 5) compute METEOR via HuggingFace evaluate
    meteor = evaluate.load("meteor")
    meteor_score = meteor.compute(predictions=preds, references=refs_list)["meteor"]

    # 6) print all metrics
    print(f"\n=== Metrics over {len(preds)} sentences ===")
    print(f"BLEU    = {bleu.score:.2f}")
    print(f"ChrF    = {chrf.score:.2f}")
    print(f"TER     = {ter.score:.2f}")
    print(f"METEOR  = {meteor_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate un-pruned ONNX translation model"
    )
    parser.add_argument("--model-dir", required=True,
                        help="Directory with tokenizer/config.json")
    parser.add_argument("--onnx-dir", required=True,
                        help="Directory with encoder.onnx & decoder.onnx")
    parser.add_argument("--src",    default="fr", help="Source language code")
    parser.add_argument("--tgt",    default="en", help="Target language code")
    parser.add_argument("--split",  default="validation[:100]",
                        help="HF dataset split (e.g. \"validation[:100]\")")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="Run on CPU or CUDA")
    args = parser.parse_args()
    main(args)
