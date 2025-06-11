#!/usr/bin/env python
import argparse
import os
from tqdm import tqdm
import torch
from transformers import MarianTokenizer, MarianMTModel
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF, TER
import evaluate


def sample_test(model_dir: str, device: str = "cpu", max_length: int = 50):
    """
    Run a single-sentence sanity-check translation.
    """
    print("Loading vocab-reduced model for sample test...")
    tokenizer = MarianTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = MarianMTModel.from_pretrained(model_dir).to(device)

    src = ["Le chat noir est sur le tapis ."]
    print(f"Tokenizing: {src}")
    inputs = tokenizer(src, return_tensors="pt", padding=True).to(device)
    print("Generating...")
    out_ids = model.generate(**inputs, max_length=max_length)
    decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    print(f"Decoded : {decoded}\n")


def load_data(src_lang: str, tgt_lang: str, split: str):
    ds = load_dataset("wmt14", f"{src_lang}-{tgt_lang}", split=split)
    return [(ex["translation"][src_lang], ex["translation"][tgt_lang]) for ex in ds]


def generate_all(src_texts, tokenizer, model, device: str, max_length: int):
    preds = []
    for txt in tqdm(src_texts, desc="Translating"):
        batch = tokenizer(txt, return_tensors="pt", padding=True).to(device)
        out_ids = model.generate(**batch, max_length=max_length)
        preds.append(tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0])
    return preds


def evaluate_metrics(model_dir: str,
                     device: str,
                     src: str,
                     tgt: str,
                     split: str,
                     max_length: int):
    """
    Compute BLEU, ChrF, TER, METEOR over a dataset split.
    """
    print("\nLoading model for metrics evaluation...")
    tokenizer = MarianTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = MarianMTModel.from_pretrained(model_dir).to(device)

    data = load_data(src, tgt, split)
    src_texts, refs = zip(*data)
    refs_list = [[r] for r in refs]

    preds = generate_all(src_texts, tokenizer, model, device, max_length)

    bleu = BLEU().corpus_score(preds, refs_list)
    chrf = CHRF().corpus_score(preds, refs_list)
    ter  = TER().corpus_score(preds, refs_list)
    meteor = evaluate.load("meteor").compute(predictions=preds, references=refs_list)["meteor"]

    print(f"=== Metrics over {len(preds)} sentences ===")
    print(f"BLEU    = {bleu.score:.2f}")
    print(f"ChrF    = {chrf.score:.2f}")
    print(f"TER     = {ter.score:.2f}")
    print(f"METEOR  = {meteor:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample & metrics evaluation for vocab-reduced MarianMT model"
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to the vocab-reduced model directory"
    )
    parser.add_argument(
        "--device", choices=["cpu","cuda"], default="cpu",
        help="Run on CPU or CUDA"
    )
    parser.add_argument(
        "--max-length", type=int, default=50,
        help="Maximum length for generation"
    )
    parser.add_argument(
        "--src", default="fr",
        help="Source language code"
    )
    parser.add_argument(
        "--tgt", default="en",
        help="Target language code"
    )
    parser.add_argument(
        "--split", default="validation[:100]",
        help="HF dataset split for evaluation"
    )
    args = parser.parse_args()

    # 1) Sample test
    print("=== Sample test on vocab-reduced model ===")
    sample_test(args.model_dir, args.device, args.max_length)

    # 2) All-metrics evaluation
    print("=== All metrics Evaluation ===")
    evaluate_metrics(
        args.model_dir,
        args.device,
        args.src,
        args.tgt,
        args.split,
        args.max_length
    )
