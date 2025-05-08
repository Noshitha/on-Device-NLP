import os
import sys
import argparse
from transformers import MarianTokenizer
from datasets import load_dataset
from sacrebleu import corpus_bleu
from tqdm import tqdm

sys.path.append("/home/njuttu_umass_edu/on-DeviceNLP-2/Embedding_Weights_Quantization")
from core.marian import MarianOnnx

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True, help="Path to quantized ONNX model dir")
parser.add_argument("--output-dir", type=str, required=True, help="Path to store BLEU evaluation results")
parser.add_argument("--split", type=str, default="validation[:50]", help="Dataset split to evaluate on")
args = parser.parse_args()

MODEL_PATH = args.model_path
OUTPUT_DIR = args.output_dir
SPLIT = args.split

SRC_LANG = "en"
TGT_LANG = "de"
MAX_LEN = 128

def evaluate_bleu(model, tokenizer, dataset):
    references = []
    hypotheses = []
    source_texts = []

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "translations.txt"), "w", encoding="utf-8") as f_trans:
        for ex in tqdm(dataset, desc="Evaluating"):
            src_text = ex["translation"][SRC_LANG]
            tgt_text = ex["translation"][TGT_LANG]

            source_texts.append(src_text)
            references.append(tgt_text)

            inputs = tokenizer(src_text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            hypotheses.append(prediction)

            f_trans.write(f"Source: {src_text}\n")
            f_trans.write(f"Reference: {tgt_text}\n")
            f_trans.write(f"Prediction: {prediction}\n")
            f_trans.write("-" * 80 + "\n")

    with open(os.path.join(OUTPUT_DIR, "sources.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(source_texts))
    with open(os.path.join(OUTPUT_DIR, "references.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(references))
    with open(os.path.join(OUTPUT_DIR, "hypotheses.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(hypotheses))

    bleu = corpus_bleu(hypotheses, [references])
    with open(os.path.join(OUTPUT_DIR, "bleu_score.txt"), "w", encoding="utf-8") as f:
        f.write(f"BLEU Score ({SPLIT}): {bleu.score:.2f}\n")
        f.write(f"Detailed BLEU: {bleu}\n")

    return bleu.score

def main():
    print(f"Loading ONNX model from {MODEL_PATH}")
    model = MarianOnnx(MODEL_PATH)
    tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)

    print("Loading WMT dataset...")
    dataset = load_dataset("wmt14", f"{TGT_LANG}-{SRC_LANG}", split=SPLIT)

    bleu_score = evaluate_bleu(model, tokenizer, dataset)
    print(f"\nThe BLEU Score (ONNX, {SPLIT}): {bleu_score:.2f}")

if __name__ == "__main__":
    main()