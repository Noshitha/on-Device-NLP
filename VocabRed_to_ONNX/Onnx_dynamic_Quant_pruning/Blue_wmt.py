import os
import sys
import argparse
from transformers import MarianTokenizer
from datasets import load_dataset
from sacrebleu import corpus_bleu
from tqdm import tqdm

# 1) Point at the folder containing `core/marian.py`:
import sys
sys.path.append("/home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/Onnx_dynamic_Quant_pruning") 
from core.marian import MarianOnnx

parser = argparse.ArgumentParser()
parser.add_argument("--model-path",  type=str, required=True)
parser.add_argument("--output-dir",  type=str, required=True)
parser.add_argument("--split",       type=str, default="validation[:50]")
args = parser.parse_args()

MODEL_PATH = args.model_path
OUTPUT_DIR = args.output_dir
SPLIT      = args.split

SRC_LANG = "fr"
TGT_LANG = "en"
MAX_LEN  = 128

def evaluate_bleu(model, tokenizer, dataset):
    sources, references, hypotheses = [], [], []
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "translations.txt"), "w", encoding="utf-8") as fout:
        for ex in tqdm(dataset, desc="Evaluating"):
            src = ex["translation"][SRC_LANG]
            tgt = ex["translation"][TGT_LANG]

            sources.append(src)
            references.append(tgt)

            inputs = tokenizer(
                src,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN
            )
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=MAX_LEN
            )
            pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            hypotheses.append(pred)

            fout.write(f"Source:    {src}\n")
            fout.write(f"Reference: {tgt}\n")
            fout.write(f"Prediction:{pred}\n")
            fout.write("-" * 80 + "\n")

    with open(os.path.join(OUTPUT_DIR, "sources.txt"),    "w", encoding="utf-8") as f:
        f.write("\n".join(sources))
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
    model.device  # ensure it loaded
    print("Loading tokenizer…")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    print("Loading WMT14 dataset…")
    dataset = load_dataset("wmt14", f"{SRC_LANG}-{TGT_LANG}", split=SPLIT)

    bleu_score = evaluate_bleu(model, tokenizer, dataset)
    print(f"\nThe BLEU Score (ONNX, {SPLIT}): {bleu_score:.2f}")

if __name__ == "__main__":
    main()
