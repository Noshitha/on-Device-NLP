import os
import sys
import argparse
from transformers import MarianTokenizer
from datasets import load_dataset
from sacrebleu import corpus_bleu
from tqdm import tqdm

# Add the path to core
sys.path.append("/home/njuttu_umass_edu/on-DeviceNLP-2/Baseline_to_ONNX/Onnx_dynamic_Quant")
from core.marian import MarianOnnx

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BLEU score for ONNX translation model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to exported ONNX model directory")
    parser.add_argument("--output-dir", type=str, help="Directory to save BLEU results (default: model_path/bleu_evaluation_results)")
    parser.add_argument("--split", type=str, default="validation[:50]", help="Dataset split (default: validation[:50])")
    parser.add_argument("--src-lang", type=str, default="en", help="Source language code")
    parser.add_argument("--tgt-lang", type=str, default="de", help="Target language code")
    parser.add_argument("--max-len", type=int, default=128, help="Max length for tokenization")
    return parser.parse_args()

def evaluate_bleu(model, tokenizer, dataset, output_dir, src_lang, tgt_lang, max_len):
    references = []
    hypotheses = []
    source_texts = []

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "translations.txt"), "w", encoding="utf-8") as f_trans:
        for ex in tqdm(dataset, desc="Evaluating"):
            src_text = ex["translation"][src_lang]
            tgt_text = ex["translation"][tgt_lang]

            source_texts.append(src_text)
            references.append(tgt_text)

            inputs = tokenizer(src_text, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            hypotheses.append(prediction)

            f_trans.write(f"Source: {src_text}\n")
            f_trans.write(f"Reference: {tgt_text}\n")
            f_trans.write(f"Prediction: {prediction}\n")
            f_trans.write("-" * 80 + "\n")

    with open(os.path.join(output_dir, "sources.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(source_texts))
    with open(os.path.join(output_dir, "references.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(references))
    with open(os.path.join(output_dir, "hypotheses.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(hypotheses))

    bleu = corpus_bleu(hypotheses, [references])

    with open(os.path.join(output_dir, "bleu_score.txt"), "w", encoding="utf-8") as f:
        f.write(f"BLEU Score ({args.split}): {bleu.score:.2f}\n")
        f.write(f"Detailed BLEU: {bleu}\n")

    return bleu.score

def main():
    global args
    args = parse_args()
    model_path = args.model_path
    output_dir = args.output_dir or os.path.join(model_path, "bleu_evaluation_results")

    print(f"Loading ONNX model from {model_path}")
    model = MarianOnnx(model_path)
    tokenizer = MarianTokenizer.from_pretrained(model_path)

    print(f"Loading WMT dataset...")
    dataset = load_dataset("wmt14", f"{args.tgt_lang}-{args.src_lang}", split=args.split)

    bleu_score = evaluate_bleu(model, tokenizer, dataset, output_dir, args.src_lang, args.tgt_lang, args.max_len)

    print(f"\nThe BLEU Score (ONNX, {args.split}): {bleu_score:.2f}")
    print(f"Results saved to {output_dir} directory")

if __name__ == "__main__":
    main()