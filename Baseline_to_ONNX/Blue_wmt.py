import os
from transformers import MarianTokenizer
from datasets import load_dataset
from sacrebleu import corpus_bleu
from tqdm import tqdm
import sys
sys.path.append("/home/njuttu_umass_edu/on-DeviceNLP-2/Baseline_to_ONNX")
from core.marian import MarianOnnx

# Configuration
MODEL_PATH = "/home/njuttu_umass_edu/on-DeviceNLP-2/outs/Helsinki-NLP_opus-mt-en-de"
SRC_LANG = "en"
TGT_LANG = "de"
MAX_LEN = 128
SPLIT = "validation[:50]"
OUTPUT_DIR = "/home/njuttu_umass_edu/on-DeviceNLP-2/outs/bleu_evaluation_results"  # Directory to save results

def evaluate_bleu(model, tokenizer, dataset):
    references = []
    hypotheses = []
    source_texts = []

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Open files for writing translations
    with open(os.path.join(OUTPUT_DIR, "translations.txt"), "w", encoding="utf-8") as f_trans:
        for ex in tqdm(dataset, desc="Evaluating"):
            src_text = ex["translation"][SRC_LANG]
            tgt_text = ex["translation"][TGT_LANG]
            
            source_texts.append(src_text)
            references.append(tgt_text)

            # Generate hypothesis
            inputs = tokenizer(src_text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            hypotheses.append(prediction)
            
            # Write to translation file
            f_trans.write(f"Source: {src_text}\n")
            f_trans.write(f"Reference: {tgt_text}\n")
            f_trans.write(f"Prediction: {prediction}\n")
            f_trans.write("-" * 80 + "\n")

    # Save all translations to separate files for easier comparison
    with open(os.path.join(OUTPUT_DIR, "sources.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(source_texts))
        
    with open(os.path.join(OUTPUT_DIR, "references.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(references))
        
    with open(os.path.join(OUTPUT_DIR, "hypotheses.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(hypotheses))

    bleu = corpus_bleu(hypotheses, [references])
    
    # Save BLEU score
    with open(os.path.join(OUTPUT_DIR, "bleu_score.txt"), "w", encoding="utf-8") as f:
        f.write(f"BLEU Score ({SPLIT}): {bleu.score:.2f}\n")
        f.write(f"Detailed BLEU: {bleu}\n")
    
    return bleu.score

def main():
    print(f"Loading ONNX model from {MODEL_PATH}")
    model = MarianOnnx(MODEL_PATH)
    tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)

    print(f"Loading WMT dataset...")
    dataset = load_dataset("wmt14", f"{TGT_LANG}-{SRC_LANG}", split=SPLIT)

    bleu_score = evaluate_bleu(model, tokenizer, dataset)
    print(f"\nThe BLEU Score (ONNX, {SPLIT}): {bleu_score:.2f}")
    print(f"Results saved to {OUTPUT_DIR} directory")

if __name__ == "__main__":
    main()