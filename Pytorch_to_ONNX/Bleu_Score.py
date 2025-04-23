import os
from transformers import MarianTokenizer
from datasets import load_dataset
from sacrebleu import corpus_bleu
from tqdm import tqdm
import sys
sys.path.append("/home/njuttu_umass_edu/on-DeviceNLP-2/Pytorch_to_ONNX")
from core.marian import MarianOnnx

# Configuration
MODEL_PATH = "./local_models/Helsinki-NLP_opus-mt-en-de"
SRC_LANG = "en"
TGT_LANG = "de"
MAX_LEN = 128
SPLIT = "validation[:50]"

def evaluate_bleu(model, tokenizer, dataset):
    references = []
    hypotheses = []

    for ex in tqdm(dataset, desc="Evaluating"):
        src_text = ex["translation"][SRC_LANG]
        tgt_text = ex["translation"][TGT_LANG]

        # Append reference
        references.append(tgt_text)

        # Generate hypothesis
        inputs = tokenizer(src_text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        hypotheses.append(prediction)

    bleu = corpus_bleu(hypotheses, [references])
    return bleu.score

def main():
    print(f"Loading ONNX model from {MODEL_PATH}")
    model = MarianOnnx(MODEL_PATH)
    tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)

    print(f"Loading WMT dataset...")
    dataset = load_dataset("wmt14", f"{TGT_LANG}-{SRC_LANG}", split=SPLIT)

    bleu_score = evaluate_bleu(model, tokenizer, dataset)
    print(f"\nThe BLEU Score (ONNX, {SPLIT}): {bleu_score:.2f}")

if __name__ == "__main__":
    main()
