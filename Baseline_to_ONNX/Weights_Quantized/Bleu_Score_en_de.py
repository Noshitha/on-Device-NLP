import os
import sys
import torch
from transformers import MarianTokenizer
from datasets import load_dataset
from sacrebleu import corpus_bleu
from tqdm import tqdm

sys.path.append("/home/njuttu_umass_edu/on-DeviceNLP-2/Embedding_Weights_Quantization")
from core.marian import MarianOnnx

SRC_LANG = "en"
TGT_LANG = "de"
MAX_LEN = 128
NUM_SAMPLES = 50 # For less memory usage
SPLIT = "train"
BATCH_SIZE = 2   # To prevent memory issues

PYTORCH_MODEL_DIR = "/home/njuttu_umass_edu/on-DeviceNLP-2/local_model/Helsinki-NLP_opus-mt-en-de"
#ONNX_MODEL_DIR = "/home/njuttu_umass_edu/on-DeviceNLP-2/quantized_embedding_out/Helsinki-NLP_opus-mt-en-de"
ONNX_MODEL_DIR = "/home/njuttu_umass_edu/on-DeviceNLP-2/embedding_Quant_out/Helsinki-NLP_opus-mt-en-de"
def prepare_dataset(dataset):
    inputs = []
    targets = []
    for ex in dataset:
        inputs.append(ex["translation"]["en"])
        targets.append(ex["translation"]["de"])
    return inputs, targets

def evaluate_bleu_onnx(model, tokenizer, inputs, targets):
    hypotheses = []
    
    # Process in batches
    for i in tqdm(range(0, len(inputs), BATCH_SIZE), desc="Processing ONNX batches"):
        # Get current batch
        batch_inputs = inputs[i:i+BATCH_SIZE]
        
        # Tokenize
        batch = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        
        # Generate translations
        output_ids = model.generate(
            input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"]
        )
        
        # Decode predictions
        predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        hypotheses.extend(predictions)
    
    # Calculate BLEU score
    bleu = corpus_bleu(hypotheses, [targets])
    return bleu.score

def main():
    print(f"\nLoading tokenizer from local model folder...")
    tokenizer = MarianTokenizer.from_pretrained(PYTORCH_MODEL_DIR)

    print(f"\nLoading OPUS Books dataset (de-en)...")
    dataset = load_dataset("opus_books", "de-en", split=SPLIT)
    dataset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))
    inputs, targets = prepare_dataset(dataset)

    print(f"\nLoading ONNX model from {ONNX_MODEL_DIR}...")
    onnx_model = MarianOnnx(ONNX_MODEL_DIR)  

    print("\nEvaluating ONNX model...")
    onnx_bleu = evaluate_bleu_onnx(onnx_model, tokenizer, inputs, targets)
    print(f"\n[ONNX] BLEU Score ({SPLIT} set, {NUM_SAMPLES} examples): {onnx_bleu:.2f}")

    # Save BLEU score to a file
    output_file = os.path.join(ONNX_MODEL_DIR, "onnx_bleu_scores.txt")
    with open(output_file, "w") as f:
        f.write(f"BLEU Scores for {SRC_LANG}->{TGT_LANG} Translation ({SPLIT} set, {NUM_SAMPLES} samples)\n")
        f.write(f"-------------------------------------------\n")
        f.write(f"ONNX BLEU: {onnx_bleu:.2f}\n")

    print(f"\n Saved BLEU score to: {output_file}")

if __name__ == "__main__":
    main()