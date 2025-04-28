import os
import torch
import onnxruntime
from transformers import MarianTokenizer
import sacrebleu
from datasets import load_dataset

#  Configuration 
ONNX_MODEL_DIR = "./outs/Helsinki-NLP_opus-mt-en-de"  # path to your ONNX model (encoder.onnx, decoder.onnx, lm_weight.bin etc.)
SAMPLE_SIZE = 500  # How many examples to evaluate

#  Load Tokenizer 
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

#  Load ONNX Model 
from core.marian import MarianOnnx
model = MarianOnnx(ONNX_MODEL_DIR)

#  Load Dataset 
# We'll use WMT14 newstest2014 subset for English-German
dataset = load_dataset("wmt14", "de-en", split="test")
dataset = dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))  # Take a sample

#  Prepare Sources and References 
sources = [ex['translation']['en'] for ex in dataset]
references = [ex['translation']['de'] for ex in dataset]

#  Generate Translations 
translations = []

batch_size = 8 
for i in range(0, len(sources), batch_size):
    batch_src = sources[i:i+batch_size]
    inputs = tokenizer(batch_src, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    preds = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
    translations.extend(decoded)

#  Evaluate BLEU 
bleu = sacrebleu.corpus_bleu(translations, [references])
print(f"BLEU score: {bleu.score:.2f}")

#  Evaluate chrF++ 
chrf = sacrebleu.corpus_chrf(translations, [references])
print(f"chrF++ score: {chrf.score:.2f}")