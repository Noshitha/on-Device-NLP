# core/benchmark_coreml.py
import torch
from transformers import MarianTokenizer
from core.marian_coreml import MarianCoreML

def verify_export_coreml(model_dir, model_name="Helsinki-NLP/opus-mt-fr-en"):
    print("\nVerifying CoreML export...\n")
    
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianCoreML(model_dir)

    text = "Hello World!"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    output_ids = model.generate(input_ids, attention_mask)
    translation = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    print(f"Input: {text}")
    print(f"Translation: {translation}")
