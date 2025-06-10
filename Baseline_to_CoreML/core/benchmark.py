# core/benchmark.py
import torch
import time
import numpy as np
from transformers import MarianTokenizer
from core.marian_coreml import MarianCoreML
from sacrebleu import corpus_bleu

def verify_export_coreml(model_dir, model_name="Helsinki-NLP/opus-mt-fr-en"):
    """Verify CoreML export with basic translation test"""
    print("\nVerifying CoreML export...\n")
         
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianCoreML(model_dir)
     
    test_texts = [
        "Hello World!",
        "How are you today?",
        "This is a test sentence.",
        "Machine translation is fascinating."
    ]
    
    print("Testing translations:")
    print("=" * 50)
    
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        start_time = time.time()
        output_ids = model.generate(input_ids, attention_mask, max_length=50)
        generation_time = time.time() - start_time
        
        try:
            translation = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        except:
            translation = f"[Generated {len(output_ids[0])} tokens - decode failed]"
        
        print(f"Input: {text}")
        print(f"Translation: {translation}")
        print(f"Time: {generation_time:.3f}s")
        print("-" * 30)

def benchmark_coreml_model(model_dir, test_sentences=None, max_length=50):
    """Comprehensive benchmark of CoreML model"""
    print("\nBenchmarking CoreML Model")
    print("=" * 40)
    
    # Default test sentences if none provided
    if test_sentences is None:
        test_sentences = [
            "Hello, how are you?",
            "The weather is nice today.",
            "I love reading books in my free time.",
            "Technology is advancing rapidly.",
            "This is a longer sentence to test the model's ability to handle more complex inputs."
        ]
    
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianCoreML(model_dir)
    
    results = {
        'translations': [],
        'generation_times': [],
        'input_lengths': [],
        'output_lengths': []
    }
    
    print(f"Testing {len(test_sentences)} sentences...")
    print("-" * 50)
    
    for i, sentence in enumerate(test_sentences):
        print(f"[{i+1}/{len(test_sentences)}] Processing: {sentence}")
        
        # Tokenize input
        inputs = tokenizer(sentence, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        input_length = input_ids.shape[1]
        
        # Generate translation
        start_time = time.time()
        try:
            output_ids = model.generate(input_ids, attention_mask, max_length=max_length)
            generation_time = time.time() - start_time
            
            # Decode output
            translation = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            output_length = len(output_ids[0])
            
            results['translations'].append(translation)
            results['generation_times'].append(generation_time)
            results['input_lengths'].append(input_length)
            results['output_lengths'].append(output_length)
            
            print(f"  → {translation}")
            print(f"  → Time: {generation_time:.3f}s | Tokens: {input_length}→{output_length}")
            
        except Exception as e:
            print(f"  → Error: {e}")
            results['translations'].append("[ERROR]")
            results['generation_times'].append(0)
            results['input_lengths'].append(input_length)
            results['output_lengths'].append(0)
        
        print()
    
    # Print summary statistics
    print("=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    
    valid_times = [t for t in results['generation_times'] if t > 0]
    if valid_times:
        avg_time = np.mean(valid_times)
        min_time = np.min(valid_times)
        max_time = np.max(valid_times)
        
        print(f"Generation Time Stats:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Min: {min_time:.3f}s")
        print(f"  Max: {max_time:.3f}s")
        
        valid_input_lengths = [results['input_lengths'][i] for i, t in enumerate(results['generation_times']) if t > 0]
        valid_output_lengths = [results['output_lengths'][i] for i, t in enumerate(results['generation_times']) if t > 0]
        
        if valid_input_lengths:
            avg_input_len = np.mean(valid_input_lengths)
            avg_output_len = np.mean(valid_output_lengths)
            tokens_per_sec = avg_output_len / avg_time if avg_time > 0 else 0
            
            print(f"\nToken Stats:")
            print(f"  Avg input length: {avg_input_len:.1f} tokens")
            print(f"  Avg output length: {avg_output_len:.1f} tokens")
            print(f"  Generation speed: {tokens_per_sec:.1f} tokens/sec")
    
    successful_translations = len([t for t in results['translations'] if t != "[ERROR]"])
    success_rate = (successful_translations / len(test_sentences)) * 100
    print(f"\nSuccess Rate: {successful_translations}/{len(test_sentences)} ({success_rate:.1f}%)")
    
    return results

def evaluate_bleu_score(model_dir, test_pairs=None):
    """Evaluate BLEU score on test data"""
    print("\nEvaluating BLEU Score")
    print("=" * 30)
    
    # Default test pairs if none provided (source, reference)
    if test_pairs is None:
        test_pairs = [
            ("Hello, how are you?", "Bonjour, comment allez-vous?"),
            ("The weather is nice today.", "Le temps est beau aujourd'hui."),
            ("I like to read books.", "J'aime lire des livres."),
            ("This is a test sentence.", "Ceci est une phrase de test."),
            ("Machine translation is improving.", "La traduction automatique s'améliore.")
        ]
    
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianCoreML(model_dir)
    
    predictions = []
    references = []
    
    for i, (source, reference) in enumerate(test_pairs):
        print(f"[{i+1}/{len(test_pairs)}] Translating: {source}")
        
        try:
            inputs = tokenizer(source, return_tensors="pt")
            output_ids = model.generate(inputs["input_ids"], inputs["attention_mask"])
            translation = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            predictions.append(translation)
            references.append([reference])  # BLEU expects list of references
            
            print(f"  Prediction: {translation}")
            print(f"  Reference:  {reference}")
            
        except Exception as e:
            print(f"  Error: {e}")
            predictions.append("")
            references.append([reference])
        
        print()
    
    # Calculate BLEU score
    if predictions and any(pred.strip() for pred in predictions):
        try:
            bleu_score = corpus_bleu(predictions, references)
            print("=" * 40)
            print(f"BLEU Score: {bleu_score.score:.2f}")
            print("=" * 40)
            return bleu_score.score
        except Exception as e:
            print(f"Error calculating BLEU: {e}")
            return 0.0
    else:
        print("No valid predictions for BLEU calculation!")
        return 0.0