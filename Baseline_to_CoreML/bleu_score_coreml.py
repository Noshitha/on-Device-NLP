# bleu_score_coreml.py
import argparse
from transformers import MarianTokenizer
from core.marian_coreml import MarianCoreML
import sacrebleu

def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def evaluate_bleu(model_dir: str, src_path: str, ref_path: str):
    """
    Read source and reference files line-by-line, generate translations
    via CoreML, then compute corpus BLEU.
    """
    print(f"\nLoading model and tokenizer from '{model_dir}'")
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model     = MarianCoreML(model_dir)

    print(f"Reading {src_path} and {ref_path} …")
    sources   = load_lines(src_path)
    references= load_lines(ref_path)

    assert len(sources) == len(references), "Source and reference must have same line count."

    predictions = []
    for src in sources:
        inputs = tokenizer(src, return_tensors="pt")
        out_ids = model.generate(inputs["input_ids"], inputs["attention_mask"], max_length=100)
        pred   = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        predictions.append(pred)

    # sacrebleu expects list of reference‐lists
    score = sacrebleu.corpus_bleu(predictions, [references])
    print("\n" + "="*40)
    print(f"Corpus BLEU = {score.score:.2f}")
    print("="*40)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate BLEU of CoreML‐generated translations (macOS only)."
    )
    p.add_argument("model_dir", help="Path to your CoreML output folder")
    p.add_argument("src",       help="Plain‐text file of source sentences, one per line")
    p.add_argument("ref",       help="Plain‐text file of reference translations, one per line")
    args = p.parse_args()
    evaluate_bleu(args.model_dir, args.src, args.ref)
