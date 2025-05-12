import torch
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
import evaluate
from comet import download_model, load_from_checkpoint
import time

def evaluate_translation(
    model_dir,
    src_lang,
    tgt_lang,
    dataset_name,
    dataset_config,
    batch_size=1,
    max_length=128,
    num_samples=500
):
    # ─── Load model & tokenizer ─────────────────────────────────────────────────
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # ─── Prepare data ────────────────────────────────────────────────────────────
    ds = load_dataset(dataset_name, dataset_config, split="test")
    dataset = ds.select(range(num_samples))  # take first num_samples examples

    # ─── Generation / timing ─────────────────────────────────────────────────────
    model.eval()
    predictions, references, src_texts = [], [], []
    total_tokens_generated = 0

    # GPU sync before we start timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_start = time.perf_counter()

    with torch.no_grad():
        for ex in dataset:
            src = ex["translation"][src_lang]
            ref = ex["translation"][tgt_lang]

            # Tokenize + move to device
            inputs = tokenizer(
                src,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Sync & generate
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if isinstance(model, torch.nn.DataParallel):
                out_ids = model.module.generate(**inputs, max_length=max_length)
            else:
                out_ids = model.generate(**inputs, max_length=max_length)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Count tokens
            total_tokens_generated += out_ids.numel()

            # Decode & collect
            pred = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            predictions.append(pred)
            references.append([ref])
            src_texts.append(src)

    # Final sync + stop timer
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_end = time.perf_counter()
    gen_time = gen_end - gen_start

    # ─── Load & compute metrics ──────────────────────────────────────────────────
    print(f"\nProcessed {len(predictions)} sentences in generation step.\n")

    bleu     = evaluate.load("sacrebleu")
    chrf     = evaluate.load("chrf")
    ter      = evaluate.load("ter")
    meteor   = evaluate.load("meteor")
    bleurt   = evaluate.load("bleurt", config_name="bleurt-20-D12")
    bertscore= evaluate.load("bertscore")

    # COMET
    model_path   = download_model("Unbabel/wmt22-comet-da")
    comet_model  = load_from_checkpoint(model_path)
    comet_inputs = [
        {"src": s, "ref": r[0], "mt": p}
        for s, r, p in zip(src_texts, references, predictions)
    ]
    comet_score = comet_model.predict(
        comet_inputs,
        batch_size=8,
        gpus=1 if torch.cuda.is_available() else 0
    )

    # Print metric results
    bleu_res   = bleu.compute(predictions=predictions, references=references)
    chrf_res   = chrf.compute(predictions=predictions, references=references)
    ter_res    = ter.compute(predictions=predictions, references=references)
    meteor_res = meteor.compute(predictions=predictions, references=[r[0] for r in references])
    bleurt_res = bleurt.compute(predictions=predictions, references=[r[0] for r in references])
    bert_res   = bertscore.compute(
                    predictions=predictions,
                    references=[r[0] for r in references],
                    lang=tgt_lang
                 )

    print(f"BLEU:  {bleu_res['score']:.2f}")
    print(f"CHRF:  {chrf_res['score']:.2f}")
    print(f"TER:   {ter_res['score']:.2f}")
    print(f"METEOR:{meteor_res['meteor']:.2f}")
    print(f"BLEURT (avg): {sum(bleurt_res['scores'])/len(bleurt_res['scores']):.4f}")
    print(f"BERTScore F1 (avg): {sum(bert_res['f1'])/len(bert_res['f1']):.4f}")
    print(f"COMET (avg):     {sum(comet_score.scores)/len(comet_score.scores):.4f}")

    # ─── Decoding speed summary ──────────────────────────────────────────────────
    sent_count = len(predictions)
    print("\n--- Decoding Speed Summary ---")
    print(f"Total generation time: {gen_time:.2f} s")
    print(f"Sentences/sec:         {sent_count / gen_time:.2f}")
    print(f"Tokens/sec:            {total_tokens_generated / gen_time:.2f}")
    print(f"Avg latency (sent):    {gen_time / sent_count:.4f} s\n")


if __name__ == "__main__":
    evaluate_translation(
        model_dir="/home/agerasashank_umass_edu/arcReduce/checkpoint-234830",
        src_lang="en",
        tgt_lang="de",
        dataset_name="wmt14",
        dataset_config="de-en",
        batch_size=1,
        max_length=128,
        num_samples=10
    )
