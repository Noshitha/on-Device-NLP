import torch
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
import evaluate
from comet import download_model, load_from_checkpoint
import time

def evaluate_baseline(
    model_name: str,
    src_lang: str,
    tgt_lang: str,
    dataset_name: str,
    dataset_config: str,
    max_length: int = 128,
    num_samples: int = 500,
):
    # ─── Load model & tokenizer ────────────────────────────────────────────────
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # ─── Prepare dataset subset ────────────────────────────────────────────────
    ds = load_dataset(dataset_name, dataset_config, split="test")
    dataset = ds.select(range(num_samples))

    # ─── Generation + timing ───────────────────────────────────────────────────
    model.eval()
    predictions, references, src_texts = [], [], []
    total_tokens = 0

    # warm up sync & timer
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        for ex in dataset:
            src = ex["translation"][src_lang]
            ref = ex["translation"][tgt_lang]

            # tokenize → device
            inputs = tokenizer(
                src,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # sync → generate → sync
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if isinstance(model, torch.nn.DataParallel):
                out_ids = model.module.generate(**inputs, max_length=max_length)
            else:
                out_ids = model.generate(**inputs, max_length=max_length)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # count tokens & decode
            total_tokens += out_ids.numel()
            pred = tokenizer.decode(out_ids[0], skip_special_tokens=True)

            predictions.append(pred)
            references.append([ref])
            src_texts.append(src)

    # final sync & stop timer
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    gen_time = t_end - t_start

    # ─── Load & compute metrics ────────────────────────────────────────────────
    print(f"\nEvaluated {len(predictions)} examples (max_length={max_length}, samples={num_samples})\n")

    bleu      = evaluate.load("sacrebleu")
    chrf      = evaluate.load("chrf")
    ter       = evaluate.load("ter")
    meteor    = evaluate.load("meteor")
    bleurt    = evaluate.load("bleurt", config_name="bleurt-20-D12")
    bertscore = evaluate.load("bertscore")

    # COMET
    comet_path  = download_model("Unbabel/wmt22-comet-da")
    comet_mod   = load_from_checkpoint(comet_path)
    comet_inputs= [
        {"src": s, "ref": r[0], "mt": p}
        for s, r, p in zip(src_texts, references, predictions)
    ]
    comet_score = comet_mod.predict(
        comet_inputs,
        batch_size=8,
        gpus=1 if torch.cuda.is_available() else 0
    )

    # metric computations
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

    # ─── Print all metric results ─────────────────────────────────────────────
    print(f"BLEU:  {bleu_res['score']:.2f}")
    print(f"CHRF:  {chrf_res['score']:.2f}")
    print(f"TER:   {ter_res['score']:.2f}")
    print(f"METEOR:{meteor_res['meteor']:.2f}")
    print(f"BLEURT (avg):   {sum(bleurt_res['scores'])/len(bleurt_res['scores']):.4f}")
    print(f"BERTScore F1 (avg): {sum(bert_res['f1'])/len(bert_res['f1']):.4f}")
    print(f"COMET (avg):     {sum(comet_score.scores)/len(comet_score.scores):.4f}")

    # ─── Decoding speed summary ────────────────────────────────────────────────
    n = len(predictions)
    print("\n--- Decoding Speed Summary ---")
    print(f"Total gen time:      {gen_time:.2f} s")
    print(f"Sentences/sec:       {n / gen_time:.2f}")
    print(f"Tokens/sec:          {total_tokens / gen_time:.2f}")
    print(f"Avg latency/sent:    {gen_time / n:.4f} s\n")


if __name__ == "__main__":
    # Example: German baseline
    evaluate_baseline(
        model_name="Helsinki-NLP/opus-mt-en-de",
        src_lang="en",
        tgt_lang="de",
        dataset_name="wmt14",
        dataset_config="de-en",
        max_length=128,
        num_samples=500
    )

    # Example: French baseline
    # evaluate_baseline(
    #     model_name="Helsinki-NLP/opus-mt-en-fr",
    #     src_lang="en",
    #     tgt_lang="fr",
    #     dataset_name="wmt14",
    #     dataset_config="fr-en",
    #     max_length=128,
    #     num_samples=10
    # )
