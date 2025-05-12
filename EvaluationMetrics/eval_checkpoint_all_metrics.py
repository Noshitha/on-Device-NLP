import torch
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader
from comet import download_model, load_from_checkpoint

def evaluate_translation(model_dir, src_lang, tgt_lang, dataset_name, dataset_config, batch_size=1, max_length=128, num_samples=500):
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    dataset = load_dataset(dataset_name, dataset_config, split=f"test[:{num_samples}]")

    def identity_collate(batch):
        return batch

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=identity_collate)

    predictions = []
    references = []
    src_texts = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            ex = batch[0]
            src_text = ex["translation"][src_lang]
            tgt_text = ex["translation"][tgt_lang]

            tokens = tokenizer(src_text, truncation=False, add_special_tokens=False).input_ids
            if len(tokens) > max_length:
                continue

            encoded_inputs = tokenizer(
                src_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(device)

            if torch.cuda.device_count() > 1:
                generated_ids = model.module.generate(**encoded_inputs, max_length=max_length)
            else:
                generated_ids = model.generate(**encoded_inputs, max_length=max_length)

            pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            predictions.append(pred_text)
            references.append([tgt_text])
            src_texts.append(src_text)

    # Load all metrics
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    ter = evaluate.load("ter")
    meteor = evaluate.load("meteor")
    bleurt = evaluate.load("bleurt", config_name="bleurt-20-D12")
    bertscore = evaluate.load("bertscore")

    # COMET setup
    model_path = download_model("wmt20-comet-da")
    comet_model = load_from_checkpoint(model_path)
    comet_inputs = [
        {"src": src, "ref": ref[0], "mt": pred}
        for src, ref, pred in zip(src_texts, references, predictions)
    ]
    comet_score = comet_model.predict(comet_inputs, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)

    # Compute metrics
    print(f"\nProcessed {len(predictions)} examples (skipped those > {max_length} tokens).")

    bleu_result = bleu.compute(predictions=predictions, references=references)
    print(f"BLEU Score: {bleu_result['score']:.2f}")

    chrf_result = chrf.compute(predictions=predictions, references=references)
    print(f"CHRF Score: {chrf_result['score']:.2f}")

    ter_result = ter.compute(predictions=predictions, references=references)
    print(f"TER Score: {ter_result['score']:.2f}")

    meteor_result = meteor.compute(predictions=predictions, references=[ref[0] for ref in references])
    print(f"METEOR Score: {meteor_result['meteor']:.2f}")

    bleurt_result = bleurt.compute(predictions=predictions, references=[ref[0] for ref in references])
    print(f"BLEURT Score (avg): {sum(bleurt_result['scores']) / len(bleurt_result['scores']):.4f}")

    bert_result = bertscore.compute(predictions=predictions, references=[ref[0] for ref in references], lang="de")
    bert_result = bertscore.compute(
        predictions=predictions,
        references=[ref[0] for ref in references],
        model_type="bert-base-multilingual-cased",
        lang="de",
        idf=False,  
    )
    print(f"BERTScore F1 (avg): {sum(bert_result['f1']) / len(bert_result['f1']):.4f}")

    print(f"COMET Score (avg): {sum(comet_score.scores)/len(comet_score.scores):.4f}")



# Run the evaluation for your shared checkpoint
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
