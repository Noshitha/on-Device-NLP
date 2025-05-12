import torch
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader
from comet import download_model, load_from_checkpoint

def evaluate_translation(model_name, src_lang, tgt_lang, dataset_name, dataset_config, batch_size=8, max_length=128, num_samples=500):
    # Load the model and tokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Load only a subset of the dataset
    dataset = load_dataset(dataset_name, dataset_config, split=f"test[:{num_samples}]")

    # Identity collate function so the DataLoader doesn't do extra merging
    def identity_collate(batch):
        return batch

    # DataLoader with batch_size=1
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=identity_collate)

    predictions = []
    references = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # batch is a list of 1 element since batch_size=1
            ex = batch[0]
            
            # Extract source and target text
            src_text = ex["translation"][src_lang]
            tgt_text = ex["translation"][tgt_lang]

            # Check token length before generation
            tokens = tokenizer(src_text, truncation=False, add_special_tokens=False).input_ids
            if len(tokens) > max_length:
               
                continue

            # Tokenize with actual truncation for generation
            encoded_inputs = tokenizer(
                src_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(device)

            # Generate translation
            if torch.cuda.device_count() > 1:
                generated_ids = model.module.generate(**encoded_inputs, max_length=max_length)
            else:
                generated_ids = model.generate(**encoded_inputs, max_length=max_length)

            # Decode predictions
            pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            predictions.append(pred_text)
            # Each reference is itself a list of one reference string
            references.append([tgt_text])

    # Compute BLEU score
    # bleu = evaluate.load("sacrebleu")
    # results = bleu.compute(predictions=predictions, references=references)

    # print(f"Processed {len(predictions)} examples (skipped those > {max_length} tokens).")
    # print(f"BLEU Score for {src_lang} → {tgt_lang}: {results['score']:.2f}")

    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    ter = evaluate.load("ter")
    meteor = evaluate.load("meteor")

    # Download and load COMET model
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)

    # COMET expects a list of dicts with: src, ref, mt
    comet_inputs = [
        {"src": ex["translation"][src_lang], "ref": ref[0], "mt": pred}
        for ex, pred, ref in zip(dataset, predictions, references)
    ]

    # Predict scores (GPU if available)
    comet_score = comet_model.predict(comet_inputs, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)


    bleu_result = bleu.compute(predictions=predictions, references=references)
    chrf_result = chrf.compute(predictions=predictions, references=references)
    ter_result = ter.compute(predictions=predictions, references=references)
    meteor_result = meteor.compute(predictions=predictions, references=references)

    print(f"Processed {len(predictions)} examples (skipped those > {max_length} tokens).")
    print(f"BLEU Score: {bleu_result['score']:.2f}")
    print(f"CHRF Score: {chrf_result['score']:.2f}")
    print(f"TER Score: {ter_result['score']:.2f}")
    print(f"METEOR Score: {meteor_result['meteor']:.2f}")
    print(f"COMET Score (avg): {sum(comet_score.scores)/len(comet_score.scores):.4f}")





# evaluate_translation(
#     model_name="Helsinki-NLP/opus-mt-en-fr",
#     src_lang="en",
#     tgt_lang="fr",
#     dataset_name="wmt14",
#     dataset_config="fr-en",
#     batch_size=1,      # Only process one sample at a time
#     max_length=128,    # Skip examples exceeding 128 tokens
#     num_samples=1000    # Evaluate first 100 test samples
# )

evaluate_translation(
    model_name="Helsinki-NLP/opus-mt-en-de",
    src_lang="en",
    tgt_lang="de",
    dataset_name="wmt14",
    dataset_config="de-en",
    batch_size=1,      # Only process one sample at a time
    max_length=128,    # Skip examples exceeding 128 tokens
    num_samples=10    # Evaluate first 100 test samples
)

# 27.76 on 500 samples
# evaluate_translation(
#     model_name="Helsinki-NLP/opus-mt-en-de",
#     src_lang="en",
#     tgt_lang="de",
#     dataset_name="wmt14",
#     dataset_config="de-en"
# )



