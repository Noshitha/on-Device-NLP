# %%
import torch
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
import evaluate
from itertools import islice
from torch.utils.data import DataLoader
import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch3/workspace/sudhanshukul_umass_edu-sudhanshuCache'
def evaluate_translation(model_name, src_lang, tgt_lang, dataset_name, dataset_config, batch_size=16, num_samples=500):
    # Load the pre-trained model and tokenizer
    print("Loading the model and tokenizer now")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    print("Loaded the model and tokenizer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print("Loading the dataset now")
    dataset = load_dataset(dataset_name, dataset_config, split="test[:5]")
    print("Loaded the dataset now")
    def identity_collate(batch):
        return batch
    dataloader = DataLoader(dataset, batch_size=batch_size,collate_fn=identity_collate)
    predictions = []
    references = []
    print(f"Evaluating {model_name} for {src_lang} → {tgt_lang} on {len(dataset)} examples in batches of {batch_size}")
    def get_texts(ex, src_lang, tgt_lang):
        
        translation = ex["translation"]
        return translation.get(src_lang, ""), translation.get(tgt_lang, "")

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            src_texts = []
            tgt_texts = []
            for ex in batch:
                
        
                #%%
                src, tgt = get_texts(ex, src_lang, tgt_lang)
                src_texts.append(src)
                tgt_texts.append(tgt)
            
            # Tokenize the batch of source texts
            encoded_inputs = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True)
            encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}
            
            # Generate translations for the batch
            generated_ids = model.module.generate(**encoded_inputs, max_length=128)


            batch_preds = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            
            predictions.extend(batch_preds)
            # sacreBLEU expects each reference to be a list of references per prediction
            references.extend([[t] for t in tgt_texts])

    bleu = evaluate.load("sacrebleu")
    # Compute BLEU; note that references need to be formatted correctly
    result = bleu.compute(predictions=predictions, references=references)

    print(f"BLEU score for {src_lang} → {tgt_lang} using {model_name}: {result['score']:.2f}\n")

#27.27 - test set of 500 examples

# evaluate_translation(
#     model_name="Helsinki-NLP/opus-mt-en-de",
#     src_lang="en",
#     tgt_lang="de",
#     dataset_name="wmt14",
#     dataset_config="de-en"
# )


evaluate_translation(
    model_name="Helsinki-NLP/opus-mt-en-fr",
    src_lang="en",
    tgt_lang="fr",
    dataset_name="wmt14",
    dataset_config="fr-en"
)
