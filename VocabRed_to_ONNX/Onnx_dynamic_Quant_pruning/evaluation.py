import argparse
import os
from tqdm import tqdm
import torch

# model & tokenization
from transformers import MarianTokenizer
from core.marian import MarianOnnx

# metrics
from sacrebleu.metrics import BLEU, CHRF, TER
import evaluate


# ──────────────────────────────────────────────────────────────────────────────
# Subclass to inject shared embedding into pruned ONNX sessions
class MarianOnnxWithEmb(MarianOnnx):
    def __init__(self, path: str, device: str = "cpu"):
        super().__init__(path, device)
        emb_path = os.path.join(path, "lm_weight.bin")
        w = torch.load(emb_path, map_location="cpu").detach().cpu().numpy()
        self._shared_emb = w

    def _encoder_forward(self, input_ids, attention_mask):
        ort_inputs = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
            "encoder.embed_tokens.weight": self._shared_emb,
        }
        last_hidden = self.encoder_session.run(None, ort_inputs)[0]
        return torch.from_numpy(last_hidden).to(input_ids.device)

    def _decoder_forward(self, input_ids, encoder_output, attention_mask):
        ort_inputs = {
            "input_ids": input_ids.cpu().numpy(),
            "encoder_hidden_states": encoder_output.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
            "decoder.embed_tokens.weight": self._shared_emb,
        }
        decoder_out = self.decoder_session.run(None, ort_inputs)[0]
        lm_logits = torch.nn.functional.linear(
            torch.from_numpy(decoder_out).to(input_ids.device),
            self.final_logits_weight,
            bias=self.final_logits_bias
        )
        return lm_logits
# ──────────────────────────────────────────────────────────────────────────────

def load_data(src_lang="fr", tgt_lang="en", split="validation[:100]"):
    from datasets import load_dataset
    ds = load_dataset("wmt14", f"{src_lang}-{tgt_lang}", split=split)
    return [(ex["translation"][src_lang], ex["translation"][tgt_lang]) for ex in ds]


def generate_all(src_texts, tokenizer, onnx_model, device="cpu"):
    preds = []
    for txt in tqdm(src_texts, desc="Translating"):
        batch = tokenizer(txt, return_tensors="pt", padding=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        out_ids = onnx_model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        preds.append(tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0])
    return preds


def main(args):
    # load tokenizer & pruned ONNX model
    tokenizer = MarianTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    onnx_model = MarianOnnxWithEmb(args.onnx_dir, device=args.device)

    # load data
    data = load_data(args.src, args.tgt, args.split)
    src_texts, refs = zip(*data)
    refs_list = [[r] for r in refs]

    # generate predictions
    preds = generate_all(src_texts, tokenizer, onnx_model, device=args.device)

    # compute sacrebleu metrics
    bleu = BLEU().corpus_score(preds, refs_list)
    chrf = CHRF().corpus_score(preds, refs_list)
    ter  = TER().corpus_score(preds, refs_list)

    # compute METEOR via Hugging Face evaluate
    meteor = evaluate.load("meteor")
    meteor_score = meteor.compute(predictions=preds, references=refs_list)["meteor"]

    # print all metrics
    print(f"\n=== Metrics over {len(preds)} sentences ===")
    print(f"BLEU    = {bleu.score:.2f}")
    print(f"ChrF    = {chrf.score:.2f}")
    print(f"TER     = {ter.score:.2f}")
    print(f"METEOR  = {meteor_score:.4f}")

    # save outputs
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Save each file separately
        pred_path = os.path.join(args.output_dir, "predictions.txt")
        ref_path = os.path.join(args.output_dir, "references.txt")
        src_path = os.path.join(args.output_dir, "source.txt")

        with open(pred_path, "w", encoding="utf-8") as f:
            f.write("\n".join(preds))

        with open(ref_path, "w", encoding="utf-8") as f:
            f.write("\n".join(refs))

        with open(src_path, "w", encoding="utf-8") as f:
            f.write("\n".join(src_texts))

        # Save all together in a readable format
        mix_path = os.path.join(args.output_dir, "full_comparison.txt")
        with open(mix_path, "w", encoding="utf-8") as f:
            for i, (src, pred, ref) in enumerate(zip(src_texts, preds, refs)):
                f.write(f"Example {i+1}:\n")
                f.write(f"SOURCE     : {src}\n")
                f.write(f"PREDICTION : {pred}\n")
                f.write(f"REFERENCE  : {ref}\n")
                f.write("\n" + "-"*60 + "\n\n")

        print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True,
                        help="Folder with tokenizer and config.json")
    parser.add_argument("--onnx-dir", required=True,
                        help="Folder with pruned ONNX files and weights")
    parser.add_argument("--src", default="fr")
    parser.add_argument("--tgt", default="en")
    parser.add_argument("--split", default="validation[:100]")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-dir", default=None,
                        help="Optional folder to save generated translations and references")
    args = parser.parse_args()
    main(args)
