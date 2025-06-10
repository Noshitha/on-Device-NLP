# verify_coreml.py
import argparse
import time
import numpy as np
import coremltools as ct
from transformers import MarianTokenizer
from core.marian_coreml import MarianCoreML

def get_fixed_length(mlpackage_path, input_name="input_ids"):
    """Read the .mlmodel spec to find the fixed sequence length."""
    mlmodel = ct.models.MLModel(str(mlpackage_path))
    spec    = mlmodel.get_spec()
    for inp in spec.description.input:
        if inp.name == input_name:
            return list(inp.type.multiArrayType.shape)[1]
    raise ValueError(f"Couldn’t find input '{input_name}' in {mlpackage_path}")

def pad_or_truncate(ids: np.ndarray, target_len: int, pad_id: int):
    """ids: (1, n); return (1, target_len)."""
    n = ids.shape[1]
    if n >= target_len:
        return ids[:, :target_len]
    pad = np.full((1, target_len - n), pad_id, dtype=ids.dtype)
    return np.concatenate([ids, pad], axis=1)

def verify_export(model_dir: str):
    print(f"\nVerifying CoreML export in '{model_dir}'\n" + "="*60)
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model     = MarianCoreML(model_dir)

    # figure out your fixed encoder/decoder seq-lens
    enc_len = get_fixed_length(f"{model_dir}/encoder.mlpackage",   "input_ids")
    dec_len = get_fixed_length(f"{model_dir}/decoder.mlpackage",   "input_ids")
    pad_id  = tokenizer.pad_token_id
    start_id= model.decoder_start_token_id

    tests = [
        "Hello world!",
        "How are you today?",
        "This is a longer sentence to see truncation/padding.",
        "Machine translation is fascinating."
    ]

    for text in tests:
        # tokenize
        toks = tokenizer(text, return_tensors="pt")
        ids  = toks.input_ids.numpy().astype(np.float32)
        mask = toks.attention_mask.numpy().astype(np.float32)

        # pad/truncate to fixed encoder length
        ids_fixed  = pad_or_truncate(ids,  enc_len, pad_id)
        mask_fixed = pad_or_truncate(mask, enc_len, 0)

        # encode
        start = time.time()
        enc_hidden, enc_mask = model.encode(ids_fixed, mask_fixed)
        
        # prepare decoder input (1×dec_len) with start token in pos0
        dec_input = np.full((1, dec_len), pad_id, dtype=np.float32)
        dec_input[0,0] = start_id

        # run one step of decoder just to sanity-check
        try:
            out = model.decoder.predict({
                "input_ids":           dec_input,
                "encoder_hidden_states": enc_hidden,
                "attention_mask":      enc_mask
            })
        except Exception as e:
            print(f"❌ Decoder predict failed: {e}")
        duration = time.time() - start

        # full generate (this will again pad internally)
        out_ids = model.generate(ids_fixed, mask_fixed, max_length=50)
        translation = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]

        print(f"Input       → {text}")
        print(f"Translation → {translation!r}")
        print(f"Time        → {duration:.3f}s")
        print("-"*60)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Sanity‐check CoreML encoder+decoder on macOS"
    )
    p.add_argument("model_dir", help="Your CoreML export folder")
    args = p.parse_args()
    verify_export(args.model_dir)