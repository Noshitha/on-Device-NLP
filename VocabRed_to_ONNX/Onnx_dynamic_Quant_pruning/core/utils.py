# ─── core/utils.py ───────────────────────────────────────────────────────────────

import os
import shutil
import torch
from transformers import AutoTokenizer
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from core.layers import MarianDecoder, MarianEncoder
from core.quantize import quantize
from transformers import MarianMTModel, MarianTokenizer

def create_model_for_provider(path: str, provider: str):
    """Create an ONNX Runtime session, given an .onnx file path and a provider."""
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    session = InferenceSession(path, options, providers=[provider])
    session.disable_fallback()
    return session

def create_marian_encoder_decoder(model_path: str, outdir: str):
    """Load MarianMTModel.from_pretrained(model_path), wrap encoder & decoder, save shared weights."""
    model = MarianMTModel.from_pretrained(model_path)
    encoder = model.get_encoder()
    decoder = model.get_decoder()

    marian_encoder = MarianEncoder(encoder).eval()
    marian_decoder = MarianDecoder(decoder).eval()

    # Save shared embeddings & bias
    torch.save(model.model.shared.weight, os.path.join(outdir, 'lm_weight.bin'))
    torch.save(model.final_logits_bias,   os.path.join(outdir, 'lm_bias.bin'))

    # Copy any tokenizer/config files that exist (vocab.json, source.spm, etc.)
    possible_files = [
        'config.json',
        'tokenizer_config.json',
        'vocab.json',
        'tokenizer.json',
        'special_tokens_map.json',
        'source.spm',
        'target.spm'
    ]
    for fname in possible_files:
        src = os.path.join(model_path, fname)
        dst = os.path.join(outdir, fname)
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)

    return marian_encoder, marian_decoder

def generate_onnx_graph(model_path: str,
                        tokenizer_path: str,
                        encoder_path: str,
                        decoder_path: str,
                        outdir: str,
                        quant: bool = True):
    """
    1) Load MarianMTModel from `model_path` (a folder with config.json+model.safetensors)
    2) Create encoder & decoder wrappers, save lm_weight.bin + lm_bias.bin to `outdir`
    3) Export encoder.onnx and decoder.onnx (with dynamic axes)
    4) If quant=True, quantize each ONNX file (int8).
    """
    # 1) Create encoder/decoder modules and save shared weights to outdir
    encoder, decoder = create_marian_encoder_decoder(model_path, outdir)

    # 2) Load tokenizer from the reduced-vocab folder, NOT from model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    inputs = tokenizer("Hello World !", return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 3) Export encoder to ONNX
    print("Exporting encoder to ONNX...")
    torch.onnx._export(
        encoder,
        (input_ids, attention_mask),
        encoder_path,
        export_params=True,
        opset_version=12,
        input_names=["input_ids", "attention_mask"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
        }
    )
    if quant:
        quantize(encoder_path)

    # 4) Export decoder to ONNX
    print("Exporting decoder to ONNX...")
    # We need encoder_hidden_states from a single forward pass
    encoder_hidden_states = encoder(input_ids, attention_mask)[0]
    torch.onnx._export(
        decoder,
        (input_ids, encoder_hidden_states, attention_mask),
        decoder_path,
        export_params=True,
        opset_version=12,
        input_names=["input_ids", "encoder_hidden_states", "attention_mask"],
        output_names=["decoder_output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
            "decoder_output": {0: "batch", 1: "sequence"},
        }
    )
    if quant:
        quantize(decoder_path)

    print("Done exporting both ONNX files.")

