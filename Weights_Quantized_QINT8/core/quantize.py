import os
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_onnx(path):
    """
    Quantize the ONNX model weights from float32 to int8.
    """
    print(f"Quantizing ONNX model: {path}...")
    quantize_dynamic(
        model_input=path,
        model_output=path,
        weight_type=QuantType.QInt8  
    )
    print("Done quantizing ONNX model.")

def quantize_embeddings(outdir):
    import torch

    lm_weight_path = os.path.join(outdir, 'lm_weight.bin')
    if not os.path.exists(lm_weight_path):
        print(f"lm_weight.bin not found in {outdir}. Skipping embedding quantization.")
        return

    print(f"Quantizing embedding weights: {lm_weight_path}...")
    lm_weight = torch.load(lm_weight_path)

    # Use per-tensor qint8 quantization
    scale = lm_weight.abs().max().item() / 127
    zero_point = 0

    q_weight = torch.quantize_per_tensor(
        lm_weight, scale=scale, zero_point=zero_point, dtype=torch.qint8
    )

    torch.save(q_weight, lm_weight_path)
    print("Done quantizing embeddings.")

    

def quantize(path, outdir=None):
    """
    Quantize ONNX model and optionally embeddings.
    """
    quantize_onnx(path)
    if outdir:
        quantize_embeddings(outdir)
