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
    """
    Quantize the embedding weight lm_weight.bin file (and optionally lm_bias.bin).
    """
    lm_weight_path = os.path.join(outdir, 'lm_weight.bin')
    if not os.path.exists(lm_weight_path):
        print(f"lm_weight.bin not found in {outdir}. Skipping embedding quantization.")
        return

    print(f"Quantizing embedding weights: {lm_weight_path}...")
    lm_weight = torch.load(lm_weight_path)

    # Simple static quantization: map float32 -> int8
    scale = lm_weight.abs().max(dim=1, keepdim=True)[0]
    scale[scale == 0] = 1  # Avoid division by zero
    quantized_weight = (lm_weight / scale).clamp(-128, 127).round().to(torch.int8)

    # Save quantized weights (can also save scale if needed for dequantization)
    torch.save(quantized_weight, lm_weight_path)
    print("Done quantizing embeddings.")

def quantize(path, outdir=None):
    """
    Quantize ONNX model and optionally embeddings.
    """
    quantize_onnx(path)
    if outdir:
        quantize_embeddings(outdir)
