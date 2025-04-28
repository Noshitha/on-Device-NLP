import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize(path):
    """
    Quantize the weights of the model from float32 to int8 for efficient inference.
    Args:
        path: Path to the exported ONNX model.
    Returns:
        None (overwrites model in-place).
    """
    print("Quantizing...")

    quantize_dynamic(
        model_input=path,
        model_output=path,
        weight_type=QuantType.QInt8  
    )

    print("Done.")
