# core/quantize_coreml.py
import coremltools as ct
from coremltools.models.neural_network.quantization_utils import quantize_weights

def quantize_coreml(model_path: str, output_path: str, nbits: int = 8):
    """
    Quantize a CoreML model to reduce size.
    Args:
        model_path: Path to input .mlmodel
        output_path: Path to output quantized .mlmodel
        nbits: 8 for int8 quantization
    """
    model = ct.models.MLModel(model_path)
    quantized_model = quantize_weights(model, nbits=nbits)
    quantized_model.save(output_path)
