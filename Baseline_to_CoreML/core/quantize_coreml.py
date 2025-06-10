# core/quantize_coreml.py
import coremltools as ct
from coremltools.models.neural_network.quantization_utils import quantize_weights

def quantize_coreml(model_path: str, output_path: str, nbits: int = 8):
    """
    Quantize a CoreML model to reduce size.
    Args:
        model_path: Path to input .mlmodel or .mlpackage
        output_path: Path to output quantized .mlmodel or .mlpackage
        nbits: 8 for int8 quantization, 16 for float16
    """
    print(f"Loading model from: {model_path}")
    model = ct.models.MLModel(model_path)
    
    print(f"Quantizing to {nbits} bits...")
    quantized_model = quantize_weights(model, nbits=nbits)
    
    print(f"Saving quantized model to: {output_path}")
    quantized_model.save(output_path)
    
    # Print size comparison
    import os
    if os.path.exists(model_path) and os.path.exists(output_path):
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        compression_ratio = original_size / quantized_size
        print(f"Original size: {original_size:.2f} MB")
        print(f"Quantized size: {quantized_size:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")

def quantize_encoder_decoder(encoder_path: str, decoder_path: str, output_dir: str, nbits: int = 8):
    """Quantize both encoder and decoder models"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    encoder_output = os.path.join(output_dir, f"encoder_quantized_{nbits}bit.mlpackage")
    decoder_output = os.path.join(output_dir, f"decoder_quantized_{nbits}bit.mlpackage")
    
    print("Quantizing encoder...")
    quantize_coreml(encoder_path, encoder_output, nbits)
    
    print("Quantizing decoder...")
    quantize_coreml(decoder_path, decoder_output, nbits)
    
    return encoder_output, decoder_output