import os
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append("/home/njuttu_umass_edu/on-DeviceNLP-2/Vocab_Red_to_ONNX/Onnx_dynamic_Quant") 

import shutil
import argparse
from transformers import MarianMTModel, MarianTokenizer,AutoTokenizer
from core.utils import generate_onnx_graph
from core.benchmark import verify_export

PARAMS = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Path to the fine-tuned model (with safetensors)")
    parser.add_argument("--tokenizer-dir", type=str, required=True, help="Path to the reduced vocabulary tokenizer directory")
    parser.add_argument("-o", "--output", type=str, default="./out", help="Output directory for ONNX export")
    parser.add_argument("--no-quantize", action="store_false", default=True, help="Disable model quantization.")
    return parser.parse_args()

def main():
    model_dir = PARAMS.model_dir
    tokenizer_dir = PARAMS.tokenizer_dir

    model_name = os.path.basename(model_dir.rstrip("/"))
    outdir = os.path.join(PARAMS.output, model_name)
    os.makedirs(outdir, exist_ok=True)

    encoder_path = os.path.join(outdir, "encoder.onnx")
    decoder_path = os.path.join(outdir, "decoder.onnx")

    generate_onnx_graph(model_dir, tokenizer_dir, encoder_path, decoder_path, outdir, quant=PARAMS.no_quantize)

    try:
        verify_export(model_dir, outdir)
    except Exception as e:
        print(f"Verification error: {e}")
        print("Skipping verification and continuing to archive...")

if __name__ == "__main__":
    PARAMS = parse_args()
    main()