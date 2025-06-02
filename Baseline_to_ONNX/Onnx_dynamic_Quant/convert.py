import os
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("/home/njuttu_umass_edu/on-DeviceNLP/Baseline_to_ONNX/Onnx_dynamic_Quant")

import shutil
import argparse
from transformers import MarianMTModel, MarianTokenizer
from core.utils import generate_onnx_graph
from core.benchmark import verify_export

PARAMS = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=str,
        help="Hugging Face model identifier, e.g., 'Helsinki-NLP/opus-mt-en-de'"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./out",
        help="Output directory."
    )
    parser.add_argument(
        "--no-quantize",
        action="store_false",
        default=True,
        help="Disable model quantization."
    )
    return parser.parse_args()

def download_model(model_name, local_dir):
    print(f"Downloading model '{model_name}' to '{local_dir}'...")
    MarianMTModel.from_pretrained(model_name).save_pretrained(local_dir)
    MarianTokenizer.from_pretrained(model_name).save_pretrained(local_dir)

def main():
    model_name = PARAMS.input
    local_model_dir = os.path.join("./local_model", model_name.replace('/', '_'))
    os.makedirs(local_model_dir, exist_ok=True)
    download_model(model_name, local_model_dir)

    outdir = os.path.join(PARAMS.output, os.path.basename(local_model_dir))
    os.makedirs(outdir, exist_ok=True)

    encoder_path = os.path.join(outdir, "encoder.onnx")
    decoder_path = os.path.join(outdir, "decoder.onnx")

    generate_onnx_graph(local_model_dir, encoder_path, decoder_path, outdir, quant=PARAMS.no_quantize)

    try:
        verify_export(local_model_dir, outdir)
    except Exception as e:
        print(f"Verification error: {e}")

    print("Creating archive file...")
    shutil.make_archive(outdir, format="zip", root_dir=outdir)
    print("Done.")

if __name__ == "__main__":
    PARAMS = parse_args()
    main()
