import numpy as np

from timeit import timeit

from transformers import MarianMTModel, MarianTokenizer

import sys
sys.path.append("/home/njuttu_umass_edu/on-DeviceNLP/Baseline_to_ONNX/Onnx_dynamic_Quant_pruned") 
from core.marian import MarianOnnx

NUMBER = 100
"""
Performance and correctness verification
Runs CPU and GPU benchmarks comparing ONNX vs PyTorch
Verifies ONNX output using np.testing.assert_allclose()
"""

def verify_export(model_path, onnx_path):
    print("Verifying export...")
    ref_model = MarianMTModel.from_pretrained(model_path ,local_files_only=True)
    model = MarianOnnx(onnx_path)

    tokenizer = MarianTokenizer.from_pretrained(model_path ,local_files_only=True)
    inputs = tokenizer(["Hello world !"], return_tensors="pt")

    ref_output = ref_model.generate(**inputs, max_length=50)
    output = model.generate(**inputs, max_length=50)

    np.testing.assert_allclose(ref_output.cpu().numpy(), output, rtol=1e-3, atol=1e-3)
    print("Model outputs from torch and ONNX Runtime are similar.")
    print("Success.")


def gpu_benchmark(model_path, onnx_path):
    model_ref = MarianMTModel.from_pretrained(model_path, local_files_only=True).to('cuda')
    model = MarianOnnx(onnx_path, device='cuda')

    tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
    input_ids = tokenizer(["Hello world !"], return_tensors="pt").to('cuda')

    print("Warming up ORT...")
    for _ in range(100):
        model.generate(**input_ids)

    print("ORT GPU: ", end="")
    timer = int(timeit(lambda: model.generate(**input_ids), number=NUMBER) * 1000)
    print(f"{timer // NUMBER} ms / sentence")

    print("PyTorch GPU: ", end="")
    timer_ref = int(timeit(lambda: model_ref.generate(**input_ids), number=NUMBER) * 1000)
    print(f"{timer_ref // NUMBER} ms / sentence")


def cpu_benchmark(model_path, onnx_path):
    model_ref = MarianMTModel.from_pretrained(model_path, local_files_only=True)
    model = MarianOnnx(onnx_path)

    tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
    input_ids = tokenizer(["Hello world !"], return_tensors="pt")

    print("Warming up ORT...")
    for _ in range(100):
        model.generate(**input_ids)

    print("ORT CPU: ", end="")
    timer = int(timeit(lambda: model.generate(**input_ids), number=NUMBER) * 1000)
    print(f"{timer // NUMBER} ms / sentence")

    print("PyTorch CPU: ", end="")
    timer_ref = int(timeit(lambda: model_ref.generate(**input_ids), number=NUMBER) * 1000)
    print(f"{timer_ref // NUMBER} ms / sentence")


if __name__ == "__main__":
    ONNX_PATH = './local_models/Helsinki-NLP_opus-mt-en-de'  # updated
    MODEL_PATH = './local_models/Helsinki-NLP_opus-mt-en-de'  # updated

    print("CPU Benchmark:\n")
    cpu_benchmark(MODEL_PATH, ONNX_PATH)

    print("\n-----\nGPU Benchmark:\n")
    gpu_benchmark(MODEL_PATH, ONNX_PATH)
