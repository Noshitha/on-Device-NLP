import numpy as np

from timeit import timeit

from transformers import MarianMTModel, MarianTokenizer

from transformers import AutoTokenizer

from core.marian import MarianOnnx

NUMBER = 100
"""
Performance and correctness verification
Runs CPU and GPU benchmarks comparing ONNX vs PyTorch
Verifies ONNX output using np.testing.assert_allclose()
"""

# def verify_export(model_path, onnx_path):
#     print("Verifying export...")

#     ref_model = MarianMTModel.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)

#     #tokenizer = MarianTokenizer.from_pretrained(model_path)
#     inputs = tokenizer(["Hello world !"], return_tensors="pt")

#     ref_output = ref_model.generate(**inputs)
#     output = model.generate(**inputs)

#     np.testing.assert_allclose(ref_output.cpu().numpy(), output, rtol=1e-3, atol=1e-3)
#     print("Model outputs from torch and ONNX Runtime are similar.")
#     print("Success.")
def verify_export(model_path, onnx_path):
    """
    Compare one-sentence generation from PyTorch vs ONNXRuntime.
    """
    print("Verifying export...")

    # 1) Load PyTorch reference model & tokenizer from model_path
    ref_model = MarianMTModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2) Instantiate the ONNX wrapper
    onnx_model = MarianOnnx(onnx_path, device="cpu")

    # 3) Tokenize a single sentence
    inputs = tokenizer(["Hello world !"], return_tensors="pt")

    # 4) PyTorch generation
    ref_output_ids = ref_model.generate(**inputs)

    # 5) ONNXRuntime generation (must pass the two tensors explicitly)
    onnx_output_ids = onnx_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

    # 6) Compare
    np.testing.assert_allclose(
        ref_output_ids.cpu().numpy(),
        onnx_output_ids.cpu().numpy(),
        rtol=1e-3,
        atol=1e-3
    )
    print("Model outputs from torch and ONNX Runtime are similar.")
    print("Success.")


def gpu_benchmark(model_path, onnx_path):
    model_ref = MarianMTModel.from_pretrained(model_path).to('cuda')
    model = MarianOnnx(onnx_path, device='cuda')

    tokenizer = MarianTokenizer.from_pretrained(model_path)
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
    model_ref = MarianMTModel.from_pretrained(model_path)
    model = MarianOnnx(onnx_path)

    tokenizer = MarianTokenizer.from_pretrained(model_path)
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
    MODEL_PATH = "/home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/checkpoint-4924000"
    ONNX_PATH   = "/home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/checkpoint-4924000/onnx_export"

    print("CPU Benchmark:\n")
    cpu_benchmark(MODEL_PATH, ONNX_PATH)

    print("\n-----\nGPU Benchmark:\n")
    gpu_benchmark(MODEL_PATH, ONNX_PATH)