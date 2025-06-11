import onnx
import numpy as np

def list_initializers(onnx_path):
    model = onnx.load(onnx_path)
    shown = []
    for init in model.graph.initializer:
        name = init.name
        dims = list(init.dims)
        count = int(np.prod(dims))
        shown.append((name, dims, count))
    # Sort by descending element‐count
    shown.sort(key=lambda x: x[2], reverse=True)
    print(f"Initializers in {onnx_path}:")
    for name, dims, count in shown:
        print(f"  • {name:40s}  {dims}  →  {count:,} elements")
    print()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python list_initializers.py <path/to/encoder_or_decoder.onnx>")
        sys.exit(1)
    list_initializers(sys.argv[1])
