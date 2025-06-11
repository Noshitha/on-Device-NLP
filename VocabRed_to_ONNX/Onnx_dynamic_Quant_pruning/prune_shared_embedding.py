# prune_shared_embeddings.py

import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto
def strip_shared_embeddings(input_model: str, output_model: str, threshold: int = 20_000_000):
    """
    Remove any initializer whose element count exceeds `threshold`.
    Default threshold (20 million) will catch the ~30M-element embedding.
    """
    model = onnx.load(input_model)
    graph = model.graph
    # Remember each initializer’s dims so we can re-add it as an input
    init_info = {init.name: list(init.dims) for init in graph.initializer}
    init_sizes = {name: int(np.prod(dims)) for name,dims in init_info.items()}
    to_remove = {name for name, size in init_sizes.items() if size > threshold}

    if not to_remove:
        print(f"No initializers over {threshold} elements found in {input_model}.")
    else:
        print(f"Found {len(to_remove)} large initializers in {input_model}:")
        for name in to_remove:
            print(f"  {name}: {init_sizes[name]} elements")

        # Remove each initializer by name
        for name in to_remove:
            # remove from initializers
            for init in graph.initializer:
                if init.name == name:
                    graph.initializer.remove(init)
                    break
    
            # # remove matching graph inputs (ValueInfoProto)
            # for inp in list(graph.input):
            #     if inp.name == name:
            #         graph.input.remove(inp)

            
        # now re-declare that tensor as a graph input so ONNXRuntime will expect it
        graph.input.append(
            helper.make_tensor_value_info(
                name,
                TensorProto.FLOAT,
                init_info[name]
            )
        )

    onnx.save(model, output_model)
    print(f"Pruned model saved to {output_model}.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prune large shared embeddings from an ONNX model."
    )
    parser.add_argument(
        "-i", "--input-model",  required=True,
        help="Path to the original ONNX file (encoder.onnx or decoder.onnx)"
    )
    parser.add_argument(
        "-o", "--output-model", required=True,
        help="Path for the pruned ONNX file"
    )
    parser.add_argument(
        "-t", "--threshold", type=int, default=20_000_000,
        help="Element‐count threshold to identify embeddings (default: 20M)"
    )
    args = parser.parse_args()
    strip_shared_embeddings(args.input_model, args.output_model, args.threshold)