# prune_by_name.py
import argparse
import onnx
from onnx import helper, TensorProto

def strip_named_initializers(input_model: str, output_model: str, names_to_strip):
    model = onnx.load(input_model)
    graph = model.graph

    # Record dims for any initializer we remove
    init_info = {init.name: list(init.dims) for init in graph.initializer}
    to_remove = [n for n in names_to_strip if n in init_info]

    if not to_remove:
        print(f"No matching initializers {names_to_strip} in {input_model}")
    else:
        print(f"Stripping {to_remove} from {input_model}")
        for name in to_remove:
            # Remove the matching initializer
            for init in graph.initializer:
                if init.name == name:
                    graph.initializer.remove(init)
                    break
            # Re‐declare it as a graph input
            graph.input.append(
                helper.make_tensor_value_info(
                    name,
                    TensorProto.FLOAT,
                    init_info[name]
                )
            )

    onnx.save(model, output_model)
    print(f"Pruned model saved to {output_model}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-model", required=True, help="Path to original ONNX")
    parser.add_argument("-o", "--output-model", required=True, help="Path for pruned ONNX")
    parser.add_argument("-n", "--names", nargs="+", required=True,
                        help="Names of initializers to strip (e.g. 'encoder.embed_tokens.weight')")
    args = parser.parse_args()
    strip_named_initializers(args.input_model, args.output_model, args.names)
