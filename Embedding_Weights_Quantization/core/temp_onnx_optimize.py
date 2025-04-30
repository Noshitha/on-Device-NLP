import onnx
import onnxsim
import onnxoptimizer

def optimize_onnx(path):
    print(f"Optimizing {path}...")

    # Load model
    model = onnx.load(path)

    # Step 1: Simplify (cleans graph)
    model, check = onnxsim.simplify(model)
    if not check:
        raise RuntimeError("Simplification failed!")

    # Step 2: Optimize (fold constants, prune graph)
    passes = [
        'eliminate_deadend',
        'eliminate_identity',
        'eliminate_nop_dropout',
        'eliminate_nop_monotone_argmax',
        'eliminate_nop_pad',
        'eliminate_unused_initializer',
        'fuse_add_bias_into_conv',
        'fuse_consecutive_squeezes',
        'fuse_consecutive_transposes',
        'fuse_matmul_add_bias_into_gemm',
    ]
    model = onnxoptimizer.optimize(model, passes)

    # Save optimized model back
    onnx.save(model, path)
    print(f"Optimization complete: {path}")
