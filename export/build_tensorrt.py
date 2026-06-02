"""Build TensorRT engines from ONNX — run this ON THE DEPLOYMENT MACHINE.

A TensorRT .engine is hardware-specific: it is tied to the GPU compute
capability, the TensorRT version, and the CUDA version of the machine that
built it. You cannot ship a prebuilt .engine and expect it to load on a
different GPU. So the app should ship ONNX files (from export_onnx.py) and
call this script once at install or first launch on each target machine.

The built engines drop straight into the tracking pipeline:
    python track/track_video_predict.py \
        --weights export/engines/best.engine \
        --reid-weights export/engines/osnet_x0_25_msmt17.engine ...

Both ultralytics (YOLO) and boxmot (ReID) auto-detect the .engine suffix and
use the TensorRT backend.

Usage:
    # Build engines for every ONNX in a directory
    python export/build_tensorrt.py --onnx-dir export/onnx --out export/engines --fp16

    # Build a single ONNX
    python export/build_tensorrt.py --onnx export/onnx/best.onnx --fp16
"""

import argparse
from pathlib import Path


def build_engine(onnx_path, engine_path, fp16=True, workspace_gb=4,
                 min_batch=1, opt_batch=1, max_batch=8):
    """Build a TensorRT engine from an ONNX file. Returns engine_path on success."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    onnx_path = Path(onnx_path)
    print(f"[TRT] Parsing {onnx_path.name}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT] FP16 enabled")
        else:
            print("[TRT] WARNING: platform has no fast FP16, building FP32")

    # Handle dynamic input shapes with an optimization profile
    inp = network.get_input(0)
    shape = inp.shape
    if -1 in shape:
        profile = builder.create_optimization_profile()
        min_shape = [min_batch if d == -1 else d for d in shape]
        opt_shape = [opt_batch if d == -1 else d for d in shape]
        max_shape = [max_batch if d == -1 else d for d in shape]
        profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        print(f"[TRT] Dynamic profile {inp.name}: min={min_shape} opt={opt_shape} max={max_shape}")
    else:
        print(f"[TRT] Static input shape: {tuple(shape)}")

    print(f"[TRT] Building engine (this can take minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Engine build failed for {onnx_path}")

    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[TRT] -> {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")
    return engine_path


def _device_tag():
    """Return a short tag describing the build hardware, for engine naming/logging."""
    try:
        import tensorrt as trt
        import torch
        gpu = torch.cuda.get_device_name(0).replace(" ", "_") if torch.cuda.is_available() else "cpu"
        return f"{gpu}_trt{trt.__version__}"
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engines from ONNX (run on deployment machine).")
    parser.add_argument("--onnx", default=None, help="Single ONNX file to build")
    parser.add_argument("--onnx-dir", default=None, help="Directory of ONNX files to build")
    parser.add_argument("--out", default="export/engines", help="Output directory for engines")
    parser.add_argument("--fp16", action="store_true", help="Build FP16 engine (recommended)")
    parser.add_argument("--workspace-gb", type=int, default=4, help="Builder workspace memory pool (GB)")
    parser.add_argument("--max-batch", type=int, default=8, help="Max batch for dynamic-shape engines")
    args = parser.parse_args()

    if not args.onnx and not args.onnx_dir:
        parser.error("Provide --onnx or --onnx-dir")

    print(f"Build hardware: {_device_tag()}")
    print("NOTE: engines are valid only on this GPU + TensorRT + CUDA combination.\n")

    onnx_files = []
    if args.onnx:
        onnx_files.append(Path(args.onnx))
    if args.onnx_dir:
        onnx_files.extend(sorted(Path(args.onnx_dir).glob("*.onnx")))

    out_dir = Path(args.out)
    built = []
    for onnx_path in onnx_files:
        engine_path = out_dir / f"{onnx_path.stem}.engine"
        built.append(build_engine(
            onnx_path, engine_path, fp16=args.fp16,
            workspace_gb=args.workspace_gb, max_batch=args.max_batch,
        ))

    print(f"\nBuilt {len(built)} engine(s) in {out_dir}/")


if __name__ == "__main__":
    main()
