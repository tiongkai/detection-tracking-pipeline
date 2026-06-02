"""Export detection (YOLO) and ReID models to ONNX.

ONNX is the PORTABLE artifact — ship these with the app. They are independent
of the target GPU, CUDA, and TensorRT version. On each deployment machine,
build a hardware-specific TensorRT engine from the ONNX with build_tensorrt.py.

Why two stages:
    PyTorch .pt  --(this script, once)-->  .onnx  (portable, ship this)
    .onnx  --(build_tensorrt.py, on target)-->  .engine  (hardware-specific)

A TensorRT .engine is tied to the exact GPU architecture + TensorRT version +
CUDA version it was built on. It will fail to load on different hardware, so
engines must be built on (or matched to) the deployment machine.

Usage:
    # Export both detection and ReID models to ONNX
    python export/export_onnx.py \
        --yolo-weights weights/best.pt \
        --reid-weights osnet_x0_25_msmt17.pt \
        --out export/onnx

    # Detection only, FP16, fixed batch
    python export/export_onnx.py --yolo-weights weights/best.pt --half
"""

import argparse
import shutil
from pathlib import Path


def export_yolo(weights, out_dir, imgsz=640, half=False, opset=17, simplify=True, dynamic=False):
    """Export a YOLO .pt to ONNX via ultralytics. Returns the output path."""
    from ultralytics import YOLO

    model = YOLO(weights)
    print(f"[YOLO] Exporting {weights} (imgsz={imgsz}, half={half}, dynamic={dynamic})")
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
        device=0,
    )
    onnx_path = Path(onnx_path)
    dest = Path(out_dir) / onnx_path.name
    if onnx_path.resolve() != dest.resolve():
        shutil.move(str(onnx_path), str(dest))
    print(f"[YOLO] -> {dest}")
    return dest


def export_reid(weights, out_dir, half=False, opset=17, simplify=True, dynamic=True):
    """Export a boxmot ReID .pt to ONNX via direct torch.onnx.export.

    Bypasses boxmot's bundled exporter (which has a fragile auto-pip-install
    decorator) and drives torch.onnx.export directly on the underlying module.
    """
    import torch
    from boxmot.appearance.reid.auto_backend import ReidAutoBackend
    from boxmot.appearance.reid.registry import ReIDModelRegistry
    from boxmot.utils.torch_utils import select_device

    weights = Path(weights)
    device = select_device("0")
    backend = ReidAutoBackend(weights=weights, device=device, half=half)
    model = backend.model.model.eval()
    model_name = ReIDModelRegistry.get_model_name(weights)

    # ReID input size depends on the model family (matches boxmot's export.py)
    if "vehicleid" in weights.name or "veri" in weights.name:
        imgsz = (256, 256)
    elif "lmbn" in model_name:
        imgsz = (384, 128)
    elif "hacnn" in model_name:
        imgsz = (160, 64)
    else:
        imgsz = (256, 128)

    dummy = torch.empty(1, 3, imgsz[0], imgsz[1]).to(device)
    if half:
        dummy = dummy.half()
        model = model.half()
    for _ in range(2):
        _ = model(dummy)

    dest = Path(out_dir) / f"{weights.stem}.onnx"
    dynamic_axes = {"images": {0: "batch"}, "output": {0: "batch"}} if dynamic else None

    print(f"[ReID] Exporting {weights} (imgsz={imgsz}, half={half}, dynamic={dynamic})")
    torch.onnx.export(
        model, dummy, str(dest),
        input_names=["images"], output_names=["output"],
        opset_version=opset, dynamic_axes=dynamic_axes,
    )

    if simplify:
        try:
            import onnxslim
            onnxslim.slim(str(dest), str(dest))
            print("[ReID] simplified with onnxslim")
        except Exception as e:
            print(f"[ReID] onnxslim skipped: {e}")

    print(f"[ReID] -> {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Export YOLO + ReID models to ONNX (portable artifact).")
    parser.add_argument("--yolo-weights", default=None, help="Path to YOLO .pt")
    parser.add_argument("--reid-weights", default=None, help="Path to ReID .pt")
    parser.add_argument("--out", default="export/onnx", help="Output directory for ONNX files")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input size")
    parser.add_argument("--half", action="store_true", help="Export in FP16")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--no-simplify", action="store_true", help="Skip onnxslim simplification")
    parser.add_argument("--dynamic", action="store_true",
                        help="Dynamic batch axis (YOLO). ReID always uses dynamic batch.")
    args = parser.parse_args()

    if not args.yolo_weights and not args.reid_weights:
        parser.error("Provide at least one of --yolo-weights or --reid-weights")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    simplify = not args.no_simplify

    exported = []
    if args.yolo_weights:
        exported.append(export_yolo(
            args.yolo_weights, out_dir, imgsz=args.imgsz, half=args.half,
            opset=args.opset, simplify=simplify, dynamic=args.dynamic,
        ))
    if args.reid_weights:
        exported.append(export_reid(
            args.reid_weights, out_dir, half=args.half,
            opset=args.opset, simplify=simplify, dynamic=True,
        ))

    print(f"\nExported {len(exported)} ONNX file(s) to {out_dir}/")
    print("These are portable — ship them. Build TensorRT engines on each target with build_tensorrt.py.")


if __name__ == "__main__":
    main()
