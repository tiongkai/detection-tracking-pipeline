# Detection

## Overview

`detect.py` runs detection inference and outputs COCO-format predictions JSON. It supports three inference methods and includes per-image latency profiling for assessing real-time viability.

This is the first step in both the detection eval pipeline and (indirectly) the tracking pipeline — the tracker relies on the same YOLO model for per-frame detections.

## Inference Methods

### YOLO (production)

Standard single-pass inference. The image is resized to 640x640, run through the model once, and NMS is applied. This is what the tracking pipeline uses in real time.

```bash
python detection/detect.py --method yolo --weights weights/best.pt \
    --gt-json .../test/_annotations.coco.json \
    --out results/detections/yolo.json
```

**When to use:** Default for everything. This is the production inference path.

### SAHI (small object improvement)

Slicing Aided Hyper Inference. The image is sliced into overlapping patches (default 640x640), each patch is run through the detector independently, then results are merged with NMS to deduplicate detections at patch boundaries.

```bash
python detection/detect.py --method sahi --weights weights/best.pt \
    --gt-json .../test/_annotations.coco.json \
    --slice-size 640 --overlap-ratio 0.2 \
    --out results/detections/sahi.json
```

**When to use:** Offline evaluation to measure recall ceiling on small objects (distant boats, heads, torsos). Not suitable for real-time tracking due to latency — a 1080p image sliced into ~6 patches means ~6x inference time.

**SAHI is model-agnostic.** It wraps any detector via `--sahi-model-type`:
- `ultralytics` (default) — for YOLO weights (`.pt`)
- `huggingface` — for HuggingFace-hosted models (RT-DETR, DETR, etc.)

### RF-DETR (experimental)

Uses the `rfdetr` package (Roboflow's RT-DETR variant). This was an experimental comparison — the production model is YOLO.

```bash
python detection/detect.py --method rfdetr --weights path/to/checkpoint.pth \
    --gt-json .../test/_annotations.coco.json \
    --out results/detections/rfdetr.json
```

**Note on SAHI + RF-DETR:** RF-DETR uses Roboflow's custom inference API, not HuggingFace transformers. `--sahi-model-type huggingface` won't load RF-DETR weights. To use SAHI with DETR-style models, use HuggingFace-hosted RT-DETR checkpoints instead. The underlying architecture (RT-DETR) is the same; only the packaging and weight format differ.

## Latency Profiling

Every run measures per-image latency and reports statistics:

```
==================================================
  Latency Report — yolo
==================================================
  Images:     1099
  Mean:       12.3 ms
  Median:     11.8 ms
  Std:        2.1 ms
  Min/Max:    9.5 / 25.4 ms
  P95:        15.7 ms
  Throughput: 81.3 FPS
==================================================
```

The first image is excluded from stats (GPU warmup: CUDA kernel compilation, memory allocation skew the timing).

**Key metric for real-time:** P95 latency, not mean. At 30 FPS you need every frame under ~33ms. If mean is 12ms but P95 is 35ms, you'll get frame drops.

Timing is saved in the output JSON under `"timing"` so you can compare methods programmatically.

## Output Format

```json
{
  "predictions": [
    {"image_id": 1, "category_id": 0, "bbox": [x, y, w, h], "score": 0.92},
    ...
  ],
  "timing": {
    "n_images": 1099,
    "mean_ms": 12.3,
    "median_ms": 11.8,
    "p95_ms": 15.7,
    "fps": 81.3,
    ...
  },
  "config": {
    "method": "yolo",
    "weights": "weights/best.pt",
    "conf": 0.001,
    "enable_nms": false
  }
}
```

See `detection/sample_detection_output.json` for the full schema.

## Cross-modal NMS

The 12-class domain-split model can fire both RGB and thermal class variants on the same object (e.g. `boat-rgb` + `boat-thermal`). Use `--enable-nms --nms-iou-thresh 0.5` to suppress the lower-confidence duplicate before saving predictions. This is the same NMS used in the tracking pipeline.

## Decisions

**YOLO is the production model.** RF-DETR was evaluated as an alternative but YOLO (YOLOv26-L) is what's deployed. All tracker tuning and parameter experiments (task 2.5) use YOLO detections.

**SAHI is for offline analysis only.** The latency multiplier (~Nx for N patches) makes it impractical for real-time tracking at 30 FPS. Its value is measuring the recall ceiling — if SAHI significantly improves recall on small objects, that tells you the model can detect them when they're large enough in the frame, and the problem is resolution loss from resizing. The fix would then be either higher-resolution inference (e.g. 1280x1280 input) or a tiled deployment strategy, not SAHI in production.

**RF-DETR weights are not compatible with SAHI's HuggingFace backend.** RF-DETR (`rfdetr` package) uses Roboflow's custom checkpoint format and inference API. SAHI's `huggingface` model type expects HuggingFace `transformers`-format checkpoints. Same underlying architecture (RT-DETR), different packaging. If you want SAHI + DETR, use HuggingFace-hosted RT-DETR weights.
