# Detection & Tracking Pipeline — CLAUDE.md

## What This Repo Does

A standalone detection and tracking pipeline for maritime surveillance across RGB and thermal domains. Contains inference, tracking, evaluation, and cross-modal NMS — no training code.

The detection model (YOLOv26-L) is pre-trained and included at `weights/best.pt`. The tracker is HybridSORT from the `boxmot` library with enhanced ReID configuration.

---

## Repository Layout

```
detection-tracking-pipeline/
├── weights/
│   └── best.pt                     # YOLOv26-L (12-class domain-split, split_v7)
├── track/
│   ├── track_video.py              # Standard HybridSORT tracking (detection every frame)
│   ├── track_video_predict.py      # Tracking + Kalman prediction + interclass NMS
│   └── cross_modal_nms.py          # Cross-modal NMS module
├── eval/
│   ├── eval.py                     # Evaluation (inference + metrics + interclass NMS)
│   ├── cross_modal_nms.py          # Cross-modal NMS module (eval copy)
│   ├── compare_models.py           # Cross-experiment comparison
│   ├── image_metrics.py            # Per-image quality metrics
│   ├── failure_analysis.py         # Failure analysis + visualisations
│   ├── compute_split_metrics.py    # Split-level metrics
│   └── eval.md                     # Eval documentation
├── configs/                        # Split + experiment YAML configs
├── track.md                        # HybridSORT parameter reference + troubleshooting
├── tracker_eval.md                 # Tracker run history, configs, eval data, quickstart
├── task.md                         # Intern task list (tracking metrics evaluation)
├── CHANGELOG.md                    # Cross-modal NMS changelog
└── README.md                       # Model documentation (architecture, training data, classes)
```

---

## Detection Model

**Architecture:** YOLOv26-L (~25M params), 640x640 input
**Weights:** `weights/best.pt`
**Framework:** Ultralytics (`from ultralytics import YOLO`)

### 12 Classes (domain-split taxonomy)

| ID | Class | Domain |
|----|-------|--------|
| 0 | boat-rgb | RGB |
| 1 | vessel-rgb | RGB |
| 2 | human-rgb | RGB |
| 3 | outboard motor-rgb | RGB |
| 4 | head-rgb | RGB |
| 5 | torso-rgb | RGB |
| 6 | boat-thermal | Thermal |
| 7 | vessel-thermal | Thermal |
| 8 | human-thermal | Thermal |
| 9 | outboard motor-thermal | Thermal |
| 10 | head-thermal | Thermal (synthetic) |
| 11 | torso-thermal | Thermal (synthetic) |

Domain is encoded in the class name. The model outputs separate classes for RGB and thermal appearances. Cross-modal NMS suppresses duplicates when both fire on the same object.

### Val Performance

mAP50: 0.790 | mAP50-95: 0.519 | Precision: 0.843 | Recall: 0.702

---

## Tracking

We use **HybridSORT** from `boxmot` with CLIP-based vehicle ReID (`clip_veri.pt`).

Two tracking scripts:
- `track/track_video.py` — standard tracking, only draws detected boxes
- `track/track_video_predict.py` — also draws Kalman-predicted boxes (dashed outline) when detection fails, supports interclass NMS via `--enable-nms`

### Current Tracker Config

See `track.md` for the full parameter reference. Key settings:

```python
HybridSort(
    reid_weights="clip_veri.pt",
    per_class=True, nr_classes=12,
    det_thresh=0.3, max_age=180, min_hits=3, iou_threshold=0.15,
    alpha=0.7, longterm_bank_length=150, longterm_reid_weight=0.25,
    longterm_reid_correction_thresh=0.5,
)
```

### Known Issues

- **Track fragmentation**: new track IDs created despite existing track being alive. See `track.md` "Troubleshooting" section for causes and tuning parameters.
- **No dead-track gallery**: once a track exceeds `max_age` (180 frames), its embeddings are gone. No cosine similarity check against dead tracks before creating new IDs.
- **No cross-class ReID**: `per_class=True` means `boat-rgb` tracks can never match `boat-thermal` detections. Cross-modal NMS (`--enable-nms`) largely averts this by suppressing the lower-confidence duplicate before it reaches the tracker (72% reduction on pt80 twilight clips). Remaining cases are boxes offset enough that IoU < threshold.

### Cross-modal NMS

The 12-class model can fire both RGB and thermal variants on the same object (e.g. `boat-rgb` + `boat-thermal`). Cross-modal NMS groups classes by stripping `-rgb`/`-thermal` suffixes and suppresses lower-confidence duplicates within each group.

Available in:
- Tracking: `--enable-nms --nms-iou-thresh 0.5`
- Eval: `--enable-interclass-nms --nms-iou-thresh 0.5`

---

## Conda Environments

| Task | Environment |
|------|-------------|
| Tracking (track/*.py) | `boat-tracker` (torch 2.7.1+cu126, ultralytics 8.4.3, boxmot 15.0.9) |
| Evaluation (eval/*.py) | `obj-det` (pycocotools, pandas, numpy, opencv-python) |

---

## Quick Start

### Run tracking with Kalman prediction + NMS

```bash
conda run -n boat-tracker python track/track_video_predict.py \
    --weights weights/best.pt \
    --source /path/to/clips \
    --out /path/to/output \
    --conf 0.3 --iou 0.5 --ema-alpha 1.0 \
    --max-coast 10 --coast-classes boat \
    --enable-nms --nms-iou-thresh 0.5
```

### Run standard tracking (no Kalman prediction)

```bash
conda run -n boat-tracker python track/track_video.py \
    --weights weights/best.pt \
    --source /path/to/clips \
    --out /path/to/output \
    --conf 0.3 --iou 0.5 --ema-alpha 1.0
```

---

## Eval Data

Videos for tracking evaluation are at `/home/lenovo6/TiongKai/obj-det/data/eval/`:

| Source | Location | Clips |
|--------|----------|-------|
| clips | data/eval/clips/ | 27 team eval clips (thermal + RGB) |
| pt80_clips | data/eval/pt80_clips/ | 54 pt80 camera clips (twilight/dusk) |
| standalone | data/eval/FishingBoat.mp4 | 1 RGB fishing boat clip |

---

## Current Work

See `task.md` for the tracking metrics evaluation task list. The goal is to move from visual inspection to automated MOT metrics (MOTA, IDF1, HOTA, ID switches) to objectively evaluate tracker configurations.
