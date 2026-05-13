# Detection & Tracking Pipeline

Detection and tracking pipeline for maritime surveillance across RGB and thermal domains.

## Git practices

Practices are not as strict here as in S&S standards, but we are still practicing trunk-based development; i.e. **create a new branch each time you're working on a new task**. Other than that, try to keep your work to informative branches e.g. `dev/fq-evaluator` or `fix/fr-scoring` and try to have informative commit messages e.g. `chore: copying over files` or `fix: updating typos and versioning` or `dev: implemented new scoring method`.

## Model

**Architecture:** YOLOv26-L (Ultralytics, ~25M params)
**Weights:** `weights/best.pt`
**Input:** 640x640px
**Training:** 150 epochs, batch 8, multi-scale, pretrained backbone

### Classes (12)

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
| 10 | head-thermal | Thermal (synthetic only) |
| 11 | torso-thermal | Thermal (synthetic only) |

Domain is encoded in the class name. The model outputs separate classes for RGB and thermal appearances of the same object type. Cross-modal NMS (see below) suppresses duplicates when both fire on the same object.

### Validation Performance (epoch 150)

| Metric | Value |
|--------|-------|
| Precision | 0.843 |
| Recall | 0.702 |
| mAP50 | 0.790 |
| mAP50-95 | 0.519 |

### Training Data

**Split:** `split_v7_original_classes` — 12-class domain-split taxonomy

| Split | Images |
|-------|--------|
| Train | 23,000 |
| Val | 1,099 |

**Train composition:** 4,458 original images expanded to 23,000 via offline augmentation.

| Domain | Originals | Augmented | Total |
|--------|-----------|-----------|-------|
| RGB | 3,521 | 17,605 (5 variants each) | 21,126 |
| Thermal | 937 | 937 (1 variant each) | 1,874 |

**Data sources (original images):**

| Source | Images | Domain | Notes |
|--------|--------|--------|-------|
| youtube | 1,553 | RGB | Web-scraped maritime footage |
| willow | 655 | Thermal | Thermal camera (test-like domain) |
| nas-ptz | 555 | RGB | PTZ camera footage |
| dahua | 421 | RGB | Poor-lighting RGB (dusk/night) |
| google search | 402 | RGB | Web images |
| xiaohongshu | 351 | RGB | Social media images |
| st | 282 | Thermal | Thermal camera (test-like domain) |
| lars | 163 | RGB | LARS maritime dataset (outboard motors) |
| phone footage | 54 | RGB | Handheld phone footage |
| waterscenes | 22 | RGB | Waterscenes dataset (outboard motors) |

### Offline Augmentation

RGB images get 5 augmented variants each:

| Variant | Description |
|---------|-------------|
| `aug_gray` | Grayscale (simulates white-hot thermal polarity) |
| `aug_gray_inv` | Inverted grayscale (black-hot polarity) |
| `aug_hot` | COLORMAP_HOT false-colour thermal appearance |
| `aug_drop` | Channel dropout (simulates alternate spectral response) |
| `aug_dark` | Brightness/noise/blur/gamma degradation (poor lighting) |

Thermal images get 1 augmented variant:

| Variant | Description |
|---------|-------------|
| `aug_inv` | Polarity flip (`cv2.bitwise_not`) — white-hot ↔ black-hot |

Augmented RGB images that simulate thermal appearance get thermal class labels (e.g. `boat-rgb` → `boat-thermal`) via `thermal_remap`.

### Online Augmentation (during training)

**Poor lighting degradation** (p=0.7):
- `RandomBrightnessContrast(brightness_limit=(-0.5, -0.1))` — dark scenes
- `GaussNoise(std_range=(0.01, 0.05))` — sensor noise
- `MotionBlur(blur_limit=(3, 15), p=0.5)` — camera movement
- `RandomGamma(gamma_limit=(30, 80))` — dark gamma

**YOLO built-in augmentation:**
- Mosaic: 1.0, Mixup: 0.1, Degrees: 5.0, Scale: 0.5, Fliplr: 0.5

### Split Strategy

- **Video-level grouping**: all frames from the same video clip are kept together (prevents temporal leakage)
- **Stratified split**: iterative multi-label stratification at video group level preserves class balance
- **80:20 ratio** for train/val
- **No test set in training split** — test evaluation uses a separate held-out COCO JSON with st, willow, and dahua sources

---

## Directory Structure

```
detection-tracking-pipeline/
├── weights/
│   └── best.pt                     # YOLOv26-L trained weights
├── track/
│   ├── track_video.py              # Standard HybridSORT tracking
│   ├── track_video_predict.py      # Tracking + Kalman prediction + interclass NMS
│   └── cross_modal_nms.py          # Cross-modal NMS module
├── detection/
│   └── detect.py                   # Detection inference (yolo, rfdetr, sahi)
├── annotation_tools/
│   ├── correct_tracks.py          # MOT track correction tool (OpenCV-based)
│   └── README.md                  # Annotation tool usage guide
├── eval/
│   ├── eval.py                     # Detection evaluation (metrics from predictions JSON)
│   ├── eval_tracking.py            # Tracking evaluation (HOTA, MOTA, IDF1, IDsw, Frag, MT/ML)
│   ├── sample_tracking_metrics.json # Schema for tracking eval JSON output
│   ├── cross_modal_nms.py          # Cross-modal NMS module (eval copy)
│   ├── compare_models.py           # Cross-experiment comparison
│   ├── image_metrics.py            # Per-image quality metrics
│   ├── failure_analysis.py         # Failure analysis + visualisations
│   └── compute_split_metrics.py    # Split-level metrics
├── configs/                        # Split + experiment YAML configs
├── track.md                        # HybridSORT parameter reference + troubleshooting
├── tracker_eval.md                 # Tracker run history + eval data
└── CHANGELOG.md                    # Cross-modal NMS changelog
```

## Quick Start

### Tracking with Kalman prediction + interclass NMS

```bash
conda run -n boat-tracker python track/track_video_predict.py \
    --weights weights/best.pt \
    --source /path/to/clips \
    --out /path/to/output \
    --conf 0.3 --iou 0.5 --ema-alpha 1.0 \
    --max-coast 10 --coast-classes boat \
    --enable-nms --nms-iou-thresh 0.5
```

### Detection Evaluation

Two-step process: generate detections, then evaluate against ground truth.

**Step 1 — Run inference**

```bash
# Standard YOLO
conda run -n obj-det python detection/detect.py \
    --weights weights/best.pt \
    --gt-json data/splits/split_v7/coco/test/_annotations.coco.json \
    --method yolo \
    --out results/detections/yolo_baseline.json

# SAHI (sliced inference — better for small objects)
conda run -n obj-det python detection/detect.py \
    --weights weights/best.pt \
    --gt-json data/splits/split_v7/coco/test/_annotations.coco.json \
    --method sahi --slice-size 640 --overlap-ratio 0.2 \
    --out results/detections/sahi_640.json
```

**Step 2 — Evaluate**

```bash
conda run -n obj-det python eval/eval.py \
    --predictions results/detections/yolo_baseline.json \
    --gt-json data/splits/split_v7/coco/test/_annotations.coco.json \
    --manifest data/splits/split_v7/split_manifest.csv \
    -o results/eval/yolo_baseline
```

Legacy mode (inference + eval in one pass) is still supported via `--config`.

### Tracking Evaluation

Two-step process: generate MOTChallenge-format tracker output, then evaluate against ground truth.

**Step 1 — Generate MOT output**

```bash
conda run -n boat-tracker python track/track_video_predict.py \
    --weights weights/best.pt \
    --source /path/to/eval/clips \
    --out results/tracker/baseline \
    --conf 0.3 --iou 0.5 --ema-alpha 1.0 \
    --max-coast 10 --coast-classes boat \
    --enable-nms --nms-iou-thresh 0.5 \
    --save-mot
```

**Step 2 — Evaluate against ground truth**

```bash
# Single config
conda run -n boat-tracker python eval/eval_tracking.py \
    --gt data/eval/gt \
    --tracker results/tracker/baseline/mot \
    -o results/tracker/eval

# Compare two configs side-by-side
conda run -n boat-tracker python eval/eval_tracking.py \
    --gt data/eval/gt \
    --tracker results/tracker/mot/baseline results/tracker/mot/tuned \
    --names baseline tuned \
    -o results/tracker/eval
```

**Output structure:**

```
results/tracker/
├── baseline/
│   ├── clip_001.mp4                   # annotated video
│   ├── clip_002.mp4
│   └── mot/                           # MOTChallenge-format tracker output (--save-mot)
│       ├── clip_001.txt
│       └── clip_002.txt
└── eval/                              # evaluation output (-o)
    ├── tracking_report.md             # markdown summary table
    └── tracking_metrics_baseline.json # per-sequence + overall metrics
```

**Ground truth directory layout** (required for step 2):

```
data/eval/gt/
├── clip_001/
│   └── gt.txt
└── clip_002/
    └── gt.txt
```

GT files use the same 9-column MOTChallenge format. See `eval/eval_tracking.py` header or `CLAUDE.md` for column definitions.

To create GT annotations from tracker output, use the correction tool — see [`annotation_tools/README.md`](annotation_tools/README.md) for the full workflow.

The JSON output schema is documented in `eval/sample_tracking_metrics.json`.

## Dependencies

| Environment | Packages |
|-------------|----------|
| `boat-tracker` | torch 2.7.1+cu126, ultralytics 8.4.3, boxmot 15.0.9 |
| `obj-det` | pycocotools, pandas, numpy, opencv-python |
