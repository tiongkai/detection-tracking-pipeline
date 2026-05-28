# compare_by_metadata.py

Compares tracker experiment results against the baseline (`exp00_baseline`) for a filtered subset of clips, using the clip metadata in `data/eval/vws_eval_set_metadata.json`.

---

## What it does

1. Filters clips from the evaluation set using metadata fields (domain, lighting, cross_modal, etc.)
2. Loads per-sequence metrics from the baseline and your chosen experiments
3. Computes deltas and averages across the filtered clips
4. Saves a markdown report and JSON file to an output directory

---

## Requirements

- Experiments must have been evaluated first (`eval/eval_tracking.py` with `-o`)
- Eval output must exist at `results/experiments/<exp_name>/eval/tracking_metrics_mot.json`
- Metadata file must exist at `data/eval/vws_eval_set_metadata.json`

---

## Usage

### Interactive mode (recommended for first use)

`conda run` blocks stdin, so interactive mode requires activating the environment first:

```bash
conda activate boat-tracker
python eval/compare_by_metadata.py
```

The script will guide you through three steps:

**Step 1 — Pick metadata fields to filter on:**
```
Available filter fields:
  1. domain
  2. lighting
  3. cross_modal
  4. multi_object
  5. camera
Select (comma-separated numbers, or Enter to skip): 1,2
```

**Step 2 — Pick values for each selected field:**
```
Filter: domain
  Available values: both, rgb, thermal
  Select (comma-separated numbers, or Enter to skip): 3

Filter: lighting
  Available values: dark, light
  Select (comma-separated numbers, or Enter to skip): 1
```

**Step 3 — Pick experiments to compare against baseline:**
```
Available experiments:
  1. exp01_iou_0.10
  2. exp02_iou_0.05
  ...
Select (comma-separated numbers, or Enter to skip): 9,10
```

Then confirm or change the output directory.

---

### Non-interactive mode (for repeating known comparisons)

```bash
conda run -n boat-tracker python eval/compare_by_metadata.py --non-interactive \
    --filter domain=thermal lighting=dark \
    --experiments exp09_nms exp10_nms_alpha0.5 \
    --out results/analysis/thermal_dark
```

Multiple `--filter` values for the same field are combined with OR. Multiple fields are combined with AND. For example:

```bash
# thermal OR both domain, AND dark lighting, AND cross_modal=true
--filter domain=thermal domain=both lighting=dark cross_modal=true
```

---

## Metadata fields

| Field | Values | Description |
|-------|--------|-------------|
| `domain` | `rgb`, `thermal`, `both` | Whether the clip uses RGB, thermal, or mixed annotations |
| `lighting` | `light`, `dark` | Lighting condition of the clip |
| `cross_modal` | `true`, `false` | Whether the clip has both RGB and thermal class annotations (best clips for testing cross-modal NMS) |
| `multi_object` | `true`, `false` | Whether the clip contains more than one annotated boat |
| `camera` | `pt80_023395`, `Cam_05`, `Media_11`, etc. | Source camera identifier |

---

## Output

Two files are saved, named after the experiments compared:

```
<out_dir>/
├── report_<exp1>_vs_<exp2>.md        # human-readable markdown tables
└── comparison_<exp1>_vs_<exp2>.json  # raw numbers for programmatic use
```

### Markdown report structure

- One table per clip showing baseline, experiment value, and delta (Δ) for each metric
- A final averages table across all matched clips
- ↑ marks improvements, ↓ marks regressions

### Metrics reported

| Metric | Higher is better | Description |
|--------|-----------------|-------------|
| MOTA | ✓ | Multi-Object Tracking Accuracy |
| IDF1 | ✓ | ID F1 — measures ID consistency |
| HOTA | ✓ | Higher Order Tracking Accuracy |
| DetA | ✓ | Detection component of HOTA |
| AssA | ✓ | Association component of HOTA |
| Precision | ✓ | TP / (TP + FP) |
| Recall | ✓ | TP / (TP + FN) |
| IDsw | ✗ | ID switches |
| FP | ✗ | False positives |
| FN | ✗ | False negatives |
| Frag | ✗ | Track fragmentations |

---

## Examples

**Compare NMS experiments on cross-modal clips only:**
```bash
conda run -n boat-tracker python eval/compare_by_metadata.py --non-interactive \
    --filter cross_modal=true \
    --experiments exp09_nms exp10_nms_alpha0.5 \
    --out results/analysis/cross_modal
```

**Compare alpha experiments on thermal dark clips:**
```bash
conda run -n boat-tracker python eval/compare_by_metadata.py --non-interactive \
    --filter domain=thermal lighting=dark \
    --experiments exp05_alpha_0.5 exp06_alpha_0.3 \
    --out results/analysis/thermal_dark_alpha
```

**Compare all experiments on multi-object clips:**
```bash
conda run -n boat-tracker python eval/compare_by_metadata.py --non-interactive \
    --filter multi_object=true \
    --experiments exp01_iou_0.10 exp02_iou_0.05 exp03_ltreid_0.6 exp04_ltreid_0.7 \
        exp05_alpha_0.5 exp06_alpha_0.3 exp07_eg_6.0 exp08_eg_8.0 \
        exp09_nms exp10_nms_alpha0.5 exp11_nms_alpha0.6 \
    --out results/analysis/multi_object_all
```
