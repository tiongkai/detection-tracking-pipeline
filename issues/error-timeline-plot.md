# Error Timeline Plot for Tracking Evaluation

## Context

We have a tracking evaluation script (`eval/eval_tracking.py`) that computes MOT metrics (MOTA, IDF1, HOTA, ID switches, fragmentation, MT/ML) by comparing tracker output against ground truth. Both use MOTChallenge format:

```
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class_id>,<visibility>
```

The eval script currently outputs:
- A markdown summary table (overall + per-sequence metrics)
- A JSON file with all metrics

**Problem:** The numbers tell us *how well* the tracker performed, but not *where* or *when* errors happen. If a clip has 14 ID switches, we don't know if they all happen during a 2-second lighting transition or are spread across the whole clip. We need a per-frame error timeline to diagnose problems and guide parameter tuning.

## Goal

Add a `--plot` flag to `eval/eval_tracking.py` that generates a per-clip error timeline plot (PNG) showing frame-by-frame tracking errors. These plots will be saved alongside the existing report files when `-o` is specified.

## What you're building on

The core function you'll use is `match_frames(gt_data, pred_data, iou_thresh)` in `eval/eval_tracking.py`. It returns a list of per-frame result dicts:

```python
[
    {
        "frame": 1,
        "matches": [(gt_id, pred_id, iou), ...],  # successful matches
        "fp_ids": [pred_id, ...],                  # false positives (no GT match)
        "fn_ids": [gt_id, ...],                    # false negatives (missed GT)
        "n_gt": 4,                                 # total GT objects this frame
        "n_pred": 5,                               # total predictions this frame
    },
    ...
]
```

ID switches are computed in `compute_clear()` by checking if a GT object's matched prediction ID changed from the previous frame. You'll need to replicate this logic to get per-frame ID switch counts (currently it's only accumulated as a total).

## Tasks

### 1. Understand the data flow (~0.5 day)

Read `eval/eval_tracking.py` end to end. Trace the path from `load_mot()` → `match_frames()` → `compute_clear()` → `evaluate_sequence()` → `evaluate_tracker()`. Understand:

- What `frame_results` contains (the list returned by `match_frames`)
- How `compute_clear()` detects ID switches (lines ~190-200: compare `prev_match[gt_id]` vs current `pred_id`)
- How `evaluate_tracker()` merges sequences with frame offsets

Run the eval script on dummy data to see the output format:

```bash
# Create minimal test data to verify your understanding
# GT: 2 objects tracked for 10 frames
# Tracker: same 2 objects but with 1 ID switch at frame 5
```

### 2. Extract per-frame error counts (~1 day)

Write a function `compute_per_frame_errors(gt_data, pred_data, iou_thresh)` that returns a list of dicts, one per frame:

```python
[
    {
        "frame": 1,
        "tp": 3,        # matched GT-prediction pairs
        "fp": 1,        # predictions with no GT match
        "fn": 0,        # GT objects with no prediction match
        "idsw": 0,      # ID switches this frame (GT matched to different pred than prev frame)
    },
    ...
]
```

This reuses `match_frames()` output but adds per-frame ID switch detection. The ID switch logic from `compute_clear()` already does this — you just need to record `idsw` per frame instead of accumulating a total.

**Tip:** Don't duplicate `match_frames`. Call it once, then iterate over `frame_results` to compute per-frame TP/FP/FN/IDsw. The TP/FP/FN are already there (`len(matches)`, `len(fp_ids)`, `len(fn_ids)`). You only need to add the ID switch tracking with `prev_match`.

### 3. Generate the timeline plot (~1.5 days)

Write a function `plot_error_timeline(per_frame_errors, clip_name, out_path)` that generates a PNG with:

**Layout: single figure, 2 subplots stacked vertically, shared x-axis (frame number)**

**Top subplot — Object counts:**
- Line: number of GT objects per frame (shows when objects enter/exit)
- Line: number of matched TPs per frame
- Shaded area between them (gap = detection failures)

**Bottom subplot — Error events:**
- Bar: FP count per frame (orange)
- Bar: FN count per frame (red)
- Markers: ID switch events (triangle markers on the x-axis at frames where switches occur, sized by count)

**Formatting:**
- Title: clip name
- X-axis: frame number
- Legends for both subplots
- Figsize ~(14, 6) so it's wide enough to read frame-level detail
- Save as PNG at 150 DPI

Use matplotlib. It's already available in the `boat-tracker` conda environment (it's a dependency of boxmot/ultralytics).

### 4. Wire into CLI and report pipeline (~0.5 day)

- Add `--plot` flag to the argparser (store_true, default False)
- When `--plot` is set and `-o` is specified, generate one PNG per clip per tracker config
- Save to `<out_dir>/plots/<tracker_name>_<clip_name>.png`
- Print the path to each generated plot

The integration point is in `evaluate_tracker()` or in `main()` after evaluation is done. You have access to `gt_data` and `pred_data` per sequence in the `for seq_name, gt_file, tracker_file in sequences` loop.

### 5. Test on real data (~0.5 day)

Once GT annotations exist (separate task), run:

```bash
conda run -n boat-tracker python eval/eval_tracking.py \
    --gt data/eval/gt \
    --tracker results/tracker/mot/baseline \
    --plot \
    -o results/tracker/eval
```

Until then, create synthetic test data (a short MOT file with known errors at known frames) and verify:
- FP/FN bars appear at the correct frames
- ID switch markers appear at the correct frames
- The GT object count line matches expected entries/exits
- Plot is readable and not cluttered

## Output structure

```
results/tracker/eval/
├── tracking_report.md
├── tracking_metrics_baseline.json
└── plots/
    ├── baseline_clip_001.png
    └── baseline_clip_002.png
```

## Dependencies

- matplotlib (already in boat-tracker env)
- numpy (already in boat-tracker env)
- No new dependencies required

## Time estimate

| Subtask | Days |
|---------|------|
| Understand data flow | 0.5 |
| Per-frame error extraction | 1 |
| Timeline plot generation | 1.5 |
| CLI integration | 0.5 |
| Testing | 0.5 |
| **Total** | **~4 days** |

## Reference

- `eval/eval_tracking.py` — the script you're modifying
- `eval/sample_tracking_metrics.json` — schema for the JSON output
- `CLAUDE.md` — full repo documentation, class definitions, MOT format spec
- `task.md` — broader project context (tracking metrics evaluation)
