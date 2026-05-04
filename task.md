# Task 2 — Tracking Metrics

## Objective

Use relevant metrics to evaluate tracker performance. These metrics allow objective, consistent comparison across tracker configurations (e.g. tuning `iou_threshold`, `alpha`, `longterm_reid_correction_thresh`) and validate whether changes actually improve tracking quality.

Currently we evaluate tracking by visual inspection of output videos. This doesn't scale and can't catch subtle regressions. We need automated metrics.

---

## Tasks

### 2.1 — Background reading (~2 days)

- [ ] **Read up on object detection and tracking**

  Gain background on detection (YOLO, DETR architectures — understand anchor-based vs anchor-free, how NMS works, confidence thresholds) and tracking (SORT → DeepSORT → ByteTrack → HybridSORT progression). This builds intuition for labelling, augmentation, and understanding why tracker parameters exist. We use **HybridSORT** from the `boxmot` library because we need ReID capability.

  Key things to understand:
  - How Kalman filter predicts bounding boxes when detection misses
  - What ReID embeddings are and how cosine similarity matching works
  - What `per_class` tracking means and why cross-modal NMS is needed (see `track.md`)
  - The three association stages in HybridSORT (high-score → BYTE → last-observation fallback)

  Read: `track.md` (parameter reference), `tracker_eval.md` (run history), boxmot repo docs

- [ ] **Identify relevant tracking metrics**

  Read MOT benchmark papers and understand what each metric measures. Focus on which ones are relevant to our use case (maritime surveillance with few objects, long occlusions, cross-domain appearance changes).

  **Core metrics to understand:**

  | Metric | What it measures | Why we care |
  |--------|-----------------|-------------|
  | **MOTA** (Multi-Object Tracking Accuracy) | Combined FP, FN, and ID switch rate | Overall tracking quality — our primary metric |
  | **IDF1** (ID F1 Score) | How consistently the correct ID is maintained over a track's lifetime | Directly measures the track fragmentation problem we're seeing |
  | **HOTA** (Higher Order Tracking Accuracy) | Balanced combination of detection and association quality | Newer metric that separates detection errors from association errors — tells us if problems are from the detector or the tracker |
  | **ID Switches** | Number of times a track ID changes on the same ground-truth object | Directly counts our fragmentation issue |
  | **Fragmentation** | Number of times a ground-truth track is interrupted (gaps) | Measures how often the detector loses an object temporarily |
  | **MT/ML** (Mostly Tracked / Mostly Lost) | % of GT tracks tracked for >80% / <20% of their lifetime | Tells us if we're maintaining tracks for the full duration |

  Deliverable: short writeup (1 page) on which metrics we'll use and why, with references to papers.

  Suggested reading:
  - MOT Challenge evaluation methodology
  - HOTA paper (Luiten et al., 2021)
  - TrackEval repo documentation

  *~1 day*

---

### 2.2 — Ground truth annotation for tracking eval (~5 days)

- [ ] **Create ground-truth tracking annotations for eval clips**

  Metrics need ground truth. Select 3–5 representative clips from `data/eval/` covering different scenarios:

  | Scenario | Suggested clip | Why |
  |----------|---------------|-----|
  | Clear RGB, single boat | `FishingBoat.mp4` | Baseline — should be easy |
  | Thermal, multiple boats | 1 x pt80 dusk clip (023399 series) | Cross-modal NMS territory |
  | Poor lighting, occlusion | 1 x `clips/` dahua segment | Tests ReID through occlusion |
  | Fast camera motion | 1 x `clips/` segment with pan | Tests KF prediction quality |

  For each clip, annotate in MOTChallenge format:
  ```
  <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<visibility>
  ```

  Use CVAT or similar annotation tool with video tracking mode.

  Deliverable: GT files in `data/eval/gt/` in MOTChallenge format, one per clip.

  *~5 days (annotation is slow — scope to 3 clips and keep each under 30s)*

---

### 2.3 — Tracking output in MOTChallenge format (~1 day)

- [ ] **Modify `track_video_predict.py` to output MOTChallenge-format text alongside the video**

  Currently the script only writes annotated mp4 video. Add a `--save-mot` flag that also writes a `.txt` file per clip in MOTChallenge format:

  ```
  <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<visibility>
  ```

  Include both detection-matched tracks and Kalman-predicted tracks (mark predicted with visibility < 1.0 or a flag). This is the input for TrackEval.

  Deliverable: `--save-mot` flag producing `.txt` files that TrackEval can ingest.

  *~1 day*

---

### 2.4 — Evaluation script using TrackEval (~3 days)

- [ ] **Integrate TrackEval for automated metric computation**

  Use the [TrackEval](https://github.com/JonathonLuiten/TrackEval) library (or `py-motmetrics`) to compute MOTA, IDF1, HOTA, ID switches, fragmentation from the MOT-format outputs against ground truth.

  Create a script `eval/eval_tracking.py` that:
  1. Takes a directory of tracker output `.txt` files and a GT directory
  2. Runs TrackEval and outputs a metrics table
  3. Supports `--compare` mode to show two configs side-by-side

  Deliverable: `eval/eval_tracking.py` that produces a table like:
  ```
  | Config           | MOTA | IDF1 | HOTA | IDsw | Frag |
  |------------------|------|------|------|------|------|
  | baseline         | 0.72 | 0.65 | 0.58 | 14   | 23   |
  | lower_iou_thresh | 0.75 | 0.71 | 0.62 | 8    | 19   |
  ```

  Tip: run TrackEval on a standard MOT benchmark example first to learn the expected directory structure and naming conventions before wiring up custom data.

  *~3 days (1 day learning TrackEval on example data + 1 day setup + 1 day testing/debugging)*

---

### 2.5 — Baseline measurement + parameter tuning experiments (~3 days)

- [ ] **Run baseline metrics on current tracker config**

  Run the eval script on the annotated clips with the current settings from `track.md`. Record as the baseline.

  *~0.5 day*

- [ ] **Experiment: `iou_threshold` tuning**

  Run with `iou_threshold` = 0.15 (current), 0.10, 0.05. Compare MOTA, IDF1, and ID switches. This tests whether KF drift after coasting is causing track fragmentation.

  *~0.5 day*

- [ ] **Experiment: ReID correction threshold tuning**

  Run with `longterm_reid_correction_thresh` = 0.5 (current), 0.6, 0.7. This tests whether the post-match filter is rejecting valid associations after appearance changes.

  *~0.5 day*

- [ ] **Experiment: `alpha` (ReID EMA) tuning**

  Run with `alpha` = 0.7 (current), 0.5, 0.3. This tests whether the short-term appearance feature is adapting fast enough to appearance changes.

  *~0.5 day*

- [ ] **Experiment: Interclass NMS impact**

  Run with and without `--enable-nms` on thermal/twilight clips. Compare ID switches — NMS should reduce duplicate track IDs on the same object.

  *~0.5 day*

- [ ] **Write up findings**

  Document results in a table comparing all configs. Recommend the best config with justification based on metrics.

  Deliverable: updated `tracker_eval.md` with metrics tables for each experiment and a recommended config.

  *~0.5 day*

---

## Time Estimate Summary

| Task | Days |
|------|------|
| 2.1 Background reading + metric selection | 2 |
| 2.2 Ground truth annotation | 5 |
| 2.3 MOTChallenge output format | 1 |
| 2.4 TrackEval integration script | 3 |
| 2.5 Baseline + 4 experiments + writeup | 3 |
| **Total** | **~14 working days** |

Tasks 2.1 and 2.2 are best done sequentially — background reading gives context for consistent annotation decisions. Tasks 2.3 and 2.4 are sequential. Task 2.5 depends on all prior tasks.
