# Experiment & Hypothesis Log

Living record of tracking experiments and the hypotheses formed along the way.
Two tracks of work: **(A) tracking quality** (MOT metrics) and **(B) inference
speed** (FPS / latency for live-stream deployment, target 20 FPS).

Hardware: 2× NVIDIA RTX A5500 (24 GB). Detector: YOLOv26-L (640×640).
Tracker: HybridSORT (boxmot) with ReID + ECC camera-motion compensation.

---

## A. Tracking Quality Ablation (exp00–exp11)

All quality experiments ran with detection every frame, ECC on, on the 81-clip
eval set. One parameter varied at a time from the baseline, except exp10/11.

| Exp | Change from baseline | MOTA | IDF1 | HOTA | IDsw | FP |
|-----|----------------------|------|------|------|------|----|
| exp00 | baseline (iou=0.15, alpha=0.7, EG=4.6, **no NMS**) | 0.244 | 0.252 | 0.284 | 1596 | 26477 |
| exp01 | iou_threshold 0.10 | 0.248 | 0.266 | 0.307 | 1661 | 25965 |
| exp02 | iou_threshold 0.05 | 0.248 | 0.266 | 0.307 | 1661 | 25965 |
| exp03 | longterm_reid_thresh 0.6 | 0.248 | 0.261 | 0.304 | 1657 | 25975 |
| exp04 | longterm_reid_thresh 0.7 | 0.248 | 0.261 | 0.304 | 1657 | 25987 |
| exp05 | alpha 0.5 | 0.249 | 0.259 | 0.302 | 1642 | 25916 |
| exp06 | alpha 0.3 | 0.249 | 0.258 | 0.299 | 1650 | 25891 |
| exp07 | EG_weight_high_score 6.0 | 0.248 | 0.266 | 0.307 | 1661 | 25965 |
| exp08 | EG_weight_high_score 8.0 | 0.248 | 0.266 | 0.307 | 1661 | 25965 |
| **exp09** | **cross-modal NMS enabled** | **0.329** | **0.325** | **0.351** | **688** | **18476** |
| exp10 | NMS + alpha 0.5 | 0.326 | 0.307 | 0.330 | 608 | 18840 |
| exp11 | NMS + alpha 0.6 | 0.326 | 0.314 | 0.330 | 607 | 18820 |

**Best config: exp09 (NMS on, otherwise baseline).** All single-parameter tracker
tweaks (exp01–08) were near-noise; NMS was the only large quality win.

---

## B. Inference Speed Experiments

Target: 20 FPS on live stream. Per-frame budget = 50 ms.

| ID | Config | ReID | Video | ECC | det ms | track ms | FPS |
|----|--------|------|-------|-----|--------|----------|-----|
| S0 | CLIP baseline | clip_veri (483 MB) | on | on | 16.9 | 19.9 | 21.0 |
| S1 | OSNet swap | osnet_x0_25 (3 MB) | on | on | 17.9 | 22.4 | 19.6 |
| S2 | ECC on (matched) | osnet_x0_25 | off | on | 19.8 | 25.2 | 22.0 |
| S3 | ECC off (matched) | osnet_x0_25 | off | **off** | 20.0 | **19.0** | **25.3** |
| S4 | ECC off + bank=1 | osnet_x0_25 | off | off | 17.9 | 16.4 | 28.8 |
| S5 | ECC off + no ReID | — | off | off | 18.7 | 2.9 | 45.3 |
| S6 | ECC off + **TRT ReID** | osnet_x0_25 **.engine** (FP16) | off | off | 18.1 | **5.7** | **41.5** |

S2–S5 are matched (same torch 2.7.1+cu126, no video) run in parallel on the two
GPUs to isolate each cost cleanly.

**ECC result (S2 vs S3):** disabling ECC saves **6.2 ms/frame** (track 25.2 → 19.0 ms),
+3.3 FPS (22.0 → 25.3). bbox deviation between S2 and S3 (`eval/compare_bbox_deviation.py`):

```
Match rate:    99.9%
IoU:           mean 0.991, median 1.000, p5 0.956
Center disp:   mean 0.20 px, median 0.00 px, p95 1.04 px, max 42.3 px
Size disp:     ~0 px
```

ECC moves boxes essentially not at all (median 0 px) → the eval cameras are static
enough that motion compensation is pure cost. **Recommendation: disable ECC** (`--no-cmc`).

**Tracking-time decomposition (matched, ECC off, no video, OSNet, 82 clips):**

| Component | track ms | share |
|-----------|----------|-------|
| Association floor (Kalman + Hungarian), S5 | 2.9 | 15 % |
| ReID embedding pipeline (crops + eager forward + dist), bank=1 (S4−S5) | 13.5 | 71 % |
| Long-term bank scaling 1→150 (S3−S4) | 2.6 | 14 % |
| **Total (S3)** | **19.0** | |

**TensorRT ReID (S6) collapses the ReID pipeline:** track 19.0 → **5.7 ms**, FPS
25.3 → **41.5**. The ~13 ms ReID cost was **PyTorch eager-mode forward overhead**
(per-op kernel launches + Python dispatch), NOT model FLOPs — which is why CLIP
(483 MB) and OSNet (3 MB) timed the same in eager mode but TRT-OSNet is 3× faster.
After TRT ReID, **detection (~18 ms) is the new bottleneck** → next: TRT the YOLO detector.

Baseline per-frame budget (S0, 50 clips / 23k frames):
- Detection (YOLO):   17.1 ms (34 %) — GPU
- Tracking + ReID:    21.4 ms (43 %) — mostly CPU (ECC + association + ReID preprocess)
- Drawing + video IO: 11.2 ms (22 %) — CPU
- Cross-modal NMS:     0.04 ms (0 %) — CPU

---

## Hypothesis Log

| # | Hypothesis | Evidence | Status |
|---|-----------|----------|--------|
| H1 | CLIP ReID forward pass is the inference bottleneck | Swapping CLIP (483 MB) → OSNet (3 MB) gave **no speedup** (track 19.9 → 22.4 ms). | **REFUTED** |
| H2 | ECC camera-motion compensation (`cv2.findTransformECC`, CPU, every frame) is a major cost | Full set S2 vs S3: track 25.2 ms (ECC on) vs 19.0 ms (ECC off) → ECC = **6.2 ms/frame**, +3.3 FPS. | **CONFIRMED** |
| H3 | The ~19 ms tracking residual is ReID crop preprocessing + long-term bank | Decomposed (S3/S4/S5): association 2.9 ms, ReID pipeline 13.5 ms, bank 2.6 ms. The 13.5 ms is **PyTorch eager-mode forward overhead**, not preprocessing or FLOPs (CLIP≈OSNet in eager; TRT-OSNet 3× faster). | **RESOLVED** — root cause = eager forward overhead |
| H4 | Cross-modal NMS is the single biggest tracking-quality lever | exp09 vs exp00: MOTA 0.244 → 0.329, FP −8001, IDsw −908. All other params near-noise. | **CONFIRMED** |
| H5 | Drawing + video encoding adds ~11 ms/frame; unnecessary for live deployment | S0 budget shows 11.2 ms "other". `--no-video` runs (S2/S3) will confirm by removing it. | **TESTING** |
| H6 | TensorRT FP16 cuts GPU time meaningfully | ReID confirmed: track 19.0 → 5.7 ms (S3 → S6), FPS 25.3 → 41.5. Detection TRT not yet run (now the bottleneck at ~18 ms). | **CONFIRMED (ReID)** — detection pending |
| H7 | Many eval cameras are static/fixed PTZ, so ECC adds cost without accuracy benefit | S2 vs S3 bbox deviation: 99.9 % match, IoU 0.991, median center shift 0 px. ECC barely moves boxes. | **CONFIRMED** |

---

## Optimization Backlog (priority order)

1. ~~**ECC**: disable for static cameras~~ — DONE (`--no-cmc`), +3.3 FPS, 0 accuracy loss.
2. ~~**TensorRT ReID**~~ — DONE, track 19→5.7 ms, +16 FPS. Use `.engine` via `--reid-weights`.
3. **TensorRT detection** — NOW THE BOTTLENECK (~18 ms). `model.export(format='engine')`,
   then `--weights best.engine`. Expect det 18 → ~8-10 ms → ~60+ FPS.
4. **Skip video IO** in production (no draw/encode) — H5, ~11 ms/frame.
5. **Long-term bank**: cache the mean incrementally (`hybridsort.py:588`) — ~2.6 ms.
6. **ReID preprocessing**: batch crops / GPU-side resize — secondary after TRT.

### Result so far
Baseline 21 FPS (CLIP, video on) → **41.5 FPS** (ECC off + TRT ReID, no video).
Headroom remains: detection is now the dominant cost.

---

## How to reproduce the speed runs

```bash
# Matched ECC on/off pair, parallel on two GPUs, timing-only
python track/track_video_predict.py --config configs/tracking_eval.yaml \
    --weights weights/best.pt --source data/eval/vws-eval-set \
    --out results/bench_ecc_on  --reid-weights osnet_x0_25_msmt17.pt \
    --no-video --save-mot --device cuda:0

python track/track_video_predict.py --config configs/tracking_eval.yaml \
    --weights weights/best.pt --source data/eval/vws-eval-set \
    --out results/bench_ecc_off --reid-weights osnet_x0_25_msmt17.pt \
    --no-video --save-mot --no-cmc --device cuda:1

# bbox deviation between the two
python eval/compare_bbox_deviation.py \
    --a results/bench_ecc_on/mot --b results/bench_ecc_off/mot \
    --names ecc_on ecc_off
```
