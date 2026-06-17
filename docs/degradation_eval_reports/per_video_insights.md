# Insights from the per-video results

Mined from `metrics.csv` / `per_video_report.md` (consistent-GT IDs, ECC-off).

## 1. The eval set is wildly heterogeneous
- `yolo` clean Recall: **0.37 → 0.92** (median 0.66) — detection ranges from near-perfect to <half.
- `gtdet` clean IDF1: **0.63 → 1.00** (median 0.86) — tracking difficulty varies too.
- Aggregate means hide this; per-video is the right granularity.

## 2. Detection is the dominant bottleneck, and it's scene-dependent
`gtdet − yolo` IDF1 gap (identity headroom unlocked by perfect detection): mean **+0.40**,
max **+0.72** (POWERBOAT), min +0.13 (BAD DECISIONS). On hard clips, fixing detection ≈
doubles IDF1.

## 3. Difficulty clusters by source video
- Easiest: **BAD DECISIONS** (recall up to 0.92, smallest gap).
- Hardest: **POWERBOAT** and **THEY REALIZED** (low recall, big gap, most fragile) — more
  distant/small boats + heavier chop.

## 4. Crowding mildly hurts tracking, not detection
- corr(#tracks, yolo recall) = **+0.31** (busier clips ≠ worse detection).
- corr(#tracks, gtdet IDF1) = **−0.21** (more objects → harder to keep IDs distinct — swaps).
- #tracks/clip: 3–16 (median 7).

## 5. Low-light is the universal threat
- Worst corruption in **27/32** videos.
- `lowlight_s4` drops recall on **every** clip (median −0.15, worst −0.38, none immune).

## 6. Invert occasionally out-hurts low-light
On 5 clips (POWERBOAT, BOAT BURIED, THEY REALIZED) colour-invert drops recall more than
`lowlight_s4` — scenes that lean on colour cues. (`grayscale_invert` case added separately.)

## Takeaways
- **Prioritize low-light detection robustness** — it's universal and dominant.
- **Report per-video, not just aggregate** — the set spans easy→very-hard.
- **Hard clips are detection-bound** — the tracker is fine once given boxes; the +0.40–0.72
  IDF1 gap is detection headroom, so detector improvements pay off most on the hard sources.
