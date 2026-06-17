# Tracking-performance insights (gtdet probe)

The `gtdet` probe feeds **perfect GT boxes** to the tracker, so these numbers isolate the
**tracker + ReID** (detection removed as a variable). DetA stays ~0.969 everywhere by
construction. Consistent-GT IDs, ECC-off.

| condition | IDF1 | AssA | IDsw | Frag |
|---|---|---|---|---|
| clean | 0.853 | 0.798 | 9.2 | 3.5 |
| lowlight s2 | 0.853 | 0.800 | 8.9 | 3.7 |
| lowlight s4 | 0.835 | 0.769 | 14.0 | 3.6 |
| jpeg s2 | 0.850 | 0.797 | 9.4 | 3.6 |
| jpeg s4 | 0.858 | 0.803 | 11.3 | 3.7 |
| shake s2 | 0.843 | 0.786 | 10.5 | 4.3 |
| **shake s4** | 0.840 | 0.780 | **20.1** | **7.4** |
| **grayscale** | **0.824** | **0.757** | 11.8 | 3.8 |
| invert | 0.847 | 0.789 | 9.8 | 3.6 |
| (grayscale_invert) | pending | | | |

## 1. The tracker is the strong link
Given perfect boxes, IDF1 moves only 0.853 → 0.824 (worst) across *all* corruptions, vs the
detector (yolo IDF1 0.30–0.45). **Fragility is in detection, not tracking.**

## 2. Grayscale = the #1 appearance/ReID stressor
Lowest AssA (0.757) and IDF1 (0.824) — removing colour hurts ReID discrimination most.
Colour-invert is milder (AssA 0.789): it preserves structure ReID uses; grayscale strips
colour entirely. (grayscale_invert expected near grayscale.)

## 3. Shake = the #1 association-instability driver
IDsw 9→**20** and Frag 3.5→**7.4** at s4, even with perfect boxes — the camera jolt breaks
the motion/IoU gate → momentary switches. AssA only dips to 0.78 because most switches
recover (swaps that don't dominate hit IDsw/Frag, not IDF1). This is the motion-model
failure the SAM3/appearance-re-acq plan targets.

## 4. Intrinsic tracker ceiling (clean, perfect boxes)
- **63% of targets are clean** (1 id, 0 swaps); **37% get multiple ids**; **494 swaps / 246 targets**.
- The boat tracks cleanly; the residual is **similar-looking humans crossing** — a pure
  ReID/association limit better detection won't fix. This is the case for best-frame ReID / SAM3.

## Takeaways
- Tracker/ReID is robust to image corruption when given boxes — invest in **detection** for
  corruption robustness.
- For *identity* gains, the levers are **ReID under monochrome** (grayscale) and
  **association under shake** (motion model breaks) — appearance-based re-acquisition.
