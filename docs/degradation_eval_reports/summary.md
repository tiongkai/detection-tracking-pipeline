# Corruption-robustness sweep

Probes: ['gtdet', 'yolo'] | kinds: ['lowlight', 'jpeg', 'shake', 'grayscale', 'invert'] | severities: [0, 2, 4] (0 = clean) | ReID: osnet_x0_25_msmt17.engine

## How to read this report

**Two probes (the core design — they isolate different failure modes):**
- **`gtdet`** — ground-truth boxes are fed to the tracker *as detections* (the detector is
  bypassed). This isolates **tracking / ReID** ability: detection is perfect (Recall ~1.0 by
  construction), so any error is pure identity/association. Best-frame gallery is meaningless
  here (uniform confidence).
- **`yolo`** — the real YOLOv26 detector runs on the (possibly degraded) frame. This measures
  end-to-end / **detector** robustness.

**Severity:** 0 = clean (no corruption); higher = stronger. `grayscale`/`invert` are binary
(reported at a single level). Camera **shake** also warps the GT boxes by the same transform.

**Detector class handling (yolo only):** before scoring, `head`/`torso` predictions are
dropped and thermal classes are mapped to their RGB base, so detections match the
boat/human/motor GT. (Without this, head+torso are ~26k structural false positives that crush
precision.)

**Config:** ECC camera-motion compensation is **OFF** (matches the real-time TRT deployment;
verified to not change accuracy on these near-static-camera clips). ReID = OSNet (TensorRT).
GT is **partially labelled** (not every boat/person is annotated) — this is why *Recall* is the
trustworthy headline and *Precision/MOTA* must be read with caution.

**Metrics:**
- **Recall** = TP / (TP+FN): fraction of GT boxes found. **Most trustworthy here** — extra
  detections can't lower it, so partial GT doesn't bias it. Headline for the detector.
- **Precision** = TP / (TP+FP): *depressed by partial GT* (real-but-unlabelled boats count as
  false positives). Treat as soft, not comparable to literature.
- **IDF1** = identity F1 via a **global 1-to-1 match** of each GT track to the predicted ID it
  overlaps most across the whole clip. Measures how consistently the *correct* identity is held.
  A track that swaps ID once but keeps the new ID for most of the clip is mostly counted
  **correct** — only the shorter (pre-swap) segment is penalized. Headline for identity.
- **IDsw** = ID switches: counted **once** each time a GT track's matched prediction ID changes
  between consecutive frames. A sustained swap = 1 (not re-counted per frame). High on `yolo`
  mostly reflects **detection dropouts** (object buried by a wave → track dies → reappears with
  a new ID), *not* tracker association — compare with the much lower `gtdet` IDsw.
- **MOTA** = 1 − (FN+FP+IDsw)/GT: overall error rate; can go **negative** when FP is high
  (inflated here by partial GT). Soft metric.
- **HOTA** = geometric mean of detection accuracy (DetA) and association accuracy (AssA); a
  balanced single number, less sensitive to the FP/partial-GT issue than MOTA.
- **Frag** = track fragmentations (times a GT track's coverage is interrupted).

**Bottom line:** trust **Recall** (detection robustness) and **IDF1 / HOTA** (identity
robustness); read **Precision / MOTA** as soft (partial GT); **IDsw** counts events and on
`yolo` is dominated by detection dropouts, not association errors.



## Probe: gtdet  (metrics: IDF1, MOTA, IDsw, HOTA)


### lowlight

| severity | IDF1 | MOTA | IDsw | HOTA |
|---|---|---|---|---|
| 0 | 0.853 | 0.967 | 9.219 | 0.875 |
| 2 | 0.853 | 0.968 | 8.906 | 0.876 |
| 4 | 0.835 | 0.965 | 14.000 | 0.858 |

### jpeg

| severity | IDF1 | MOTA | IDsw | HOTA |
|---|---|---|---|---|
| 0 | 0.853 | 0.967 | 9.219 | 0.875 |
| 2 | 0.850 | 0.967 | 9.375 | 0.874 |
| 4 | 0.858 | 0.967 | 11.281 | 0.878 |

### shake

| severity | IDF1 | MOTA | IDsw | HOTA |
|---|---|---|---|---|
| 0 | 0.853 | 0.967 | 9.219 | 0.875 |
| 2 | 0.843 | 0.967 | 10.469 | 0.868 |
| 4 | 0.840 | 0.963 | 20.094 | 0.864 |

### grayscale

| severity | IDF1 | MOTA | IDsw | HOTA |
|---|---|---|---|---|
| 0 | 0.853 | 0.967 | 9.219 | 0.875 |
| 1 | 0.824 | 0.966 | 11.750 | 0.851 |

### invert

| severity | IDF1 | MOTA | IDsw | HOTA |
|---|---|---|---|---|
| 0 | 0.853 | 0.967 | 9.219 | 0.875 |
| 1 | 0.847 | 0.967 | 9.812 | 0.870 |

## Probe: yolo  (metrics: Recall, Precision)


### lowlight

| severity | Recall | Precision |
|---|---|---|
| 0 | 0.669 | 0.591 |
| 2 | 0.676 | 0.561 |
| 4 | 0.516 | 0.481 |

### jpeg

| severity | Recall | Precision |
|---|---|---|
| 0 | 0.669 | 0.591 |
| 2 | 0.667 | 0.588 |
| 4 | 0.642 | 0.564 |

### shake

| severity | Recall | Precision |
|---|---|---|
| 0 | 0.669 | 0.591 |
| 2 | 0.676 | 0.569 |
| 4 | 0.651 | 0.536 |

### grayscale

| severity | Recall | Precision |
|---|---|---|
| 0 | 0.669 | 0.591 |
| 1 | 0.636 | 0.598 |

### invert

| severity | Recall | Precision |
|---|---|---|
| 0 | 0.669 | 0.591 |
| 1 | 0.605 | 0.593 |
