# Best-frame vs FIFO ReID gallery — across augmented conditions (yolo)

Both ECC-off, OSNet-TRT. Delta = best - fifo. Positive IDF1/HOTA/AssA = best-frame helps.

| condition | IDF1 fifo | IDF1 best | dIDF1 | dHOTA | dAssA | dIDsw | dRecall |
|---|---|---|---|---|---|---|---|
| clean | 0.453 | 0.449 | -0.004 | -0.002 | -0.004 | +0.2 | -0.000 |
| lowlight_s2 | 0.433 | 0.428 | -0.005 | -0.002 | -0.004 | +0.2 | +0.000 |
| lowlight_s4 | 0.298 | 0.296 | -0.002 | +0.000 | +0.004 | -0.8 | -0.000 |
| jpeg_s2 | 0.449 | 0.448 | -0.001 | -0.001 | -0.002 | -0.1 | -0.000 |
| jpeg_s4 | 0.406 | 0.402 | -0.004 | -0.002 | -0.004 | +0.2 | -0.000 |
| shake_s2 | 0.457 | 0.447 | -0.010 | -0.008 | -0.019 | +1.0 | +0.000 |
| shake_s4 | 0.422 | 0.418 | -0.004 | -0.004 | -0.010 | -1.0 | -0.000 |
| grayscale_s1 | 0.443 | 0.438 | -0.005 | -0.004 | -0.010 | -0.3 | -0.000 |
| invert_s1 | 0.395 | 0.395 | -0.000 | +0.000 | +0.000 | -0.3 | +0.000 |
| grayscale_invert_s1 | 0.421 | 0.419 | -0.002 | -0.002 | -0.004 | +0.2 | +0.000 |
