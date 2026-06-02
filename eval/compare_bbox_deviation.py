"""Compare bounding-box deviation between two tracker MOT output directories.

Use case: quantify how much tracked boxes shift when a setting changes
(e.g. ECC camera-motion compensation on vs off). Track IDs are not assumed
to be consistent between runs — boxes are matched spatially per frame by
maximum IoU, then positional deviation is measured on matched pairs.

Metrics per matched box pair:
    - IoU between the two boxes
    - center displacement (pixels)
    - size difference (width/height delta)

Also reports unmatched boxes (present in one run but not the other), which
indicate where the two configs produced different track counts.

Usage:
    python eval/compare_bbox_deviation.py \
        --a results/bench_ecc_on/mot \
        --b results/bench_ecc_off/mot \
        --names ecc_on ecc_off \
        --iou-thresh 0.3
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_mot(path):
    """Load MOT file -> {frame: [(x, y, w, h), ...]}."""
    frames = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            frame = int(p[0])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            frames[frame].append((x, y, w, h))
    return frames


def iou_matrix(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    iy1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def center(box):
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def compare_sequence(mot_a, mot_b, iou_thresh):
    a = load_mot(mot_a)
    b = load_mot(mot_b)
    all_frames = sorted(set(a) | set(b))

    ious, center_disp, size_disp = [], [], []
    n_a = n_b = n_matched = 0

    for fr in all_frames:
        boxes_a = a.get(fr, [])
        boxes_b = b.get(fr, [])
        n_a += len(boxes_a)
        n_b += len(boxes_b)
        if not boxes_a or not boxes_b:
            continue
        iou = iou_matrix(boxes_a, boxes_b)
        ri, ci = linear_sum_assignment(-iou)
        for r, c in zip(ri, ci):
            if iou[r, c] >= iou_thresh:
                n_matched += 1
                ious.append(iou[r, c])
                cax, cay = center(boxes_a[r])
                cbx, cby = center(boxes_b[c])
                center_disp.append(np.hypot(cax - cbx, cay - cby))
                size_disp.append(abs(boxes_a[r][2] - boxes_b[c][2]) +
                                 abs(boxes_a[r][3] - boxes_b[c][3]))

    return {
        "n_a": n_a, "n_b": n_b, "n_matched": n_matched,
        "ious": ious, "center_disp": center_disp, "size_disp": size_disp,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare bbox deviation between two MOT output dirs.")
    parser.add_argument("--a", required=True, help="First MOT directory (reference)")
    parser.add_argument("--b", required=True, help="Second MOT directory (comparison)")
    parser.add_argument("--names", nargs=2, default=["A", "B"], help="Display names")
    parser.add_argument("--iou-thresh", type=float, default=0.3,
                        help="Min IoU to consider two boxes a matched pair")
    args = parser.parse_args()

    dir_a, dir_b = Path(args.a), Path(args.b)
    seqs = sorted(f.stem for f in dir_a.glob("*.txt"))

    all_ious, all_center, all_size = [], [], []
    tot_a = tot_b = tot_matched = 0

    print(f"Comparing {args.names[0]} (A) vs {args.names[1]} (B), IoU thresh={args.iou_thresh}\n")
    print(f"{'Sequence':<55} {'boxes_A':>8} {'boxes_B':>8} {'matched':>8} {'meanIoU':>8} {'ctrDev':>8}")
    print("-" * 100)

    for seq in seqs:
        mot_b = dir_b / f"{seq}.txt"
        if not mot_b.exists():
            continue
        r = compare_sequence(dir_a / f"{seq}.txt", mot_b, args.iou_thresh)
        all_ious += r["ious"]; all_center += r["center_disp"]; all_size += r["size_disp"]
        tot_a += r["n_a"]; tot_b += r["n_b"]; tot_matched += r["n_matched"]
        miou = np.mean(r["ious"]) if r["ious"] else 0
        mctr = np.mean(r["center_disp"]) if r["center_disp"] else 0
        print(f"{seq[:54]:<55} {r['n_a']:>8} {r['n_b']:>8} {r['n_matched']:>8} {miou:>8.3f} {mctr:>8.2f}")

    print("\n" + "=" * 70)
    print("OVERALL BBOX DEVIATION")
    print("=" * 70)
    print(f"Total boxes:   A={tot_a}  B={tot_b}  matched={tot_matched}")
    if tot_a and tot_b:
        print(f"Match rate:    {tot_matched/max(tot_a,tot_b)*100:.1f}% "
              f"(unmatched boxes indicate track-count differences)")
    if all_ious:
        print(f"\nMatched-pair deviation:")
        print(f"  IoU:              mean={np.mean(all_ious):.3f}  median={np.median(all_ious):.3f}  p5={np.percentile(all_ious,5):.3f}")
        print(f"  Center disp (px): mean={np.mean(all_center):.2f}  median={np.median(all_center):.2f}  p95={np.percentile(all_center,95):.2f}  max={np.max(all_center):.1f}")
        print(f"  Size disp (px):   mean={np.mean(all_size):.2f}  median={np.median(all_size):.2f}  p95={np.percentile(all_size,95):.2f}")
        print(f"\nInterpretation: IoU~1.0 and center disp~0 means the two configs")
        print(f"produce nearly identical boxes. Larger deviation = the change moved boxes.")


if __name__ == "__main__":
    main()
