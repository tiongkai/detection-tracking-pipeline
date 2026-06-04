"""Fast recall/precision sweep over conf and max_coast.

Reuses the IoU matching from eval_tracking.py but computes only single-threshold
CLEAR-style counts (TP/FP/FN -> recall/precision) — skips the slow HOTA loop.

max_coast is applied as a post-filter on the visibility column: runs are produced
once at max_coast=60 (vis = 1 - age/60 for coasting boxes), so simulating a smaller
max_coast T means keeping coasting boxes with vis >= 1 - T/60. Detection-matched
boxes (vis = 1.00) are always kept.

Usage:
    python eval/sweep_recall.py --gt data/eval/gt/mot \
        --runs conf0.3=results/sweep_conf0.3/mot conf0.2=results/sweep_conf0.2/mot ... \
        --built-max-coast 60 --max-coasts 10 20 30 60 --iou 0.5
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_tracking import iou_matrix, classes_compatible


def load_mot_filtered(path, vis_min):
    """Load MOT, keeping detected boxes (vis==1.0) and coasting boxes with vis>=vis_min."""
    frames = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            vis = float(p[8]) if len(p) > 8 else 1.0
            if vis < 0.999 and vis < vis_min:
                continue  # coasting box beyond the simulated max_coast
            frame = int(p[0])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            cls = int(float(p[7])) if len(p) > 7 else 1
            frames[frame].append((x, y, w, h, cls))
    return frames


def load_gt(path):
    frames = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            conf = float(p[6]) if len(p) > 6 else 1.0
            if conf <= 0:
                continue
            frame = int(p[0])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            cls = int(float(p[7])) if len(p) > 7 else 1
            frames[frame].append((x, y, w, h, cls))
    return frames


def count_seq(gt, pred, iou_thresh):
    tp = fp = fn = 0
    for fr in sorted(set(gt) | set(pred)):
        g = gt.get(fr, [])
        d = pred.get(fr, [])
        if not g:
            fp += len(d); continue
        if not d:
            fn += len(g); continue
        gb = [(x, y, w, h) for x, y, w, h, _ in g]
        db = [(x, y, w, h) for x, y, w, h, _ in d]
        iou = iou_matrix(gb, db)
        cost = 1 - iou
        cost[iou < iou_thresh] = 1e6
        for r in range(len(g)):
            for c in range(len(d)):
                if not classes_compatible(g[r][4], d[c][4]):
                    cost[r, c] = 1e6
        ri, ci = linear_sum_assignment(cost)
        m = 0
        matched_d = set()
        for r, c in zip(ri, ci):
            if iou[r, c] >= iou_thresh:
                m += 1; matched_d.add(c)
        tp += m
        fn += len(g) - m
        fp += len(d) - m
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="name=mot_dir pairs, one per conf value")
    parser.add_argument("--built-max-coast", type=int, default=60,
                        help="max_coast the runs were produced with")
    parser.add_argument("--max-coasts", type=int, nargs="+", default=[10, 20, 30, 60])
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    gt_dir = Path(args.gt)
    gt_seqs = {f.parent.name: f for f in gt_dir.rglob("gt.txt")}

    runs = {}
    for r in args.runs:
        name, path = r.split("=", 1)
        runs[name] = Path(path)

    print(f"Recall sweep @ IoU {args.iou}  (rows = conf, cols = max_coast)\n")
    header = f"{'conf':<10}" + "".join(f"mc={mc:<8}" for mc in args.max_coasts)
    print(header)
    print("-" * len(header))

    grid = {}
    for cname, mot_dir in runs.items():
        row = f"{cname:<10}"
        for mc in args.max_coasts:
            vis_min = 1 - mc / args.built_max_coast
            TP = FP = FN = 0
            for seq, gtf in gt_seqs.items():
                pf = mot_dir / f"{seq}.txt"
                if not pf.exists():
                    continue
                gt = load_gt(gtf)
                pred = load_mot_filtered(pf, vis_min)
                tp, fp, fn = count_seq(gt, pred, args.iou)
                TP += tp; FP += fp; FN += fn
            recall = TP / (TP + FN) if (TP + FN) else 0
            prec = TP / (TP + FP) if (TP + FP) else 0
            grid[(cname, mc)] = (recall, prec, TP, FP, FN)
            row += f"{recall:<11.3f}"
        print(row)

    print("\nPrecision (same layout):")
    print(header)
    print("-" * len(header))
    for cname in runs:
        row = f"{cname:<10}"
        for mc in args.max_coasts:
            row += f"{grid[(cname,mc)][1]:<11.3f}"
        print(row)

    print("\nDetail (recall / precision / TP / FP / FN):")
    for (cname, mc), (rec, prec, tp, fp, fn) in grid.items():
        print(f"  conf={cname} mc={mc}: R={rec:.3f} P={prec:.3f} TP={tp} FP={fp} FN={fn}")


if __name__ == "__main__":
    main()
