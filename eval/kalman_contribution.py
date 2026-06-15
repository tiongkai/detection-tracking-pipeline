"""Quantify how many missed detections the Kalman filter (coasting) recovers.

track_video_predict.py writes two kinds of boxes to its MOT output:
  - detection-matched tracks  -> visibility column == 1.00
  - Kalman-predicted (coasting) tracks, emitted when the detector missed an object
    but the track is still alive -> visibility < 1.00 (decays with coast age)

So we can read the tracker MOT and, for every GT box the tracker recovers (a true
positive at IoU>=thr), attribute it to either a real detection or a Kalman
prediction. Kalman-attributed TPs are exactly the misses the KF recovered.

Reports per clip and overall:
  - TP_det / TP_kf      : true positives from detection vs from KF coasting
  - FP_kf               : KF-predicted boxes with no GT (the precision cost)
  - recall_det_only     : recall if KF boxes were discarded (TP_det / GT)
  - recall_with_kf      : recall including KF boxes
  - kf_recall_gain (pp) : recall_with_kf - recall_det_only  (the KF's contribution)

Single tracker dir:
    .venv/bin/python eval/kalman_contribution.py --gt <labels> --tracker <mot_dir>

Across a robustness sweep (yolo probe, KF gain vs corruption):
    .venv/bin/python eval/kalman_contribution.py --sweep results/robustness_all \
        --labels eval_videos/wavy-boats/labels --aug-root eval_videos/wavy-boats/aug_all
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from eval_tracking import load_mot, classes_compatible   # noqa: E402
from robustness_sweep import normalize_pred              # noqa: E402

VIS_DET = 0.999          # visibility >= this => detection-matched; below => Kalman-predicted


def _xyxy(e):
    return (e[1], e[2], e[1] + e[3], e[2] + e[4])


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def analyze(gt_data, pred_data, iou_thr=0.5):
    """Per-frame greedy IoU matching (class-aware); attribute TPs to det vs KF."""
    c = dict(TP_det=0, TP_kf=0, FP_det=0, FP_kf=0, FN=0)
    for f in set(gt_data) | set(pred_data):
        gts, prs = gt_data.get(f, []), pred_data.get(f, [])
        pairs = []
        for gi, g in enumerate(gts):
            for pi, p in enumerate(prs):
                if classes_compatible(g[6], p[6]):
                    i = _iou(_xyxy(g), _xyxy(p))
                    if i >= iou_thr:
                        pairs.append((i, gi, pi))
        pairs.sort(reverse=True)
        gm, pm = set(), set()
        for _, gi, pi in pairs:
            if gi in gm or pi in pm:
                continue
            gm.add(gi); pm.add(pi)
            c["TP_det" if prs[pi][7] >= VIS_DET else "TP_kf"] += 1
        for pi, p in enumerate(prs):
            if pi not in pm:
                c["FP_det" if p[7] >= VIS_DET else "FP_kf"] += 1
        c["FN"] += len(gts) - len(gm)
    return c


def summarize(c):
    gt = c["TP_det"] + c["TP_kf"] + c["FN"]
    tp = c["TP_det"] + c["TP_kf"]
    return {
        **c,
        "GT": gt,
        "recall_det_only": c["TP_det"] / gt if gt else 0.0,
        "recall_with_kf": tp / gt if gt else 0.0,
        "kf_recall_gain": c["TP_kf"] / gt if gt else 0.0,
    }


def score_dir(gt_dir, tracker_dir, iou_thr=0.5):
    total = dict(TP_det=0, TP_kf=0, FP_det=0, FP_kf=0, FN=0)
    n = 0
    for gt_file in sorted(Path(gt_dir).glob("*.txt")):
        pred_file = Path(tracker_dir) / gt_file.name
        if not pred_file.exists():
            continue
        gt_data = load_mot(str(gt_file))
        pred_data = normalize_pred(load_mot(str(pred_file)))
        c = analyze(gt_data, pred_data, iou_thr)
        for k in total:
            total[k] += c[k]
        n += 1
    return summarize(total), n


def _fmt(s):
    return (f"GT {s['GT']:6d} | TP_det {s['TP_det']:6d} | TP_kf {s['TP_kf']:5d} | "
            f"FP_kf {s['FP_kf']:5d} | recall det-only {s['recall_det_only']:.3f} "
            f"-> +KF {s['recall_with_kf']:.3f}  (KF gain +{s['kf_recall_gain']*100:.2f} pp)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", help="GT labels dir (flat <clip>.txt)")
    ap.add_argument("--tracker", help="tracker MOT dir (<clip>.txt)")
    ap.add_argument("--iou", type=float, default=0.5)
    # sweep mode
    ap.add_argument("--sweep", help="robustness sweep out root (uses yolo probe)")
    ap.add_argument("--labels", help="clean GT labels dir (for sweep mode)")
    ap.add_argument("--aug-root", help="aug root holding <cond>/labels (for sweep mode)")
    ap.add_argument("--probe", default="yolo")
    args = ap.parse_args()

    if args.sweep:
        root = Path(args.sweep) / args.probe
        conds = sorted(p.name for p in root.iterdir() if (p / "mot").exists())
        rows = []
        for cond in conds:
            gtd = args.labels if cond == "clean" else str(Path(args.aug_root) / cond / "labels")
            s, n = score_dir(gtd, root / cond / "mot", args.iou)
            rows.append((cond, s))
            print(f"[{cond:12s}] ({n} clips) {_fmt(s)}")
        out = Path(args.sweep) / f"kalman_contribution_{args.probe}.md"
        md = [f"# Kalman-filter miss-recovery ({args.probe} probe)\n",
              "| condition | GT | TP_det | TP_kf | FP_kf | recall det-only | recall +KF | KF gain (pp) |",
              "|---|---|---|---|---|---|---|---|"]
        for cond, s in rows:
            md.append(f"| {cond} | {s['GT']} | {s['TP_det']} | {s['TP_kf']} | {s['FP_kf']} | "
                      f"{s['recall_det_only']:.3f} | {s['recall_with_kf']:.3f} | "
                      f"+{s['kf_recall_gain']*100:.2f} |")
        out.write_text("\n".join(md) + "\n")
        print(f"\n-> {out}")
    else:
        if not (args.gt and args.tracker):
            ap.error("provide --gt and --tracker, or --sweep/--labels/--aug-root")
        s, n = score_dir(args.gt, args.tracker, args.iou)
        print(f"({n} clips) {_fmt(s)}")


if __name__ == "__main__":
    main()
