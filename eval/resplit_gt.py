"""Re-split GT track IDs at long gaps, then re-score — fairer identity evaluation.

If a GT track disappears for more than `--threshold` frames and reappears with the
SAME id, the tracker can't reasonably keep that id (after max_age its track dies and
it mints a new id). Keeping one GT id across the gap charges the tracker an unfair
ID-switch / IDFN. This splits each GT track at gaps > threshold (post-gap segment
gets a fresh id), writes the re-split GT, and (optionally) re-scores a tracker MOT
dir against both the original and re-split GT to show the fairness delta.

NOTE: only IDENTITY metrics change (IDF1/IDsw/HOTA/AssA). Detection (Recall/
Precision/DetA) is id-agnostic and is unaffected — so this fairer-tracking eval does
not change the YOLO detection numbers.

Generate re-split GT:
    .venv/bin/python eval/resplit_gt.py --gt eval_videos/wavy-boats/labels \
        --threshold 180 --out eval_videos/wavy-boats/labels_resplit180

Generate + compare against a tracker run:
    ... --tracker results/robustness_all/yolo/clean/mot
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from eval_tracking import load_mot, evaluate_sequence_custom   # noqa: E402
from robustness_sweep import normalize_pred                    # noqa: E402

METRICS = ["Recall", "Precision", "IDF1", "IDsw", "Frag", "MOTA", "HOTA", "AssA"]


def resplit_file(in_path, out_path, threshold):
    rows = [l.rstrip("\n").split(",") for l in open(in_path) if l.strip()]
    by_id = defaultdict(list)                       # id -> [(frame, row_idx)]
    for i, p in enumerate(rows):
        by_id[int(float(p[1]))].append((int(float(p[0])), i))
    next_id = (max(int(float(p[1])) for p in rows) + 1) if rows else 1
    remap = {}
    n_splits = 0
    for tid, fl in by_id.items():
        fl.sort()
        cur, prev = tid, None
        for frame, idx in fl:
            if prev is not None and (frame - prev - 1) > threshold:
                cur = next_id
                next_id += 1
                n_splits += 1
            remap[idx] = cur
            prev = frame
    out = []
    for i, p in enumerate(rows):
        p = list(p)
        p[1] = str(remap[i])
        out.append(",".join(p))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(out) + "\n")
    return n_splits


def score(gt_dir, mot_dir):
    per = []
    for gt_file in sorted(Path(gt_dir).glob("*.txt")):
        pred = Path(mot_dir) / gt_file.name
        if not pred.exists():
            continue
        m, _ = evaluate_sequence_custom(load_mot(str(gt_file)),
                                        normalize_pred(load_mot(str(pred))), 0.5)
        per.append(m)
    n = len(per)
    return {k: (sum(p[k] for p in per) / n if n else float("nan")) for k in METRICS}, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default="eval_videos/wavy-boats/labels")
    ap.add_argument("--threshold", type=int, default=180,
                    help="split a GT track when a gap exceeds this many frames (default 180 = max_age)")
    ap.add_argument("--out", default=None, help="output re-split GT dir (default labels_resplit<thr>)")
    ap.add_argument("--tracker", default=None, help="tracker MOT dir to re-score (orig vs re-split GT)")
    args = ap.parse_args()

    gt = ROOT / args.gt
    out = Path(args.out) if args.out else gt.parent / f"{gt.name}_resplit{args.threshold}"

    total_splits = 0
    for f in sorted(gt.glob("*.txt")):
        total_splits += resplit_file(f, out / f.name, args.threshold)
    print(f"re-split GT at gaps > {args.threshold} frames: +{total_splits} new ids -> {out}")

    if args.tracker:
        orig, n = score(gt, ROOT / args.tracker)
        new, _ = score(out, ROOT / args.tracker)
        print(f"\nre-scoring {args.tracker} ({n} clips): original GT  ->  re-split GT")
        print(f"{'metric':<10} {'orig':>9} {'resplit':>9} {'delta':>9}")
        for k in METRICS:
            print(f"{k:<10} {orig[k]:>9.3f} {new[k]:>9.3f} {new[k]-orig[k]:>+9.3f}")


if __name__ == "__main__":
    main()
