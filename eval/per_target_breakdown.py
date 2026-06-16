"""Per-target (per GT track) metric breakdown, for each video.

For each clip and each GT track id, shows how that specific target fared — which
targets fragment, swap, or get missed. Uses the same per-frame Hungarian matching
(IoU>=0.5, class-compatible) as the official evaluator, and the yolo-probe class
normalization (drop head/torso, thermal->rgb).

Per target:
  gt_frames  : length of the GT track (frames present)
  det_recall : fraction of its frames detected at all (matched to any pred id)
  n_ids      : distinct tracker IDs it was given
  dom_id     : the tracker ID covering most of it
  dom_frac   : dom_id frames / gt_frames  (per-target identity score; 1.0 = perfect)
  swaps      : ID switches on this target (same rule as IDsw)
  assoc      : sum of squared ID shares over matched frames (1.0 = single clean ID)
  dist       : {tracker_id: frames} fingerprint

    .venv/bin/python eval/per_target_breakdown.py \
        --gt eval_videos/wavy-boats/labels \
        --tracker results/robustness_all/yolo/clean/mot \
        --out results/robustness_all/per_target_yolo_clean.md
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from eval_tracking import load_mot, match_frames        # noqa: E402
from robustness_sweep import normalize_pred             # noqa: E402


def per_target(gt_data, pred_data, iou=0.5):
    gt_frames = Counter()
    for entries in gt_data.values():
        for e in entries:
            if e[5] > 0:
                gt_frames[e[0]] += 1

    matched = defaultdict(Counter)     # gt_id -> Counter(pred_id -> frames)
    swaps = Counter()
    prev = {}                          # gt_id -> last matched pred id (persists across gaps)
    for fr in match_frames(gt_data, pred_data, iou):     # frame-ordered
        for gid, pid, _ in fr["matches"]:
            matched[gid][pid] += 1
            if gid in prev and prev[gid] != pid:
                swaps[gid] += 1
            prev[gid] = pid

    rows = []
    for gid in sorted(gt_frames):
        gf = gt_frames[gid]
        dist = matched[gid]
        mtot = sum(dist.values())
        dom_id, dom_n = (dist.most_common(1)[0] if dist else (None, 0))
        assoc = sum((c / mtot) ** 2 for c in dist.values()) if mtot else 0.0
        rows.append({
            "gt_id": gid, "gt_frames": gf,
            "det_recall": mtot / gf if gf else 0.0,
            "n_ids": len(dist),
            "dom_id": dom_id, "dom_frac": dom_n / gf if gf else 0.0,
            "swaps": swaps[gid],
            "assoc": assoc,
            "dist": dict(dist.most_common(5)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default="eval_videos/wavy-boats/labels")
    ap.add_argument("--tracker", required=True, help="tracker MOT dir (<clip>.txt)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="markdown output path")
    args = ap.parse_args()

    md = [f"# Per-target breakdown\n\nGT: `{args.gt}`  |  tracker: `{args.tracker}`\n",
          "Columns: gt_id · gt_frames · det_recall · n_ids · dom_id · dom_frac · swaps · assoc · dist\n"]
    csv_rows = []
    for gt_file in sorted(Path(args.gt).glob("*.txt")):
        pred = Path(args.tracker) / gt_file.name
        if not pred.exists():
            continue
        rows = per_target(load_mot(str(gt_file)), normalize_pred(load_mot(str(pred))), args.iou)
        clip = gt_file.stem
        md.append(f"\n## {clip}\n")
        md.append("| gt_id | frames | det_recall | n_ids | dom_id | dom_frac | swaps | assoc | dist |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            md.append(f"| {r['gt_id']} | {r['gt_frames']} | {r['det_recall']:.3f} | {r['n_ids']} | "
                      f"{r['dom_id']} | {r['dom_frac']:.3f} | {r['swaps']} | {r['assoc']:.3f} | {r['dist']} |")
            csv_rows.append({"clip": clip, **{k: v for k, v in r.items() if k != "dist"}})
        # quick per-clip console line: how many targets are "clean" (1 id, no swap)
        clean = sum(1 for r in rows if r["n_ids"] == 1 and r["swaps"] == 0)
        print(f"{clip[:50]}: {len(rows)} targets, {clean} clean, "
              f"{sum(r['swaps'] for r in rows)} swaps")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(md) + "\n")
        # also a flat CSV next to it
        csv_path = Path(args.out).with_suffix(".csv")
        if csv_rows:
            head = ["clip", "gt_id", "gt_frames", "det_recall", "n_ids", "dom_id",
                    "dom_frac", "swaps", "assoc"]
            with open(csv_path, "w") as f:
                f.write(",".join(head) + "\n")
                for r in csv_rows:
                    f.write(",".join(str(r.get(c, "")) for c in head) + "\n")
        print(f"\n-> {args.out}  (+ {csv_path.name})")


if __name__ == "__main__":
    main()
