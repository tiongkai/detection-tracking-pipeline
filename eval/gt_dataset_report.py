"""Ground-truth dataset breakdown — per video and overall.

For each clip: #frames, #tracks, #boxes, and per-class track/box counts (plus track
length stats and how many tracks have gaps). Helps document the eval set.

    .venv/bin/python eval/gt_dataset_report.py \
        --gt eval_videos/wavy-boats/labels \
        --out eval_videos/wavy-boats/gt_dataset_report.md
"""
import argparse
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = {0: "boat", 1: "vessel", 2: "human", 3: "motor", 4: "head", 5: "torso",
               6: "boat-th", 7: "vessel-th", 8: "human-th", 9: "motor-th",
               10: "head-th", 11: "torso-th"}


def analyze(path):
    # id -> list of (frame, class)
    tracks = defaultdict(list)
    boxes_per_cls = defaultdict(int)
    max_frame = 0
    for l in open(path):
        if not l.strip():
            continue
        p = l.split(",")
        frame = int(float(p[0])); tid = int(float(p[1])); cls = int(float(p[7]))
        tracks[tid].append((frame, cls))
        boxes_per_cls[cls] += 1
        max_frame = max(max_frame, frame)

    tracks_per_cls = defaultdict(int)
    lengths = []
    gapped = 0
    for tid, items in tracks.items():
        # dominant class for this track
        cc = defaultdict(int)
        for _, c in items:
            cc[c] += 1
        dom = max(cc, key=cc.get)
        tracks_per_cls[dom] += 1
        frs = sorted(f for f, _ in items)
        lengths.append(len(set(frs)))
        if any(b - a - 1 > 0 for a, b in zip(frs, frs[1:])):
            gapped += 1

    return {
        "frames": max_frame,
        "n_tracks": len(tracks),
        "n_boxes": sum(boxes_per_cls.values()),
        "classes": sorted(boxes_per_cls),
        "tracks_per_cls": dict(tracks_per_cls),
        "boxes_per_cls": dict(boxes_per_cls),
        "gapped": gapped,
        "len_min": min(lengths) if lengths else 0,
        "len_med": sorted(lengths)[len(lengths)//2] if lengths else 0,
        "len_max": max(lengths) if lengths else 0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default="eval_videos/wavy-boats/labels")
    ap.add_argument("--out", default="eval_videos/wavy-boats/gt_dataset_report.md")
    args = ap.parse_args()

    files = sorted(Path(args.gt).glob("*.txt"))
    stats = {f.stem: analyze(f) for f in files}

    # which classes appear anywhere (for stable columns)
    cls_seen = sorted({c for s in stats.values() for c in s["classes"]})
    cname = lambda c: CLASS_NAMES.get(c, str(c))

    hdr = (["video", "frames", "tracks", "boxes", "n_classes"]
           + [f"{cname(c)}_trk" for c in cls_seen]
           + [f"{cname(c)}_box" for c in cls_seen]
           + ["gapped_trk", "len(min/med/max)"])
    md = ["# GT dataset breakdown\n",
          f"{len(files)} videos · classes present: {', '.join(cname(c) for c in cls_seen)}\n",
          "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]

    tot = defaultdict(int)
    tot_trk_cls = defaultdict(int); tot_box_cls = defaultdict(int)
    for name in sorted(stats):
        s = stats[name]
        row = [name[:42], s["frames"], s["n_tracks"], s["n_boxes"], len(s["classes"])]
        row += [s["tracks_per_cls"].get(c, 0) for c in cls_seen]
        row += [s["boxes_per_cls"].get(c, 0) for c in cls_seen]
        row += [s["gapped"], f"{s['len_min']}/{s['len_med']}/{s['len_max']}"]
        md.append("| " + " | ".join(str(x) for x in row) + " |")
        tot["frames"] += s["frames"]; tot["tracks"] += s["n_tracks"]; tot["boxes"] += s["n_boxes"]
        tot["gapped"] += s["gapped"]
        for c in cls_seen:
            tot_trk_cls[c] += s["tracks_per_cls"].get(c, 0)
            tot_box_cls[c] += s["boxes_per_cls"].get(c, 0)

    total_row = (["**TOTAL**", tot["frames"], tot["tracks"], tot["boxes"], ""]
                 + [tot_trk_cls[c] for c in cls_seen]
                 + [tot_box_cls[c] for c in cls_seen]
                 + [tot["gapped"], ""])
    md.append("| " + " | ".join(str(x) for x in total_row) + " |")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md) + "\n")
    print(f"{len(files)} videos -> {args.out}")
    print(f"TOTAL: {tot['tracks']} tracks, {tot['boxes']} boxes across {tot['frames']} frames")
    for c in cls_seen:
        print(f"  {cname(c)}: {tot_trk_cls[c]} tracks, {tot_box_cls[c]} boxes")


if __name__ == "__main__":
    main()
