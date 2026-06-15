#!/usr/bin/env python3
"""Render ground-truth bounding boxes with track IDs onto source videos."""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ID_COLORS = [
    (230, 25, 75),   (60, 180, 75),   (255, 225, 25),  (0, 130, 200),
    (245, 130, 48),  (145, 30, 180),  (70, 240, 240),  (240, 50, 230),
    (210, 245, 60),  (250, 190, 212), (0, 128, 128),   (220, 190, 255),
    (170, 110, 40),  (255, 250, 200), (128, 0, 0),     (170, 255, 195),
    (128, 128, 0),   (255, 215, 180), (0, 0, 128),     (128, 128, 128),
]


def color_for_id(track_id):
    return ID_COLORS[track_id % len(ID_COLORS)]


def load_gt(gt_path):
    """Load MOT-format gt.txt into {frame_number: [(id, x, y, w, h), ...]}."""
    annotations = defaultdict(list)
    with open(gt_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            frame = int(row[0])
            tid = int(row[1])
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            annotations[frame].append((tid, x, y, w, h))
    return annotations


def draw_boxes(frame, boxes, thickness=2):
    for tid, x, y, w, h in boxes:
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        color = color_for_id(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"ID:{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def process_sequence(video_path, gt_path, output_path):
    annotations = load_gt(gt_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        boxes = annotations.get(frame_idx, [])
        draw_boxes(frame, boxes)

        fnum_label = f"Frame {frame_idx}"
        cv2.putText(frame, fnum_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, fnum_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()

    n_ids = len({tid for boxes in annotations.values() for tid, *_ in boxes})
    print(f"  {frame_idx} frames, {n_ids} IDs, {len(annotations)} annotated frames -> {output_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", default="data/eval/gt/mot",
                        help="Directory containing GT sequence folders with gt.txt")
    parser.add_argument("--video-dir", default="data/eval/vws-eval-set",
                        help="Directory containing source videos")
    parser.add_argument("--output-dir", default="results/gt_visualized",
                        help="Output directory for rendered videos")
    parser.add_argument("--seq", default=None,
                        help="Process only this sequence name (omit to process all)")
    parser.add_argument("--flat", action="store_true",
                        help="Flat layout: gt-dir holds <seq>.txt files (matched to "
                             "<seq>.<ext> in video-dir) instead of <seq>/gt.txt folders")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    gt_dir = (Path(args.gt_dir) if Path(args.gt_dir).is_absolute() else base / args.gt_dir)
    video_dir = (Path(args.video_dir) if Path(args.video_dir).is_absolute() else base / args.video_dir)
    output_dir = (Path(args.output_dir) if Path(args.output_dir).is_absolute() else base / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.flat:
        sequences = sorted(p.stem for p in gt_dir.glob("*.txt"))
    else:
        sequences = sorted([
            d.name for d in gt_dir.iterdir()
            if d.is_dir() and (d / "gt.txt").exists()
        ])

    if args.seq:
        if args.seq not in sequences:
            print(f"Sequence '{args.seq}' not found in {gt_dir}")
            sys.exit(1)
        sequences = [args.seq]

    print(f"Processing {len(sequences)} sequences...")
    ok, fail = 0, 0
    for seq in sequences:
        gt_path = (gt_dir / f"{seq}.txt") if args.flat else (gt_dir / seq / "gt.txt")
        video_path = None
        for ext in (".mp4", ".mkv", ".avi"):
            candidate = video_dir / f"{seq}{ext}"
            if candidate.exists():
                video_path = candidate
                break
        if video_path is None:
            print(f"  SKIP {seq}: no matching video")
            fail += 1
            continue

        out_path = output_dir / f"{seq}_gt.mp4"
        print(f"[{ok + fail + 1}/{len(sequences)}] {seq}")
        if process_sequence(video_path, gt_path, out_path):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} succeeded, {fail} failed. Output in {output_dir}")


if __name__ == "__main__":
    main()
