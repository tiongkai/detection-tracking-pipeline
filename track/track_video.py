"""YOLO + HybridSORT tracking pipeline with EMA box smoothing.

Runs YOLO detection → HybridSORT multi-object tracking → per-track
exponential moving average on box coordinates to reduce jitter.
Outputs annotated MP4 videos, one per input clip.

Usage:
    conda run -n boat-tracker python pipeline/track/track_video.py \
        --weights results/yolo26l_split_v1_v4_merged/weights/best.pt \
        --source /path/to/clips \
        --out results/tracking_output \
        --conf 0.3 --iou 0.5 --ema-alpha 0.5
"""
import argparse
import glob
import hashlib
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from boxmot import HybridSort
from ultralytics import YOLO


def get_color(track_id: int) -> tuple:
    h = hashlib.md5(str(track_id).encode()).digest()
    return int(h[0]), int(h[1]), int(h[2])


class BoxSmoother:
    """Per-track exponential moving average on box coordinates."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.states = {}

    def smooth(self, track_id: int, box: np.ndarray) -> np.ndarray:
        if track_id not in self.states:
            self.states[track_id] = box.copy().astype(np.float64)
        else:
            self.states[track_id] = (
                self.alpha * box.astype(np.float64)
                + (1 - self.alpha) * self.states[track_id]
            )
        return self.states[track_id].copy()

    def prune(self, active_ids: set):
        self.states = {k: v for k, v in self.states.items() if k in active_ids}


def draw_tracks(frame, tracks, class_names, smoother):
    """Draw smoothed bounding boxes with track IDs onto the frame."""
    active_ids = set()
    for trk in tracks:
        x1, y1, x2, y2 = trk[:4]
        track_id = int(trk[4])
        conf = float(trk[5])
        cls_id = int(trk[6])

        active_ids.add(track_id)
        smoothed = smoother.smooth(track_id, np.array([x1, y1, x2, y2]))
        sx1, sy1, sx2, sy2 = [int(round(v)) for v in smoothed]

        color = get_color(track_id)
        cls_name = class_names.get(cls_id, str(cls_id))
        label = f"#{track_id} {cls_name} {conf:.2f}"

        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (sx1, sy1 - th - 6), (sx1 + tw, sy1), color, -1)
        cv2.putText(
            frame, label, (sx1, sy1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    smoother.prune(active_ids)


def process_clip(
    model, tracker, class_names, clip_path, out_path,
    conf=0.3, iou=0.5, ema_alpha=0.5,
):
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {clip_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    smoother = BoxSmoother(alpha=ema_alpha)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.predict(frame, conf=conf, iou=iou, verbose=False)
        r = results[0]

        dets = np.empty((0, 6), dtype=np.float32)
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy().reshape(-1, 1)
            clss = r.boxes.cls.cpu().numpy().reshape(-1, 1)
            dets = np.hstack([xyxy, confs, clss]).astype(np.float32)

        tracks = tracker.update(dets, frame)
        if len(tracks) > 0:
            draw_tracks(frame, tracks, class_names, smoother)

        writer.write(frame)

        if frame_idx % 100 == 0 or frame_idx == total:
            print(f"  {frame_idx}/{total} frames", flush=True)

    cap.release()
    writer.release()


def run(weights, source_dir, out_dir, conf=0.3, iou=0.5, ema_alpha=0.5, device="cuda:0"):
    model = YOLO(weights)
    class_names = model.names

    clips = sorted(glob.glob(os.path.join(source_dir, "*")))
    clips = [c for c in clips if Path(c).suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".ts"}]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(clips)} clips -> {out_dir}")
    print(f"Classes: {class_names}")
    print(f"Tracker: HybridSORT | EMA alpha: {ema_alpha} | conf: {conf} | device: {device}")

    for i, clip in enumerate(clips):
        name = Path(clip).stem
        out_mp4 = out_dir / f"{name}.mp4"
        if out_mp4.exists():
            print(f"[{i+1}/{len(clips)}] SKIP: {name}")
            continue

        tracker = HybridSort(
            reid_weights=Path("clip_veri.pt"),
            device=device,
            half=False,
            per_class=True,
            nr_classes=len(class_names),
            det_thresh=conf,
            max_age=180,
            min_hits=3,
            iou_threshold=0.15,
            use_custom_kf=True,
        )

        print(f"[{i+1}/{len(clips)}] {name}", flush=True)
        process_clip(model, tracker, class_names, clip, out_mp4, conf=conf, iou=iou, ema_alpha=ema_alpha)
        size_mb = out_mp4.stat().st_size / 1e6
        print(f"  -> {out_mp4.name} ({size_mb:.1f} MB)")

    print(f"Done. {len(list(out_dir.glob('*.mp4')))} mp4 files in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO + HybridSORT tracking with EMA box smoothing"
    )
    parser.add_argument("--weights", required=True, help="Path to YOLO .pt weights")
    parser.add_argument("--source", required=True, help="Directory of video clips")
    parser.add_argument("--out", required=True, help="Output directory for annotated videos")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--ema-alpha", type=float, default=1.0,
                        help="EMA smoothing factor (0=full history, 1=no smoothing)")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    args = parser.parse_args()
    run(args.weights, args.source, args.out,
        conf=args.conf, iou=args.iou, ema_alpha=args.ema_alpha, device=args.device)
