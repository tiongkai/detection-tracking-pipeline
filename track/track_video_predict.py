"""YOLO + HybridSORT tracking with Kalman prediction for missing detections.

Like track_video.py, but also outputs Kalman-predicted bounding boxes for
tracks that are coasting (no matching detection). Predicted boxes are drawn
with dashed outlines to distinguish them from detection-matched boxes.

Usage:
    conda run -n boat-tracker python pipeline/track/track_video_predict.py \
        --weights results/yolo26l_split_v7_original_classes/weights/best.pt \
        --source /path/to/clips \
        --out results/tracking_output \
        --conf 0.3 --iou 0.5 --ema-alpha 1.0
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
from boxmot.trackers.hybridsort.hybridsort import convert_x_to_bbox
from ultralytics import YOLO
from cross_modal_nms import cross_modal_nms


def get_color(track_id: int) -> tuple:
    h = hashlib.md5(str(track_id).encode()).digest()
    return int(h[0]), int(h[1]), int(h[2])


def draw_dashed_rect(frame, pt1, pt2, color, thickness=2, dash_length=10):
    x1, y1 = pt1
    x2, y2 = pt2
    for edge in [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]:
        (sx, sy), (ex, ey) = edge
        length = np.hypot(ex - sx, ey - sy)
        if length < 1:
            continue
        dx, dy = (ex - sx) / length, (ey - sy) / length
        pos = 0.0
        draw = True
        while pos < length:
            seg = min(dash_length, length - pos)
            px1 = int(sx + dx * pos)
            py1 = int(sy + dy * pos)
            px2 = int(sx + dx * (pos + seg))
            py2 = int(sy + dy * (pos + seg))
            if draw:
                cv2.line(frame, (px1, py1), (px2, py2), color, thickness)
            pos += seg
            draw = not draw


class BoxSmoother:
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


def get_all_active_tracks(tracker):
    """Get all active tracks across all classes (handles per_class mode)."""
    if tracker.per_class_active_tracks is not None:
        all_tracks = []
        for cls_id in tracker.per_class_active_tracks:
            all_tracks.extend(tracker.per_class_active_tracks[cls_id])
        return all_tracks
    return tracker.active_tracks


def get_coasting_tracks(tracker, matched_ids: set, class_names: dict, min_hits: int = 1, max_coast: int = 30, coast_cls_ids: set = None):
    """Extract Kalman-predicted boxes for tracks that are coasting (no detection match).
    If coast_cls_ids is set, only predict for those classes."""
    coasting = []
    for trk in get_all_active_tracks(tracker):
        tid = trk.id + 1
        if tid in matched_ids:
            continue
        if trk.hits < min_hits:
            continue
        if trk.time_since_update > max_coast:
            continue
        if coast_cls_ids is not None and int(trk.cls) not in coast_cls_ids:
            continue
        bbox = convert_x_to_bbox(trk.kf.x)[0][:4]
        coasting.append({
            "track_id": tid,
            "bbox": bbox,
            "cls": int(trk.cls),
            "conf": float(trk.conf),
            "age": trk.time_since_update,
        })
    return coasting


def draw_tracks(frame, tracks, coasting, class_names, smoother):
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

    for ct in coasting:
        track_id = ct["track_id"]
        active_ids.add(track_id)
        smoothed = smoother.smooth(track_id, ct["bbox"])
        sx1, sy1, sx2, sy2 = [int(round(v)) for v in smoothed]

        color = get_color(track_id)
        cls_name = class_names.get(ct["cls"], str(ct["cls"]))
        label = f"#{track_id} {cls_name} [predicted]"

        draw_dashed_rect(frame, (sx1, sy1), (sx2, sy2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (sx1, sy1 - th - 6), (sx1 + tw, sy1), color, -1)
        cv2.putText(
            frame, label, (sx1, sy1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    smoother.prune(active_ids)


def process_clip(
    model, tracker, class_names, clip_path, out_path,
    conf=0.3, iou=0.5, ema_alpha=0.5, det_interval=1, max_coast=30, coast_cls_ids=None,
    class_groups=None, nms_iou_thresh=0.5, mot_path=None,
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
    mot_lines = [] if mot_path else None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        run_det = (frame_idx % det_interval == 1) or (det_interval == 1)

        dets = np.empty((0, 6), dtype=np.float32)
        if run_det:
            results = model.predict(frame, conf=conf, iou=iou, verbose=False)
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy().reshape(-1, 1)
                clss = r.boxes.cls.cpu().numpy().reshape(-1, 1)
                dets = np.hstack([xyxy, confs, clss]).astype(np.float32)

        if class_groups and len(dets) > 0:
            dets = cross_modal_nms(dets, class_groups, nms_iou_thresh)

        tracks = tracker.update(dets, frame)

        matched_ids = set()
        if len(tracks) > 0:
            matched_ids = {int(t[4]) for t in tracks}

        coasting = get_coasting_tracks(tracker, matched_ids, class_names, max_coast=max_coast, coast_cls_ids=coast_cls_ids)

        if mot_lines is not None:
            if len(tracks) > 0:
                for trk in tracks:
                    x1, y1, x2, y2 = trk[:4]
                    tid = int(trk[4])
                    c = float(trk[5])
                    cls_id = int(trk[6])
                    mot_lines.append(
                        f"{frame_idx},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{c:.4f},{cls_id},1.00"
                    )
            for ct in coasting:
                bx1, by1, bx2, by2 = ct["bbox"]
                vis = max(0.1, 1.0 - ct["age"] / max_coast)
                mot_lines.append(
                    f"{frame_idx},{ct['track_id']},{bx1:.2f},{by1:.2f},{bx2-bx1:.2f},{by2-by1:.2f},"
                    f"{ct['conf']:.4f},{ct['cls']},{vis:.2f}"
                )

        draw_tracks(frame, tracks, coasting, class_names, smoother)

        writer.write(frame)

        if frame_idx % 100 == 0 or frame_idx == total:
            n_matched = len(tracks) if len(tracks) > 0 else 0
            n_coast = len(coasting)
            det_flag = "DET" if run_det else "KF"
            print(f"  {frame_idx}/{total} frames [{det_flag}] (det: {n_matched}, predicted: {n_coast})", flush=True)

    cap.release()
    writer.release()

    if mot_path and mot_lines:
        Path(mot_path).parent.mkdir(parents=True, exist_ok=True)
        Path(mot_path).write_text("\n".join(mot_lines) + "\n")


def build_class_groups(class_names):
    """Build cross-modal NMS groups for the 12-class domain-split taxonomy.
    Groups model classes that represent the same object type across RGB/thermal."""
    name_to_ids = {}
    for cls_id, name in class_names.items():
        base = name.replace("-rgb", "").replace("-thermal", "")
        name_to_ids.setdefault(base, set()).add(cls_id)
    return {base: ids for base, ids in name_to_ids.items() if len(ids) > 1}


def run(weights, source_dir, out_dir, conf=0.3, iou=0.5, ema_alpha=0.5, device="cuda:0", det_interval=1, max_coast=30, coast_classes=None, nms_iou_thresh=0.5, enable_nms=False, save_mot=False):
    model = YOLO(weights)
    class_names = model.names

    clips = sorted(glob.glob(os.path.join(source_dir, "*")))
    clips = [c for c in clips if Path(c).suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".ts"}]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coast_cls_ids = None
    if coast_classes:
        coast_cls_ids = {k for k, v in class_names.items() if any(c in v.lower() for c in coast_classes)}
        print(f"Kalman coast limited to: {[class_names[i] for i in sorted(coast_cls_ids)]}")

    class_groups = None
    if enable_nms:
        class_groups = build_class_groups(class_names)
        print(f"Cross-modal NMS enabled (iou_thresh={nms_iou_thresh}): {class_groups}")

    mot_dir = None
    if save_mot:
        mot_dir = out_dir / "mot"
        mot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(clips)} clips -> {out_dir}")
    print(f"Classes: {class_names}")
    print(f"Tracker: HybridSORT (with Kalman prediction) | EMA alpha: {ema_alpha} | conf: {conf} | det_interval: {det_interval} | max_coast: {max_coast} | device: {device}")
    if mot_dir:
        print(f"MOT output: {mot_dir}")

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
            min_hits=1 if det_interval > 1 else 3,
            iou_threshold=0.15,
            use_custom_kf=True,
            alpha=0.7,
            longterm_bank_length=150,
            longterm_reid_weight=0.25,
            with_longterm_reid=True,
            with_longterm_reid_correction=True,
            longterm_reid_correction_thresh=0.5,
            longterm_reid_correction_thresh_low=0.5,
        )

        mot_path = str(mot_dir / f"{name}.txt") if mot_dir else None

        print(f"[{i+1}/{len(clips)}] {name} (det every {det_interval} frame(s))", flush=True)
        process_clip(model, tracker, class_names, clip, out_mp4, conf=conf, iou=iou, ema_alpha=ema_alpha, det_interval=det_interval, max_coast=max_coast, coast_cls_ids=coast_cls_ids, class_groups=class_groups, nms_iou_thresh=nms_iou_thresh, mot_path=mot_path)
        size_mb = out_mp4.stat().st_size / 1e6
        print(f"  -> {out_mp4.name} ({size_mb:.1f} MB)")

    print(f"Done. {len(list(out_dir.glob('*.mp4')))} mp4 files in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO + HybridSORT tracking with Kalman prediction for coasting tracks"
    )
    parser.add_argument("--weights", required=True, help="Path to YOLO .pt weights")
    parser.add_argument("--source", required=True, help="Directory of video clips")
    parser.add_argument("--out", required=True, help="Output directory for annotated videos")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--ema-alpha", type=float, default=1.0,
                        help="EMA smoothing factor (0=full history, 1=no smoothing)")
    parser.add_argument("--det-interval", type=int, default=1,
                        help="Run detection every N frames (1=every frame, 2=every other frame)")
    parser.add_argument("--max-coast", type=int, default=10,
                        help="Max frames to show Kalman-predicted boxes before hiding (default 10 = ~0.3s at 30fps)")
    parser.add_argument("--coast-classes", nargs="*", default=None,
                        help="Only predict for these classes (substring match, e.g. 'boat'). Default: all classes")
    parser.add_argument("--enable-nms", action="store_true",
                        help="Enable cross-modal NMS (suppress duplicate detections across RGB/thermal class pairs)")
    parser.add_argument("--nms-iou-thresh", type=float, default=0.5,
                        help="IoU threshold for cross-modal NMS (default 0.5)")
    parser.add_argument("--save-mot", action="store_true",
                        help="Save MOTChallenge-format tracking output to <out>/mot/")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    args = parser.parse_args()
    run(args.weights, args.source, args.out,
        conf=args.conf, iou=args.iou, ema_alpha=args.ema_alpha, device=args.device,
        det_interval=args.det_interval, max_coast=args.max_coast, coast_classes=args.coast_classes,
        nms_iou_thresh=args.nms_iou_thresh, enable_nms=args.enable_nms, save_mot=args.save_mot)
