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
import csv
import glob
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
# Vendored HybridSort with the best-frame ReID gallery feature.
# Defaults to "fifo" gallery, which reproduces stock boxmot HybridSort behavior.
from hybridsort_bestframe import HybridSort
from boxmot.trackers.hybridsort.hybridsort import convert_x_to_bbox
from ultralytics import YOLO
from cross_modal_nms import cross_modal_nms


class _NoCMC:
    """No-op camera-motion compensation: always returns an identity warp.

    Used to disable HybridSORT's ECC (cv2.findTransformECC) motion
    compensation, which runs on CPU every frame and is unnecessary for
    static/fixed cameras.
    """

    def apply(self, img, dets):
        return np.eye(3)


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


def load_gt_dets(path, conf_override=1.0):
    """Load a MOT-format file into {frame_idx: ndarray[N,6]} of [x1,y1,x2,y2,conf,cls].

    Used for GT-as-detections mode: ground-truth boxes are fed to the tracker in
    place of detector output, so tracking/ReID can be evaluated independently of
    detection quality. The conf column is overridden (default 1.0) so every GT box
    is treated as a confident detection regardless of the GT file's own conf value.
    """
    rows = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame = int(float(parts[0]))
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            cls = int(float(parts[7])) if len(parts) > 7 else 0
            rows[frame].append([x, y, x + w, y + h, conf_override, cls])
    return {fr: np.array(v, dtype=np.float32) for fr, v in rows.items()}


def process_clip(
    model, tracker, class_names, clip_path, out_path,
    conf=0.3, iou=0.5, ema_alpha=0.5, det_interval=1, max_coast=30, coast_cls_ids=None,
    class_groups=None, nms_iou_thresh=0.5, mot_path=None, track_cls_ids=None,
    timing_path=None, no_video=False, timings_only=False, gt_dets=None,
):
    # timings_only is the strictest output mode: no video AND no MOT (pure benchmark).
    no_video = no_video or timings_only
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {clip_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    if not no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    smoother = BoxSmoother(alpha=ema_alpha)
    mot_lines = [] if mot_path else None
    frame_timings = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        t_frame_start = time.perf_counter()
        run_det = (frame_idx % det_interval == 1) or (det_interval == 1)

        dets = np.empty((0, 6), dtype=np.float32)
        t_det = 0.0
        t_nms = 0.0
        n_dets_before_nms = 0
        n_dets_after_nms = 0

        if gt_dets is not None:
            # GT-as-detections mode: feed ground-truth boxes to the tracker instead
            # of YOLO output, isolating tracking/ReID ability from detection quality.
            # The image `frame` is still passed to tracker.update so ReID runs on the
            # (possibly degraded) crops.
            dets = gt_dets.get(frame_idx, np.empty((0, 6), dtype=np.float32))
        elif run_det:
            t0 = time.perf_counter()
            results = model.predict(frame, conf=conf, iou=iou, verbose=False)
            t_det = time.perf_counter() - t0

            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy().reshape(-1, 1)
                clss = r.boxes.cls.cpu().numpy().reshape(-1, 1)
                dets = np.hstack([xyxy, confs, clss]).astype(np.float32)

        if track_cls_ids is not None and len(dets) > 0:
            mask = np.isin(dets[:, 5].astype(int), list(track_cls_ids))
            dets = dets[mask]

        n_dets_before_nms = len(dets)
        if gt_dets is None and class_groups and len(dets) > 0:
            t0 = time.perf_counter()
            dets = cross_modal_nms(dets, class_groups, nms_iou_thresh)
            t_nms = time.perf_counter() - t0
        n_dets_after_nms = len(dets)

        t0 = time.perf_counter()
        tracks = tracker.update(dets, frame)
        t_track = time.perf_counter() - t0

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

        if not no_video:
            draw_tracks(frame, tracks, coasting, class_names, smoother)
            writer.write(frame)

        t_total = time.perf_counter() - t_frame_start
        frame_timings.append({
            "frame": frame_idx,
            "det_ms": t_det * 1000,
            "nms_ms": t_nms * 1000,
            "track_ms": t_track * 1000,
            "total_ms": t_total * 1000,
            "n_dets": n_dets_before_nms,
            "n_dets_after_nms": n_dets_after_nms,
            "n_tracks": len(tracks) if len(tracks) > 0 else 0,
            "n_coasting": len(coasting),
            "ran_det": run_det,
        })

        if frame_idx % 100 == 0 or frame_idx == total:
            n_matched = len(tracks) if len(tracks) > 0 else 0
            n_coast = len(coasting)
            det_flag = "DET" if run_det else "KF"
            avg_fps = 1000.0 / (sum(t["total_ms"] for t in frame_timings) / len(frame_timings))
            print(f"  {frame_idx}/{total} frames [{det_flag}] (det: {n_matched}, predicted: {n_coast}) | {avg_fps:.1f} FPS", flush=True)

    cap.release()
    if writer:
        writer.release()

    if mot_path and mot_lines and not timings_only:
        Path(mot_path).parent.mkdir(parents=True, exist_ok=True)
        Path(mot_path).write_text("\n".join(mot_lines) + "\n")

    if timing_path and frame_timings:
        Path(timing_path).parent.mkdir(parents=True, exist_ok=True)
        with open(timing_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=frame_timings[0].keys())
            writer.writeheader()
            writer.writerows(frame_timings)

    return frame_timings


def build_class_groups(class_names):
    """Build cross-modal NMS groups for the 12-class domain-split taxonomy.
    Groups model classes that represent the same object type across RGB/thermal."""
    name_to_ids = {}
    for cls_id, name in class_names.items():
        base = name.replace("-rgb", "").replace("-thermal", "")
        name_to_ids.setdefault(base, set()).add(cls_id)
    return {base: ids for base, ids in name_to_ids.items() if len(ids) > 1}


def summarize_timings(timings, clip_name):
    if not timings:
        return {}
    det_frames = [t for t in timings[1:] if t["ran_det"]]
    all_frames = timings[1:]
    if not all_frames:
        return {}

    det_ms = [t["det_ms"] for t in det_frames] if det_frames else [0]
    track_ms = [t["track_ms"] for t in all_frames]
    nms_ms = [t["nms_ms"] for t in det_frames if t["nms_ms"] > 0] or [0]
    total_ms = [t["total_ms"] for t in all_frames]

    summary = {
        "clip": clip_name,
        "n_frames": len(timings),
        "det_mean_ms": np.mean(det_ms),
        "det_p50_ms": np.median(det_ms),
        "det_p95_ms": np.percentile(det_ms, 95),
        "track_mean_ms": np.mean(track_ms),
        "track_p50_ms": np.median(track_ms),
        "track_p95_ms": np.percentile(track_ms, 95),
        "nms_mean_ms": np.mean(nms_ms),
        "total_mean_ms": np.mean(total_ms),
        "total_p50_ms": np.median(total_ms),
        "total_p95_ms": np.percentile(total_ms, 95),
        "fps_mean": 1000.0 / np.mean(total_ms),
        "fps_p5": 1000.0 / np.percentile(total_ms, 95),
    }

    print(f"  Timing: det {summary['det_mean_ms']:.1f}ms | "
          f"track {summary['track_mean_ms']:.1f}ms | "
          f"nms {summary['nms_mean_ms']:.1f}ms | "
          f"total {summary['total_mean_ms']:.1f}ms | "
          f"{summary['fps_mean']:.1f} FPS (p5={summary['fps_p5']:.1f})")

    return summary


def run(weights, source_dir, out_dir, conf=0.3, iou=0.5, ema_alpha=0.5, device="cuda:0",
        det_interval=1, max_coast=30, coast_classes=None, nms_iou_thresh=0.5,
        enable_nms=False, save_mot=False, track_classes=None,
        tracker_iou=0.15, max_age=180, alpha=0.7, longterm_bank_length=150,
        longterm_reid_weight=0.25, longterm_reid_correction_thresh=0.5,
        longterm_reid_correction_thresh_low=0.5,
        no_video=False, reid_weights="clip_veri.pt", no_cmc=False,
        with_reid=True, with_longterm_reid=True, timings_only=False,
        reid_gallery="fifo", gallery_k=12, gallery_diversity=0.10,
        gt_dets_dir=None):
    model = YOLO(weights)
    class_names = model.names
    if gt_dets_dir is not None:
        print(f"GT-as-detections mode: feeding GT boxes from {gt_dets_dir} (YOLO bypassed)")

    # timings_only: pure benchmark — only timing CSVs are written (no video, no MOT).
    if timings_only:
        no_video = True

    clips = sorted(glob.glob(os.path.join(source_dir, "*")))
    clips = [c for c in clips if Path(c).suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".ts"}]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timing_dir = out_dir / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)

    track_cls_ids = None
    if track_classes:
        track_cls_ids = {k for k, v in class_names.items() if any(c in v.lower() for c in track_classes)}
        print(f"Tracking limited to: {[class_names[i] for i in sorted(track_cls_ids)]}")

    coast_cls_ids = None
    if coast_classes:
        coast_cls_ids = {k for k, v in class_names.items() if any(c in v.lower() for c in coast_classes)}
        print(f"Kalman coast limited to: {[class_names[i] for i in sorted(coast_cls_ids)]}")

    class_groups = None
    if enable_nms:
        class_groups = build_class_groups(class_names)
        print(f"Cross-modal NMS enabled (iou_thresh={nms_iou_thresh}): {class_groups}")

    mot_dir = None
    if save_mot and not timings_only:
        mot_dir = out_dir / "mot"
        mot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(clips)} clips -> {out_dir}")
    print(f"Classes: {class_names}")
    gallery_desc = reid_gallery if not with_reid else (
        f"{reid_gallery}" + (f"(k={gallery_k},div={gallery_diversity})" if reid_gallery == "best" else ""))
    print(f"ReID: {reid_weights} | gallery: {gallery_desc} | Video output: {'OFF' if no_video else 'ON'}")
    print(f"Tracker: HybridSORT | tracker_iou={tracker_iou} alpha={alpha} max_age={max_age} "
          f"longterm_bank={longterm_bank_length} longterm_w={longterm_reid_weight} "
          f"correction_thresh={longterm_reid_correction_thresh}")
    print(f"Detection: conf={conf} iou={iou} | EMA alpha={ema_alpha} | det_interval={det_interval} | max_coast={max_coast}")
    if mot_dir:
        print(f"MOT output: {mot_dir}")

    all_summaries = []

    for i, clip in enumerate(clips):
        name = Path(clip).stem
        timing_path = str(timing_dir / f"{name}.csv")

        skip_file = out_dir / f"{name}.mp4" if not no_video else Path(timing_path)
        if skip_file.exists():
            print(f"[{i+1}/{len(clips)}] SKIP: {name}")
            continue

        out_mp4 = out_dir / f"{name}.mp4"

        tracker = HybridSort(
            reid_weights=Path(reid_weights),
            device=device,
            half=False,
            per_class=True,
            nr_classes=len(class_names),
            det_thresh=conf,
            max_age=max_age,
            min_hits=1 if det_interval > 1 else 3,
            iou_threshold=tracker_iou,
            use_custom_kf=True,
            alpha=alpha,
            longterm_bank_length=longterm_bank_length,
            longterm_reid_weight=longterm_reid_weight,
            with_reid=with_reid,
            with_longterm_reid=with_longterm_reid,
            with_longterm_reid_correction=with_longterm_reid,
            longterm_reid_correction_thresh=longterm_reid_correction_thresh,
            longterm_reid_correction_thresh_low=longterm_reid_correction_thresh_low,
            reid_gallery=reid_gallery,
            gallery_k=gallery_k,
            gallery_diversity=gallery_diversity,
        )

        if no_cmc:
            # Replace ECC camera-motion compensation with a no-op (identity warp).
            # ECC runs cv2.findTransformECC on CPU every frame — a major cost for
            # static/fixed cameras where motion compensation is unnecessary.
            tracker.cmc = _NoCMC()

        mot_path = str(mot_dir / f"{name}.txt") if mot_dir else None

        gt_dets = None
        if gt_dets_dir is not None:
            gt_file = Path(gt_dets_dir) / f"{name}.txt"
            if not gt_file.exists():
                print(f"[{i+1}/{len(clips)}] SKIP {name}: no GT-dets file {gt_file.name}")
                continue
            gt_dets = load_gt_dets(gt_file)

        print(f"[{i+1}/{len(clips)}] {name} (det every {det_interval} frame(s))", flush=True)
        frame_timings = process_clip(
            model, tracker, class_names, clip, out_mp4,
            conf=conf, iou=iou, ema_alpha=ema_alpha, det_interval=det_interval,
            max_coast=max_coast, coast_cls_ids=coast_cls_ids,
            class_groups=class_groups, nms_iou_thresh=nms_iou_thresh,
            mot_path=mot_path, track_cls_ids=track_cls_ids,
            timing_path=timing_path, no_video=no_video, timings_only=timings_only,
            gt_dets=gt_dets,
        )
        if not no_video:
            size_mb = out_mp4.stat().st_size / 1e6
            print(f"  -> {out_mp4.name} ({size_mb:.1f} MB)")

        summary = summarize_timings(frame_timings, name)
        if summary:
            all_summaries.append(summary)

    if all_summaries:
        summary_path = timing_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2)

        summary_csv = timing_dir / "summary.csv"
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            writer.writeheader()
            writer.writerows(all_summaries)

        overall_fps = 1000.0 / np.mean([s["total_mean_ms"] for s in all_summaries])
        overall_det = np.mean([s["det_mean_ms"] for s in all_summaries])
        overall_track = np.mean([s["track_mean_ms"] for s in all_summaries])
        print(f"\n{'='*60}")
        print(f"Overall: {overall_fps:.1f} FPS | det {overall_det:.1f}ms | track {overall_track:.1f}ms")
        print(f"Timing logs: {timing_dir}")
        print(f"{'='*60}")

    print(f"Done. {len(list(out_dir.glob('*.mp4')))} mp4 files in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO + HybridSORT tracking with Kalman prediction for coasting tracks"
    )
    parser.add_argument("--config", default=None, help="Path to experiment config YAML (sets all defaults)")
    parser.add_argument("--weights", required=True, help="Path to YOLO .pt weights")
    parser.add_argument("--source", required=True, help="Directory of video clips")
    parser.add_argument("--out", default=None, help="Output directory (defaults to config's parent directory)")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold")
    parser.add_argument("--ema-alpha", type=float, default=None,
                        help="EMA smoothing factor (0=full history, 1=no smoothing)")
    parser.add_argument("--det-interval", type=int, default=None,
                        help="Run detection every N frames (1=every frame, 2=every other frame)")
    parser.add_argument("--max-coast", type=int, default=None,
                        help="Max frames to show Kalman-predicted boxes before hiding")
    parser.add_argument("--track-classes", nargs="*", default=None,
                        help="Only track these classes (substring match, e.g. 'boat'). Default: all classes")
    parser.add_argument("--coast-classes", nargs="*", default=None,
                        help="Only predict for these classes (substring match, e.g. 'boat'). Default: all classes")
    parser.add_argument("--enable-nms", action="store_true",
                        help="Enable cross-modal NMS (suppress duplicate detections across RGB/thermal class pairs)")
    parser.add_argument("--nms-iou-thresh", type=float, default=None,
                        help="IoU threshold for cross-modal NMS")
    parser.add_argument("--save-mot", action="store_true",
                        help="Save MOTChallenge-format tracking output to <out>/mot/")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video rendering — timing and MOT output only")
    parser.add_argument("--timings-only", action="store_true",
                        help="Pure benchmark — write only timing CSVs (no video, no MOT)")
    parser.add_argument("--reid-weights", default=None,
                        help="Path to ReID model weights (default: clip_veri.pt)")
    parser.add_argument("--no-cmc", action="store_true",
                        help="Disable ECC camera-motion compensation (no-op warp). "
                             "Speeds up tracking for static/fixed cameras.")
    parser.add_argument("--longterm-bank-length", type=int, default=None,
                        help="Override long-term ReID embedding bank depth (default 150 from config)")
    parser.add_argument("--no-longterm-reid", action="store_true",
                        help="Disable long-term ReID bank + correction (ablation)")
    parser.add_argument("--no-reid", action="store_true",
                        help="Disable ReID entirely — no crop extraction or embeddings (ablation)")
    parser.add_argument("--reid-gallery", choices=["fifo", "best"], default=None,
                        help="ReID appearance gallery: 'fifo' (stock, last-N mean) or "
                             "'best' (top-k highest-confidence diverse views)")
    parser.add_argument("--gallery-k", type=int, default=None,
                        help="Number of best frames to keep per track (--reid-gallery best). Default 12")
    parser.add_argument("--gallery-diversity", type=float, default=None,
                        help="Min cosine distance between kept views (--reid-gallery best). Default 0.10")
    parser.add_argument("--device", default=None, help="Torch device")
    parser.add_argument("--gt-dets", default=None,
                        help="Directory of MOT-format GT files (<clip>.txt). When set, GT "
                             "boxes are fed to the tracker as detections instead of YOLO, "
                             "isolating tracking/ReID ability from detection quality. "
                             "ReID still runs on the (possibly degraded) image crops.")
    args = parser.parse_args()

    # Base defaults
    p = dict(
        conf=0.3, iou=0.5, ema_alpha=1.0, det_interval=1, max_coast=10,
        track_classes=None, coast_classes=None, enable_nms=False, nms_iou_thresh=0.5,
        save_mot=False, device="cuda:0",
        tracker_iou=0.15, max_age=180, alpha=0.7, longterm_bank_length=150,
        longterm_reid_weight=0.25, longterm_reid_correction_thresh=0.5,
        longterm_reid_correction_thresh_low=0.5,
        reid_gallery="fifo", gallery_k=12, gallery_diversity=0.10,
    )

    # Apply config file values
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        print(f"Config: {args.config}")
        det = cfg.get("detection", {})
        trk = cfg.get("tracker", {})
        kal = cfg.get("kalman", {})
        mapping = {
            "conf": det.get("conf"),
            "iou": det.get("iou"),
            "enable_nms": det.get("enable_nms"),
            "nms_iou_thresh": det.get("nms_iou_thresh"),
            "tracker_iou": trk.get("iou_threshold"),
            "max_age": trk.get("max_age"),
            "alpha": trk.get("alpha"),
            "longterm_bank_length": trk.get("longterm_bank_length"),
            "longterm_reid_weight": trk.get("longterm_reid_weight"),
            "longterm_reid_correction_thresh": trk.get("longterm_reid_correction_thresh"),
            "longterm_reid_correction_thresh_low": trk.get("longterm_reid_correction_thresh_low"),
            "reid_gallery": trk.get("reid_gallery"),
            "gallery_k": trk.get("gallery_k"),
            "gallery_diversity": trk.get("gallery_diversity"),
            "max_coast": kal.get("max_coast"),
            "track_classes": (
                [det["track_classes"]] if isinstance(det.get("track_classes"), str)
                else det.get("track_classes")
            ),
            "coast_classes": (
                [kal["coast_classes"]] if isinstance(kal.get("coast_classes"), str)
                else kal.get("coast_classes")
            ),
        }
        p.update({k: v for k, v in mapping.items() if v is not None})

    # CLI flags override config (only when explicitly provided)
    if args.conf is not None: p["conf"] = args.conf
    if args.iou is not None: p["iou"] = args.iou
    if args.ema_alpha is not None: p["ema_alpha"] = args.ema_alpha
    if args.det_interval is not None: p["det_interval"] = args.det_interval
    if args.max_coast is not None: p["max_coast"] = args.max_coast
    if args.track_classes is not None: p["track_classes"] = args.track_classes
    if args.coast_classes is not None: p["coast_classes"] = args.coast_classes
    if args.enable_nms: p["enable_nms"] = True
    if args.nms_iou_thresh is not None: p["nms_iou_thresh"] = args.nms_iou_thresh
    if args.save_mot: p["save_mot"] = True
    if args.no_video: p["no_video"] = True
    if args.timings_only: p["timings_only"] = True
    if args.reid_weights is not None: p["reid_weights"] = args.reid_weights
    if args.no_cmc: p["no_cmc"] = True
    if args.longterm_bank_length is not None: p["longterm_bank_length"] = args.longterm_bank_length
    if args.no_longterm_reid: p["with_longterm_reid"] = False
    if args.no_reid: p["with_reid"] = False
    if args.reid_gallery is not None: p["reid_gallery"] = args.reid_gallery
    if args.gallery_k is not None: p["gallery_k"] = args.gallery_k
    if args.gallery_diversity is not None: p["gallery_diversity"] = args.gallery_diversity
    if args.device is not None: p["device"] = args.device
    if args.gt_dets is not None: p["gt_dets_dir"] = args.gt_dets

    # Resolve output directory: CLI > config parent dir > error
    out = args.out
    if out is None:
        if args.config:
            out = str(Path(args.config).parent)
            p["save_mot"] = True   # always write MOT output when driven by config
            print(f"Output dir: {out} (from config parent)")
        else:
            parser.error("--out is required when --config is not specified")

    run(args.weights, args.source, out, **p)
