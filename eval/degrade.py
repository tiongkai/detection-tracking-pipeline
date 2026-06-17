"""Generate degraded (corrupted) copies of eval videos for robustness evaluation.

This is a DATA GENERATOR — it writes augmented videos (and GT) to disk. Inference
is run separately with track/track_video_predict.py on the generated videos, so the
two concerns stay decoupled.

Two corruption families (see docs/degradation_eval_design.md):
  - geometry-preserving (lowlight, grayscale, invert, jpeg, motion_blur, defocus,
    contrast, fog): pixels change, boxes do NOT move -> GT is re-emitted unchanged.
  - geometry-changing (shake): a per-frame affine warps the frame; the SAME affine
    is applied to the GT boxes so the augmented GT stays aligned.

Each corruption has 5 severity levels (1..5), provisional ranges to be re-anchored
to real operating conditions (stream bitrate, dusk footage) later.

Batch a directory (typical use):
    .venv/bin/python eval/degrade.py \
        --video-dir eval_videos/wavy-boats/videos \
        --gt-dir    eval_videos/wavy-boats/labels \
        --kind lowlight --severity 3 \
        --out-dir   eval_videos/wavy-boats/aug/lowlight_s3
    # -> aug/lowlight_s3/videos/<clip>.mp4 and aug/lowlight_s3/labels/<clip>.txt

Single video:
    .venv/bin/python eval/degrade.py --video clip.mp4 --kind shake --severity 4 --out out.mp4

As a library:
    deg = make_degrader("lowlight", 3, seed=0)
    frame2, affine = deg.apply(frame, idx)            # affine None unless shake
    boxes2 = apply_affine_boxes(boxes_xyxy, affine)   # only for shake
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

KINDS = ["lowlight", "grayscale", "invert", "grayscale_invert", "jpeg", "motion_blur",
         "defocus", "contrast", "fog", "shake"]

# ---- geometry-preserving corruptions -------------------------------------------------

def _lowlight(img, sev, rng):
    gain = [0.6, 0.45, 0.3, 0.2, 0.12][sev - 1]
    gamma = [1.2, 1.5, 1.8, 2.2, 2.6][sev - 1]
    sigma = [3, 6, 10, 15, 22][sev - 1]
    x = img.astype(np.float32) / 255.0
    x = np.power(x, gamma) * gain                       # darken (gamma + gain)
    x = x + rng.normal(0, sigma / 255.0, x.shape)       # read/shot noise
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def _grayscale(img, sev, rng):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)          # 3-ch so downstream is unchanged


def _invert(img, sev, rng):
    return 255 - img                                    # colour negative (per-channel)


def _grayscale_invert(img, sev, rng):
    # grayscale then invert: black/white negative (monochrome, thermal-like)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(255 - g, cv2.COLOR_GRAY2BGR)


def _jpeg(img, sev, rng):
    # aggressive ladder so blocking is clearly visible even at 1080p
    q = [40, 25, 15, 8, 4][sev - 1]
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def _motion_blur(img, sev, rng):
    k = [3, 7, 11, 15, 21][sev - 1]
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k                         # horizontal motion
    return cv2.filter2D(img, -1, kernel)


def _defocus(img, sev, rng):
    r = [1, 2, 3, 4, 6][sev - 1]
    k = 2 * r + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def _contrast(img, sev, rng):
    f = [0.8, 0.65, 0.5, 0.38, 0.28][sev - 1]
    mean = img.reshape(-1, 3).mean(axis=0)
    x = (img.astype(np.float32) - mean) * f + mean
    return np.clip(x, 0, 255).astype(np.uint8)


def _fog(img, sev, rng):
    t = [0.85, 0.7, 0.55, 0.4, 0.28][sev - 1]           # transmission (lower = foggier)
    A = 220.0                                           # atmospheric light
    x = img.astype(np.float32) * t + A * (1 - t)
    return np.clip(x, 0, 255).astype(np.uint8)


_GEOM_PRESERVING = {
    "lowlight": _lowlight, "grayscale": _grayscale, "invert": _invert,
    "grayscale_invert": _grayscale_invert,
    "jpeg": _jpeg, "motion_blur": _motion_blur, "defocus": _defocus,
    "contrast": _contrast, "fog": _fog,
}


class _SimpleDegrader:
    """Stateless per-frame corruption (geometry-preserving). Returns (frame, None)."""

    def __init__(self, kind, severity, seed=0):
        self.fn = _GEOM_PRESERVING[kind]
        self.sev = severity
        self.rng = np.random.default_rng(seed)

    def apply(self, frame, idx):
        return self.fn(frame, self.sev, self.rng), None


# ---- geometry-changing corruption: camera shake --------------------------------------

class _ShakeDegrader:
    """Synthetic camera shake: a temporally-smoothed random walk of 2D translation +
    small rotation, applied as an affine warp. A ~6% pre-zoom keeps content in-frame
    (no black borders, boxes never exit). Returns (frame, affine 2x3) so the SAME
    affine can be applied to GT boxes via apply_affine_boxes().
    """

    def __init__(self, severity, seed=0):
        self.sev = severity
        self.rng = np.random.default_rng(seed)
        self.trans_frac = [0.005, 0.012, 0.022, 0.035, 0.05][severity - 1]
        self.rot_deg = [0.2, 0.5, 1.0, 1.5, 2.0][severity - 1]
        self.zoom = 1.06
        self._dx = self._dy = self._da = 0.0            # smoothed random-walk state

    def _step(self, scale):
        a = 0.7                                          # AR(1) smoothing -> real-looking shake
        self._dx = a * self._dx + (1 - a) * self.rng.normal(0, scale)
        self._dy = a * self._dy + (1 - a) * self.rng.normal(0, scale)
        self._da = a * self._da + (1 - a) * self.rng.normal(0, 1.0)

    def apply(self, frame, idx):
        h, w = frame.shape[:2]
        self._step(1.0)
        tx = self._dx * self.trans_frac * w
        ty = self._dy * self.trans_frac * h
        ang = self._da * self.rot_deg
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, self.zoom)
        M[0, 2] += tx
        M[1, 2] += ty
        out = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        return out, M


def apply_affine_boxes(boxes_xyxy, M):
    """Apply a 2x3 affine to xyxy boxes; re-fit axis-aligned boxes. boxes: (N,>=4)."""
    if M is None or len(boxes_xyxy) == 0:
        return boxes_xyxy
    b = np.asarray(boxes_xyxy, dtype=np.float32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    corners = np.stack([
        np.stack([x1, y1], 1), np.stack([x2, y1], 1),
        np.stack([x2, y2], 1), np.stack([x1, y2], 1),
    ], 1)                                               # (N,4,2)
    ones = np.ones((corners.shape[0], 4, 1), np.float32)
    warped = np.concatenate([corners, ones], 2) @ M.T   # (N,4,2)
    out = b.copy()
    out[:, 0] = warped[:, :, 0].min(1); out[:, 1] = warped[:, :, 1].min(1)
    out[:, 2] = warped[:, :, 0].max(1); out[:, 3] = warped[:, :, 1].max(1)
    return out


def make_degrader(kind, severity, seed=0):
    if kind not in KINDS:
        raise ValueError(f"unknown corruption '{kind}', choose from {KINDS}")
    if not 1 <= severity <= 5:
        raise ValueError("severity must be 1..5")
    if kind == "shake":
        return _ShakeDegrader(severity, seed)
    return _SimpleDegrader(kind, severity, seed)


# ---- generation -----------------------------------------------------------------------

def _load_gt_rows(path):
    rows = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            p = line.split(",")
            rows[int(float(p[0]))].append(p)
    return rows


def degrade_video(in_path, out_path, kind, severity, seed=0,
                  gt_in=None, gt_out=None, frames=0):
    """Write a degraded copy of one video. If gt_in given, also write GT to gt_out
    (re-emitted unchanged for geometry-preserving kinds; affine-warped for shake)."""
    deg = make_degrader(kind, severity, seed)
    cap = cv2.VideoCapture(str(in_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    gt_rows = _load_gt_rows(gt_in) if gt_in else None
    out_lines = [] if gt_out else None

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or (frames and idx >= frames):
            break
        idx += 1
        frame, M = deg.apply(frame, idx)
        out.write(frame)
        if gt_rows is not None:
            rows = gt_rows.get(idx, [])
            if M is not None and rows:
                boxes = np.array([[float(p[2]), float(p[3]),
                                   float(p[2]) + float(p[4]), float(p[3]) + float(p[5])]
                                  for p in rows], np.float32)
                wb = apply_affine_boxes(boxes, M)
                # clamp to frame; drop boxes that shook (partly) out of view, else a
                # zero-size crop would crash downstream ReID (cv2.resize on empty).
                wb[:, [0, 2]] = wb[:, [0, 2]].clip(0, w)
                wb[:, [1, 3]] = wb[:, [1, 3]].clip(0, h)
                for p, b in zip(rows, wb):
                    if (b[2] - b[0]) < 2 or (b[3] - b[1]) < 2:
                        continue
                    q = list(p)
                    q[2], q[3] = f"{b[0]:.2f}", f"{b[1]:.2f}"
                    q[4], q[5] = f"{b[2]-b[0]:.2f}", f"{b[3]-b[1]:.2f}"
                    out_lines.append(",".join(q))
            else:
                out_lines.extend(",".join(p) for p in rows)

    cap.release(); out.release()
    if out_lines is not None:
        Path(gt_out).parent.mkdir(parents=True, exist_ok=True)
        Path(gt_out).write_text("\n".join(out_lines) + "\n")
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--severity", type=int, default=3, help="1..5")
    ap.add_argument("--seed", type=int, default=0, help="seed for stochastic kinds (shake/noise)")
    ap.add_argument("--frames", type=int, default=0, help="limit frames (0 = all)")
    # single
    ap.add_argument("--video", help="single input video")
    ap.add_argument("--out", help="single output video")
    ap.add_argument("--gt", help="single input GT (MOT .txt)")
    ap.add_argument("--gt-out", help="single output GT")
    # batch
    ap.add_argument("--video-dir", help="batch: input video directory")
    ap.add_argument("--out-dir", help="batch: output root (writes videos/ and labels/)")
    ap.add_argument("--gt-dir", help="batch: flat GT directory (<clip>.txt)")
    args = ap.parse_args()

    if args.video_dir:
        in_dir = Path(args.video_dir)
        out_root = Path(args.out_dir)
        vids = sorted(p for p in in_dir.iterdir()
                      if p.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov"})
        print(f"{args.kind} sev{args.severity}: {len(vids)} videos -> {out_root}")
        for v in vids:
            gt_in = Path(args.gt_dir) / f"{v.stem}.txt" if args.gt_dir else None
            if gt_in and not gt_in.exists():
                gt_in = None
            gt_out = (out_root / "labels" / f"{v.stem}.txt") if (args.gt_dir and gt_in) else None
            n = degrade_video(v, out_root / "videos" / f"{v.stem}.mp4",
                              args.kind, args.severity, args.seed, gt_in, gt_out, args.frames)
            print(f"  {v.stem}: {n} frames" + (" (+GT)" if gt_out else ""))
    else:
        if not (args.video and args.out):
            ap.error("provide --video and --out (single) or --video-dir and --out-dir (batch)")
        n = degrade_video(args.video, args.out, args.kind, args.severity, args.seed,
                          args.gt, args.gt_out, args.frames)
        print(f"{args.kind} sev{args.severity}: {n} frames -> {args.out}"
              + (f" (+GT {args.gt_out})" if args.gt_out else ""))


if __name__ == "__main__":
    main()
