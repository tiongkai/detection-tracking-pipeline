"""MOT track correction tool.

Loads tracker output (MOT format) + video, lets users fix track IDs and
bounding boxes via keyboard/mouse, and exports corrected GT.

Usage:
    python annotation_tools/correct_tracks.py \
        --video path/to/clip.mp4 \
        --mot path/to/clip.txt \
        --out path/to/corrected.txt
"""

import argparse
import copy
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {
    0: "boat-rgb", 1: "vessel-rgb", 2: "human-rgb",
    3: "outboard motor-rgb", 4: "head-rgb", 5: "torso-rgb",
    6: "boat-thermal", 7: "vessel-thermal", 8: "human-thermal",
    9: "outboard motor-thermal", 10: "head-thermal", 11: "torso-thermal",
}


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

class TrackData:
    """In-memory MOT data with editing operations."""

    def __init__(self):
        # tracks[track_id][frame] = [x, y, w, h, conf, cls, vis]
        self.tracks: dict[int, dict[int, list]] = {}
        self.keyframes: set[tuple[int, int]] = set()  # (track_id, frame)
        self._undo_stack: list[dict] = []
        self.max_frame = 0

    def load_mot(self, path: str):
        """Load MOT-format file into tracks dict."""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split(",")
                frame = int(p[0])
                tid = int(p[1])
                x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
                conf = float(p[6]) if len(p) > 6 else 1.0
                cls = int(float(p[7])) if len(p) > 7 else 0
                vis = float(p[8]) if len(p) > 8 else 1.0

                if tid not in self.tracks:
                    self.tracks[tid] = {}
                self.tracks[tid][frame] = [x, y, w, h, conf, cls, vis]
                self.keyframes.add((tid, frame))
                self.max_frame = max(self.max_frame, frame)

    def save_mot(self, path: str):
        """Save tracks to MOT-format file."""
        lines = []
        for tid in sorted(self.tracks):
            for frame in sorted(self.tracks[tid]):
                x, y, w, h, conf, cls, vis = self.tracks[tid][frame]
                lines.append(
                    f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},{int(cls)},{vis:.2f}"
                )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines) + "\n")

    def get_frame_tracks(self, frame: int) -> list[tuple[int, list]]:
        """Return [(track_id, [x,y,w,h,conf,cls,vis]), ...] for a frame."""
        result = []
        for tid, frames in self.tracks.items():
            if frame in frames:
                result.append((tid, frames[frame]))
        return result

    def track_ids(self) -> list[int]:
        """All track IDs sorted."""
        return sorted(self.tracks.keys())

    def track_frame_range(self, tid: int) -> tuple[int, int]:
        """Return (first_frame, last_frame) for a track."""
        frames = sorted(self.tracks[tid].keys())
        return frames[0], frames[-1]

    # --- Edit operations ---

    def _save_undo(self):
        """Snapshot current state for undo."""
        self._undo_stack.append({
            "tracks": copy.deepcopy(self.tracks),
            "keyframes": copy.deepcopy(self.keyframes),
        })
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def undo(self) -> bool:
        """Restore previous state. Returns True if undo was performed."""
        if not self._undo_stack:
            return False
        state = self._undo_stack.pop()
        self.tracks = state["tracks"]
        self.keyframes = state["keyframes"]
        return True

    def reassign_id(self, old_id: int, new_id: int):
        """Merge old_id track into new_id. All frames from old_id are reassigned."""
        if old_id == new_id or old_id not in self.tracks:
            return
        self._save_undo()
        if new_id not in self.tracks:
            self.tracks[new_id] = {}
        for frame, box in self.tracks[old_id].items():
            if frame not in self.tracks[new_id]:
                self.tracks[new_id][frame] = box
                self.keyframes.add((new_id, frame))
            self.keyframes.discard((old_id, frame))
        del self.tracks[old_id]

    def update_box(self, tid: int, frame: int, x: float, y: float, w: float, h: float):
        """Update bounding box for a track at a specific frame. Marks as keyframe."""
        if tid not in self.tracks or frame not in self.tracks[tid]:
            return
        self._save_undo()
        old = self.tracks[tid][frame]
        self.tracks[tid][frame] = [x, y, w, h, old[4], old[5], old[6]]
        self.keyframes.add((tid, frame))

    def delete_track(self, tid: int):
        """Delete an entire track."""
        if tid not in self.tracks:
            return
        self._save_undo()
        for frame in list(self.tracks[tid]):
            self.keyframes.discard((tid, frame))
        del self.tracks[tid]

    def delete_box(self, tid: int, frame: int):
        """Delete a single box from a track at a specific frame."""
        if tid not in self.tracks or frame not in self.tracks[tid]:
            return
        self._save_undo()
        self.keyframes.discard((tid, frame))
        del self.tracks[tid][frame]
        if not self.tracks[tid]:
            del self.tracks[tid]

    def interpolate_track(self, tid: int):
        """Linearly interpolate between keyframes for a track."""
        if tid not in self.tracks:
            return
        kf_frames = sorted(f for f in self.tracks[tid] if (tid, f) in self.keyframes)
        if len(kf_frames) < 2:
            return
        self._save_undo()
        self.tracks[tid] = {f: self.tracks[tid][f] for f in kf_frames}
        for i in range(len(kf_frames) - 1):
            f_start, f_end = kf_frames[i], kf_frames[i + 1]
            box_start = self.tracks[tid][f_start]
            box_end = self.tracks[tid][f_end]
            n_gaps = f_end - f_start
            for f in range(f_start + 1, f_end):
                alpha = (f - f_start) / n_gaps
                interp = [
                    box_start[j] + alpha * (box_end[j] - box_start[j])
                    for j in range(4)
                ]
                interp.extend(box_start[4:])
                self.tracks[tid][f] = interp


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def get_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic color per track ID (matches tracker output)."""
    h = hashlib.md5(str(track_id).encode()).digest()
    return int(h[0]), int(h[1]), int(h[2])


class Renderer:
    """Draws annotated frames and track panel overlay."""

    PANEL_WIDTH = 220
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, track_data: TrackData):
        self.td = track_data
        self.panel_scroll = 0

    def draw_frame(self, frame_img: np.ndarray, frame_idx: int,
                   selected_tid=None, drag_box=None) -> np.ndarray:
        """Draw all track boxes on the frame. Returns annotated copy."""
        vis = frame_img.copy()
        entries = self.td.get_frame_tracks(frame_idx)

        for tid, box in entries:
            x, y, w, h, conf, cls, _ = box
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            color = get_color(tid)
            thickness = 3 if tid == selected_tid else 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            is_kf = (tid, frame_idx) in self.td.keyframes
            cls_name = CLASS_NAMES.get(int(cls), str(int(cls)))
            label = f"#{tid} {cls_name}"
            if not is_kf:
                label += " [interp]"

            (tw, th), _ = cv2.getTextSize(label, self.FONT, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 4), self.FONT, 0.5, (255, 255, 255), 1)

        if drag_box:
            dx, dy, dw, dh = drag_box
            cv2.rectangle(vis, (int(dx), int(dy)), (int(dx + dw), int(dy + dh)),
                          (0, 255, 255), 2)

        return vis

    def draw_panel(self, panel_h: int, frame_idx: int, max_frame: int,
                   selected_tid=None) -> np.ndarray:
        """Draw track list panel. Returns panel image."""
        panel = np.zeros((panel_h, self.PANEL_WIDTH, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        cv2.putText(panel, f"Frame {frame_idx}/{max_frame}", (10, 25),
                    self.FONT, 0.5, (200, 200, 200), 1)
        cv2.line(panel, (0, 35), (self.PANEL_WIDTH, 35), (80, 80, 80), 1)

        track_ids = self.td.track_ids()
        row_h = 22
        visible_rows = (panel_h - 50) // row_h
        self.panel_scroll = max(0, min(self.panel_scroll, max(0, len(track_ids) - visible_rows)))
        visible_ids = track_ids[self.panel_scroll:self.panel_scroll + visible_rows]

        for i, tid in enumerate(visible_ids):
            y_pos = 50 + i * row_h
            f_start, f_end = self.td.track_frame_range(tid)
            active = f_start <= frame_idx <= f_end
            color = get_color(tid)

            if tid == selected_tid:
                cv2.rectangle(panel, (0, y_pos - row_h + 6), (self.PANEL_WIDTH, y_pos + 6),
                              (60, 60, 60), -1)

            cv2.rectangle(panel, (8, y_pos - 10), (18, y_pos), color, -1)

            text_color = (255, 255, 255) if active else (120, 120, 120)
            label = f"#{tid}  {f_start}-{f_end}"
            cv2.putText(panel, label, (24, y_pos), self.FONT, 0.4, text_color, 1)

        return panel

    def compose(self, frame_img: np.ndarray, frame_idx: int, max_frame: int,
                selected_tid=None, drag_box=None) -> np.ndarray:
        """Compose annotated frame + panel side by side."""
        annotated = self.draw_frame(frame_img, frame_idx, selected_tid, drag_box)
        panel = self.draw_panel(annotated.shape[0], frame_idx, max_frame, selected_tid)
        return np.hstack([annotated, panel])

    def panel_track_at_y(self, y: int):
        """Return track ID at panel y coordinate, or None."""
        if y < 50:
            return None
        row_h = 22
        idx = self.panel_scroll + (y - 50) // row_h
        track_ids = self.td.track_ids()
        if 0 <= idx < len(track_ids):
            return track_ids[idx]
        return None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class App:
    """Main application — OpenCV window loop with keyboard/mouse interaction."""

    WINDOW = "MOT Correction Tool"

    def __init__(self, video_path: str, mot_path: str, out_path: str):
        self.video_path = video_path
        self.out_path = out_path

        self.td = TrackData()
        self.td.load_mot(mot_path)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.td.max_frame = max(self.td.max_frame, self.total_frames)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0

        self.renderer = Renderer(self.td)
        self.frame_idx = 1
        self.selected_tid = None
        self.cached_frame = None
        self.dirty = False

        self.mouse_x = 0
        self.mouse_y = 0
        self.dragging = False
        self.drag_start = None
        self.drag_tid = None
        self.drag_handle = None
        self.drag_orig_box = None
        self._last_composed_shape = None  # (h, w) of last displayed image

    def _read_frame(self, idx: int):
        """Read a specific frame from the video (1-indexed)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
        ret, frame = self.cap.read()
        return frame if ret else None

    def _refresh(self):
        self.cached_frame = self._read_frame(self.frame_idx)

    def _display(self, drag_box=None):
        if self.cached_frame is None:
            return
        composed = self.renderer.compose(
            self.cached_frame, self.frame_idx, self.total_frames,
            self.selected_tid, drag_box,
        )

        h, w = composed.shape[:2]
        bar = np.zeros((30, w, 3), dtype=np.uint8)
        bar[:] = (40, 40, 40)
        status = "h/l=nav  H/L=skip10  <>=skip30  r=reassign  d=del box  x=del track  k=interp  u=undo  s=save  q=quit"
        cv2.putText(bar, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        composed = np.vstack([composed, bar])

        self._last_composed_shape = composed.shape[:2]
        cv2.imshow(self.WINDOW, composed)

    def _scale_mouse(self, x: int, y: int) -> tuple[int, int]:
        """Map window-space mouse coordinates to image-space coordinates."""
        if self._last_composed_shape is None:
            return x, y
        try:
            rx, ry, rw, rh = cv2.getWindowImageRect(self.WINDOW)
        except Exception:
            return x, y
        if rw <= 0 or rh <= 0:
            return x, y
        img_h, img_w = self._last_composed_shape
        sx = img_w / rw
        sy = img_h / rh
        return int((x - rx) * sx), int((y - ry) * sy)

    def _goto_frame(self, idx: int):
        self.frame_idx = max(1, min(idx, self.total_frames))
        self._refresh()

    def _input_popup(self, title: str) -> str | None:
        """Show a small OpenCV popup to collect text input. Returns string or None if cancelled."""
        WIN = "Input"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 320, 90)
        text = ""
        while True:
            popup = np.zeros((90, 320, 3), dtype=np.uint8)
            popup[:] = (50, 50, 50)
            cv2.putText(popup, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.rectangle(popup, (10, 32), (310, 60), (80, 80, 80), -1)
            cv2.rectangle(popup, (10, 32), (310, 60), (150, 150, 150), 1)
            cv2.putText(popup, text + "|", (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(popup, "Enter=confirm   Esc=cancel", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)
            cv2.imshow(WIN, popup)
            key = cv2.waitKey(50) & 0xFF
            if key == 13:  # Enter
                cv2.destroyWindow(WIN)
                return text or None
            elif key == 27:  # Esc
                cv2.destroyWindow(WIN)
                return None
            elif key == 8:  # Backspace
                text = text[:-1]
            elif 32 <= key <= 126:  # all printable ASCII
                text += chr(key)

    def _handle_key(self, key: int):
        if key == 83 or key == ord("l"):
            self._goto_frame(self.frame_idx + 1)
        elif key == 81 or key == ord("h"):
            self._goto_frame(self.frame_idx - 1)
        elif key == ord("L"):
            self._goto_frame(self.frame_idx + 10)
        elif key == ord("H"):
            self._goto_frame(self.frame_idx - 10)
        elif key == ord("."):
            self._goto_frame(self.frame_idx + 1)
        elif key == ord(","):
            self._goto_frame(self.frame_idx - 1)
        elif key == ord(">"):
            self._goto_frame(self.frame_idx + 30)
        elif key == ord("<"):
            self._goto_frame(self.frame_idx - 30)

        elif key == ord("g"):
            if self.selected_tid is not None and self.selected_tid in self.td.tracks:
                f_start, _ = self.td.track_frame_range(self.selected_tid)
                self._goto_frame(f_start)

        elif key == ord("r"):
            if self.selected_tid is not None:
                raw = self._input_popup(f"Reassign track #{self.selected_tid} to ID:")
                if raw is not None:
                    try:
                        new_id = int(raw)
                        self.td.reassign_id(self.selected_tid, new_id)
                        self.selected_tid = new_id
                        self.dirty = True
                        print(f"  Reassigned to #{new_id}")
                    except ValueError:
                        pass

        elif key == ord("x"):
            if self.selected_tid is not None:
                print(f"  Deleted track #{self.selected_tid}")
                self.td.delete_track(self.selected_tid)
                self.selected_tid = None
                self.dirty = True

        elif key == ord("d"):
            if self.selected_tid is not None:
                self.td.delete_box(self.selected_tid, self.frame_idx)
                self.dirty = True

        elif key == ord("k"):
            if self.selected_tid is not None:
                self.td.interpolate_track(self.selected_tid)
                self.dirty = True
                print(f"  Interpolated track #{self.selected_tid}")

        elif key == ord("u"):
            if self.td.undo():
                self.dirty = True
                print("  Undo")

        elif key == ord("s"):
            self.td.save_mot(self.out_path)
            self.dirty = False
            print(f"  Saved to {self.out_path}")

    # --- Mouse ---

    def _hit_test(self, x: int, y: int):
        """Find which track box contains (x, y). Returns (track_id, handle)."""
        entries = self.td.get_frame_tracks(self.frame_idx)
        for tid, box in reversed(entries):
            bx, by, bw, bh = box[:4]
            if abs(x - (bx + bw)) < 12 and abs(y - (by + bh)) < 12:
                return tid, "br"
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return tid, "move"
        return None, None

    def _on_mouse(self, event, x, y, flags, param):
        x, y = self._scale_mouse(x, y)
        self.mouse_x = x
        self.mouse_y = y
        frame_w = self.cached_frame.shape[1] if self.cached_frame is not None else 0

        if event == cv2.EVENT_LBUTTONDOWN and x >= frame_w:
            panel_y = y
            tid = self.renderer.panel_track_at_y(panel_y)
            if tid is not None:
                self.selected_tid = tid
                f_start, _ = self.td.track_frame_range(tid)
                self._goto_frame(f_start)
            return

        if event == cv2.EVENT_MOUSEWHEEL and x >= frame_w:
            if flags > 0:
                self.renderer.panel_scroll = max(0, self.renderer.panel_scroll - 3)
            else:
                self.renderer.panel_scroll += 3
            return

        if event == cv2.EVENT_LBUTTONDOWN and x < frame_w:
            tid, handle = self._hit_test(x, y)
            if tid is not None:
                self.selected_tid = tid
                self.dragging = True
                self.drag_start = (x, y)
                self.drag_tid = tid
                self.drag_handle = handle
                self.drag_orig_box = list(self.td.tracks[tid][self.frame_idx][:4])
            else:
                self.selected_tid = None
            return

        if event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.drag_start[0]
            dy = y - self.drag_start[1]
            ob = self.drag_orig_box
            if self.drag_handle == "move":
                new_box = (ob[0] + dx, ob[1] + dy, ob[2], ob[3])
            elif self.drag_handle == "br":
                new_box = (ob[0], ob[1], max(10, ob[2] + dx), max(10, ob[3] + dy))
            else:
                return
            self._display(drag_box=new_box)
            return

        if event == cv2.EVENT_LBUTTONUP and self.dragging:
            dx = x - self.drag_start[0]
            dy = y - self.drag_start[1]
            ob = self.drag_orig_box
            if self.drag_handle == "move":
                nx, ny, nw, nh = ob[0] + dx, ob[1] + dy, ob[2], ob[3]
            elif self.drag_handle == "br":
                nx, ny = ob[0], ob[1]
                nw, nh = max(10, ob[2] + dx), max(10, ob[3] + dy)
            else:
                self.dragging = False
                return

            if abs(dx) > 2 or abs(dy) > 2:
                self.td.update_box(self.drag_tid, self.frame_idx, nx, ny, nw, nh)
                self.dirty = True

            self.dragging = False
            self.drag_tid = None
            self.drag_handle = None
            self.drag_orig_box = None

    # --- Main loop ---

    def run(self):
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 1280, 720)
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)
        self._refresh()

        print(f"\nLoaded {len(self.td.tracks)} tracks, {self.total_frames} frames")
        print(f"Output: {self.out_path}\n")

        while True:
            self._display()
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                if self.dirty:
                    resp = self._input_popup("Unsaved changes. Save? (y/n):")
                    if resp and resp.lower() == "y":
                        self.td.save_mot(self.out_path)
                        print(f"  Saved to {self.out_path}")
                break
            elif key != 255:
                self._handle_key(key)

        self.cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_tests():
    """Data model self-tests."""
    test_in = (
        "1,1,100.00,200.00,50.00,60.00,0.9500,0,1.00\n"
        "1,2,300.00,400.00,70.00,80.00,0.8700,6,1.00\n"
        "2,1,105.00,205.00,50.00,60.00,0.9300,0,1.00\n"
        "5,1,120.00,220.00,50.00,60.00,0.9100,0,1.00\n"
    )
    Path("/tmp/test_mot_in.txt").write_text(test_in)

    td = TrackData()
    td.load_mot("/tmp/test_mot_in.txt")
    assert len(td.tracks) == 2
    assert td.max_frame == 5
    assert td.track_frame_range(1) == (1, 5)
    assert td.track_frame_range(2) == (1, 1)
    assert len(td.get_frame_tracks(1)) == 2
    print("PASS: load")

    td.reassign_id(2, 1)
    assert 2 not in td.tracks
    print("PASS: reassign_id")

    td.undo()
    assert 2 in td.tracks
    print("PASS: undo")

    td.update_box(1, 1, 110.0, 210.0, 55.0, 65.0)
    assert td.tracks[1][1][0] == 110.0
    print("PASS: update_box")

    td2 = TrackData()
    td2.load_mot("/tmp/test_mot_in.txt")
    td2.interpolate_track(1)
    assert 3 in td2.tracks[1], "Frame 3 should be interpolated"
    assert 4 in td2.tracks[1], "Frame 4 should be interpolated"
    x3 = td2.tracks[1][3][0]
    assert 106 < x3 < 114, f"Interpolated x should be ~110, got {x3}"
    print("PASS: interpolate_track")

    td.delete_box(1, 1)
    assert 1 not in td.tracks[1]
    print("PASS: delete_box")

    td2.save_mot("/tmp/test_mot_out.txt")
    out = Path("/tmp/test_mot_out.txt").read_text()
    lines = [l for l in out.strip().split("\n") if l]
    assert len(lines) == 6, f"Expected 6 lines, got {len(lines)}"
    print("PASS: save roundtrip")

    print("\nAll tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOT track correction tool")
    parser.add_argument("--video", help="Path to video file")
    parser.add_argument("--mot", help="Path to MOT-format track file")
    parser.add_argument("--out", default=None,
                        help="Output path for corrected MOT file (default: <mot>_corrected.txt)")
    parser.add_argument("--test", action="store_true", help="Run data model self-tests")
    args = parser.parse_args()

    if args.test:
        _run_tests()
        sys.exit(0)

    if not args.video or not args.mot:
        parser.error("--video and --mot are required (or use --test)")

    out_path = args.out or str(Path(args.mot).with_suffix("")) + "_corrected.txt"
    app = App(args.video, args.mot, out_path)
    app.run()
