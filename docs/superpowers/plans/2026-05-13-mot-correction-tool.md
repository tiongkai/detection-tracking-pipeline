# MOT Correction Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an OpenCV-based MOT annotation correction tool that loads tracker output + video, lets users fix track IDs and bounding boxes, and exports corrected GT in MOTChallenge format.

**Architecture:** Single-file tool split into three layers: (1) `TrackData` — in-memory MOT data model with ID reassignment, bbox editing, keyframe interpolation, and undo; (2) `Renderer` — draws boxes, labels, track panel onto video frames; (3) `App` — OpenCV window loop handling keyboard/mouse input and coordinating the other two. No GUI framework — pure OpenCV highgui + mouse callbacks.

**Tech Stack:** Python, OpenCV (`cv2`), numpy. No additional dependencies beyond what's already in the `boat-tracker` conda environment.

---

## File Structure

```
annotation_tools/
└── correct_tracks.py    # Single-file tool (~500 lines)
```

One file, three classes:
- `TrackData` — loads/saves MOT txt, stores per-track-per-frame boxes, handles ID reassignment, bbox edits, keyframe interpolation, undo stack
- `Renderer` — draws annotated frame + track panel overlay
- `App` — main loop, keyboard dispatch, mouse state machine

---

## Data Model

The core data structure is a dict-of-dicts:

```python
# tracks[track_id][frame] = [x, y, w, h, conf, cls, vis]
tracks: dict[int, dict[int, list]]
```

This makes ID reassignment O(1) — just move the inner dict to a new key. Keyframe interpolation iterates the sorted frame keys for a single track.

A separate set tracks which (track_id, frame) pairs are "keyframes" — user-edited boxes that anchors interpolation. On load, every box from the MOT file is marked as a keyframe (since they're all "real" data).

---

## Task 1: TrackData — Load, Save, Data Model

**Files:**
- Create: `annotation_tools/correct_tracks.py`

This task creates the file and implements the data layer only. No rendering, no interaction.

- [ ] **Step 1: Create file with TrackData class — load MOT**

```python
"""MOT track correction tool.

Usage:
    python annotation_tools/correct_tracks.py \
        --video path/to/clip.mp4 \
        --mot path/to/clip.txt \
        --out path/to/corrected.txt
"""

import argparse
import copy
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {
    0: "boat-rgb", 1: "vessel-rgb", 2: "human-rgb",
    3: "outboard motor-rgb", 4: "head-rgb", 5: "torso-rgb",
    6: "boat-thermal", 7: "vessel-thermal", 8: "human-thermal",
    9: "outboard motor-thermal", 10: "head-thermal", 11: "torso-thermal",
}


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
```

- [ ] **Step 2: Verify load/save roundtrip works**

Add a temporary test block at the bottom:

```python
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test-roundtrip":
        # Create a small MOT file, load it, save it, compare
        test_in = "1,1,100.00,200.00,50.00,60.00,0.9500,0,1.00\n1,2,300.00,400.00,70.00,80.00,0.8700,6,1.00\n2,1,105.00,205.00,50.00,60.00,0.9300,0,1.00\n"
        Path("/tmp/test_mot_in.txt").write_text(test_in)
        td = TrackData()
        td.load_mot("/tmp/test_mot_in.txt")
        assert len(td.tracks) == 2, f"Expected 2 tracks, got {len(td.tracks)}"
        assert td.max_frame == 2
        assert td.track_frame_range(1) == (1, 2)
        assert td.track_frame_range(2) == (1, 1)
        ft = td.get_frame_tracks(1)
        assert len(ft) == 2, f"Expected 2 tracks at frame 1, got {len(ft)}"
        td.save_mot("/tmp/test_mot_out.txt")
        out = Path("/tmp/test_mot_out.txt").read_text()
        assert "1,1,100.00,200.00,50.00,60.00,0.9500,0,1.00" in out
        assert "1,2,300.00,400.00,70.00,80.00,0.8700,6,1.00" in out
        assert "2,1,105.00,205.00,50.00,60.00,0.9300,0,1.00" in out
        print("PASS: load/save roundtrip")
        sys.exit(0)
```

Run: `python annotation_tools/correct_tracks.py --test-roundtrip`
Expected: `PASS: load/save roundtrip`

- [ ] **Step 3: Commit**

```bash
git add annotation_tools/correct_tracks.py
git commit -m "feat: add TrackData class with MOT load/save"
```

---

## Task 2: TrackData — Edit Operations (reassign, bbox edit, undo)

**Files:**
- Modify: `annotation_tools/correct_tracks.py`

Add the editing operations to `TrackData`. These are the core mutations the UI will call.

- [ ] **Step 1: Add _save_undo, reassign_id, update_box methods**

Add these methods to the `TrackData` class, after `track_frame_range`:

```python
    def _save_undo(self):
        """Snapshot current state for undo."""
        self._undo_stack.append({
            "tracks": copy.deepcopy(self.tracks),
            "keyframes": copy.deepcopy(self.keyframes),
        })
        # Cap undo stack at 50 to limit memory
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
        # Move all frames from old to new (new_id's existing frames take precedence on conflict)
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
```

- [ ] **Step 2: Add interpolate_track method**

Add after `delete_box`:

```python
    def interpolate_track(self, tid: int):
        """Linearly interpolate between keyframes for a track.

        Only fills gaps between existing keyframes — does not extrapolate.
        Non-keyframe entries are replaced by interpolated values.
        """
        if tid not in self.tracks:
            return
        kf_frames = sorted(f for f in self.tracks[tid] if (tid, f) in self.keyframes)
        if len(kf_frames) < 2:
            return
        self._save_undo()
        # Remove non-keyframe entries
        self.tracks[tid] = {f: self.tracks[tid][f] for f in kf_frames}
        # Interpolate between consecutive keyframes
        for i in range(len(kf_frames) - 1):
            f_start, f_end = kf_frames[i], kf_frames[i + 1]
            box_start = self.tracks[tid][f_start]
            box_end = self.tracks[tid][f_end]
            n_gaps = f_end - f_start
            for f in range(f_start + 1, f_end):
                alpha = (f - f_start) / n_gaps
                interp = [
                    box_start[j] + alpha * (box_end[j] - box_start[j])
                    for j in range(4)  # x, y, w, h
                ]
                # Keep conf, cls, vis from nearest keyframe
                interp.extend(box_start[4:])
                self.tracks[tid][f] = interp
```

- [ ] **Step 3: Add tests for edit operations**

Extend the `--test-roundtrip` block (replace the existing one):

```python
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test-roundtrip":
        test_in = "1,1,100.00,200.00,50.00,60.00,0.9500,0,1.00\n1,2,300.00,400.00,70.00,80.00,0.8700,6,1.00\n2,1,105.00,205.00,50.00,60.00,0.9300,0,1.00\n5,1,120.00,220.00,50.00,60.00,0.9100,0,1.00\n"
        Path("/tmp/test_mot_in.txt").write_text(test_in)
        td = TrackData()
        td.load_mot("/tmp/test_mot_in.txt")

        # Test load
        assert len(td.tracks) == 2
        assert td.max_frame == 5
        print("PASS: load")

        # Test reassign
        td.reassign_id(2, 1)
        assert 2 not in td.tracks
        assert 1 in td.get_frame_tracks(1)[0]
        print("PASS: reassign_id")

        # Test undo
        td.undo()
        assert 2 in td.tracks
        print("PASS: undo")

        # Test update_box
        td.update_box(1, 1, 110.0, 210.0, 55.0, 65.0)
        assert td.tracks[1][1][0] == 110.0
        print("PASS: update_box")

        # Test interpolation
        td2 = TrackData()
        td2.load_mot("/tmp/test_mot_in.txt")
        td2.interpolate_track(1)
        assert 3 in td2.tracks[1], "Frame 3 should be interpolated"
        assert 4 in td2.tracks[1], "Frame 4 should be interpolated"
        x3 = td2.tracks[1][3][0]
        assert 106 < x3 < 114, f"Interpolated x at frame 3 should be between 105 and 120, got {x3}"
        print("PASS: interpolate_track")

        # Test delete
        td.delete_box(1, 1)
        assert 1 not in td.tracks[1]
        print("PASS: delete_box")

        # Test save
        td2.save_mot("/tmp/test_mot_out.txt")
        out = Path("/tmp/test_mot_out.txt").read_text()
        assert len(out.strip().split("\n")) == 6, "Should have 5 original + 2 interpolated frames for track 1 + 1 for track 2"
        print("PASS: save roundtrip")

        print("\nAll tests passed.")
        sys.exit(0)
```

Run: `python annotation_tools/correct_tracks.py --test-roundtrip`
Expected: `All tests passed.`

- [ ] **Step 4: Commit**

```bash
git add annotation_tools/correct_tracks.py
git commit -m "feat: add TrackData edit operations — reassign, bbox update, interpolation, undo"
```

---

## Task 3: Renderer — Draw Annotated Frame + Track Panel

**Files:**
- Modify: `annotation_tools/correct_tracks.py`

Add the `Renderer` class that draws boxes on a video frame and renders the track panel overlay.

- [ ] **Step 1: Add color utility and Renderer class**

Add after the `TrackData` class:

```python
import hashlib


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
                   selected_tid: int | None = None,
                   drag_box: tuple | None = None) -> np.ndarray:
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

        # Draw drag box preview if active
        if drag_box:
            dx, dy, dw, dh = drag_box
            cv2.rectangle(vis, (int(dx), int(dy)), (int(dx + dw), int(dy + dh)),
                          (0, 255, 255), 2)

        return vis

    def draw_panel(self, panel_h: int, frame_idx: int, max_frame: int,
                   selected_tid: int | None = None) -> np.ndarray:
        """Draw track list panel. Returns panel image."""
        panel = np.zeros((panel_h, self.PANEL_WIDTH, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        # Header
        cv2.putText(panel, f"Frame {frame_idx}/{max_frame}", (10, 25),
                    self.FONT, 0.5, (200, 200, 200), 1)
        cv2.line(panel, (0, 35), (self.PANEL_WIDTH, 35), (80, 80, 80), 1)

        # Track list
        track_ids = self.td.track_ids()
        row_h = 22
        visible_rows = (panel_h - 50) // row_h
        self.panel_scroll = max(0, min(self.panel_scroll, len(track_ids) - visible_rows))
        visible_ids = track_ids[self.panel_scroll:self.panel_scroll + visible_rows]

        for i, tid in enumerate(visible_ids):
            y_pos = 50 + i * row_h
            f_start, f_end = self.td.track_frame_range(tid)
            active = f_start <= frame_idx <= f_end
            color = get_color(tid)

            # Highlight selected
            if tid == selected_tid:
                cv2.rectangle(panel, (0, y_pos - row_h + 6), (self.PANEL_WIDTH, y_pos + 6),
                              (60, 60, 60), -1)

            # Color swatch
            cv2.rectangle(panel, (8, y_pos - 10), (18, y_pos), color, -1)

            # Track info
            text_color = (255, 255, 255) if active else (120, 120, 120)
            label = f"#{tid}  {f_start}-{f_end}"
            cv2.putText(panel, label, (24, y_pos), self.FONT, 0.4, text_color, 1)

        return panel

    def compose(self, frame_img: np.ndarray, frame_idx: int, max_frame: int,
                selected_tid: int | None = None,
                drag_box: tuple | None = None) -> np.ndarray:
        """Compose annotated frame + panel side by side."""
        annotated = self.draw_frame(frame_img, frame_idx, selected_tid, drag_box)
        panel = self.draw_panel(annotated.shape[0], frame_idx, max_frame, selected_tid)
        return np.hstack([annotated, panel])

    def panel_track_at_y(self, y: int) -> int | None:
        """Return track ID at panel y coordinate, or None."""
        if y < 50:
            return None
        row_h = 22
        idx = self.panel_scroll + (y - 50) // row_h
        track_ids = self.td.track_ids()
        if 0 <= idx < len(track_ids):
            return track_ids[idx]
        return None
```

- [ ] **Step 2: Commit**

```bash
git add annotation_tools/correct_tracks.py
git commit -m "feat: add Renderer class — frame annotation + track panel"
```

---

## Task 4: App — Main Loop, Keyboard Controls, Frame Navigation

**Files:**
- Modify: `annotation_tools/correct_tracks.py`

Add the `App` class with the main loop, keyboard handling, and frame navigation. No mouse interaction yet — that's Task 5.

- [ ] **Step 1: Add App class with main loop and keyboard controls**

Add after the `Renderer` class:

```python
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
        self.frame_idx = 1  # MOT is 1-indexed
        self.selected_tid: int | None = None
        self.cached_frame: np.ndarray | None = None
        self.dirty = False

        # Mouse state
        self.mouse_x = 0
        self.mouse_y = 0
        self.dragging = False
        self.drag_start = None
        self.drag_tid: int | None = None
        self.drag_handle: str | None = None  # "move", "br" (bottom-right resize)
        self.drag_orig_box: list | None = None

    def _read_frame(self, idx: int) -> np.ndarray | None:
        """Read a specific frame from the video (1-indexed)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
        ret, frame = self.cap.read()
        return frame if ret else None

    def _refresh(self):
        """Re-read current frame and update display."""
        self.cached_frame = self._read_frame(self.frame_idx)

    def _display(self, drag_box=None):
        """Render and show current frame."""
        if self.cached_frame is None:
            return
        composed = self.renderer.compose(
            self.cached_frame, self.frame_idx, self.total_frames,
            self.selected_tid, drag_box,
        )

        # Status bar at bottom
        h, w = composed.shape[:2]
        bar = np.zeros((30, w, 3), dtype=np.uint8)
        bar[:] = (40, 40, 40)
        status = "KEYS: arrows=nav  shift+arrows=skip10  r=reassign  d=delete  k=interpolate  u=undo  s=save  q=quit"
        cv2.putText(bar, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        composed = np.vstack([composed, bar])

        cv2.imshow(self.WINDOW, composed)

    def _goto_frame(self, idx: int):
        """Navigate to a specific frame."""
        self.frame_idx = max(1, min(idx, self.total_frames))
        self._refresh()

    def _prompt_id(self, prompt: str) -> int | None:
        """Prompt user for a track ID via terminal input."""
        try:
            val = input(f"\n  {prompt}: ")
            return int(val)
        except (ValueError, EOFError):
            return None

    def _handle_key(self, key: int):
        """Process keyboard input."""
        # Arrow keys (macOS/Linux key codes)
        if key == 83 or key == ord("l"):  # right arrow or l
            self._goto_frame(self.frame_idx + 1)
        elif key == 81 or key == ord("h"):  # left arrow or h
            self._goto_frame(self.frame_idx - 1)
        elif key == ord("L"):  # shift+L = skip 10 forward
            self._goto_frame(self.frame_idx + 10)
        elif key == ord("H"):  # shift+H = skip 10 back
            self._goto_frame(self.frame_idx - 10)
        elif key == ord("."):  # skip 1 forward (alternate)
            self._goto_frame(self.frame_idx + 1)
        elif key == ord(","):  # skip 1 back (alternate)
            self._goto_frame(self.frame_idx - 1)
        elif key == ord(">"):  # skip 30 forward
            self._goto_frame(self.frame_idx + 30)
        elif key == ord("<"):  # skip 30 back
            self._goto_frame(self.frame_idx - 30)

        # Track selection by clicking panel (handled in mouse), but also number jump
        elif key == ord("g"):  # go to track — jump to first frame
            if self.selected_tid is not None:
                f_start, _ = self.td.track_frame_range(self.selected_tid)
                self._goto_frame(f_start)

        # Reassign selected track
        elif key == ord("r"):
            if self.selected_tid is not None:
                new_id = self._prompt_id(f"Reassign track #{self.selected_tid} to ID")
                if new_id is not None:
                    self.td.reassign_id(self.selected_tid, new_id)
                    self.selected_tid = new_id
                    self.dirty = True
                    print(f"  Reassigned to #{new_id}")

        # Delete selected track
        elif key == ord("x"):
            if self.selected_tid is not None:
                print(f"  Deleted track #{self.selected_tid}")
                self.td.delete_track(self.selected_tid)
                self.selected_tid = None
                self.dirty = True

        # Delete single box
        elif key == ord("d"):
            if self.selected_tid is not None:
                self.td.delete_box(self.selected_tid, self.frame_idx)
                self.dirty = True

        # Interpolate selected track
        elif key == ord("k"):
            if self.selected_tid is not None:
                self.td.interpolate_track(self.selected_tid)
                self.dirty = True
                print(f"  Interpolated track #{self.selected_tid}")

        # Undo
        elif key == ord("u"):
            if self.td.undo():
                self.dirty = True
                print("  Undo")

        # Save
        elif key == ord("s"):
            self.td.save_mot(self.out_path)
            self.dirty = False
            print(f"  Saved to {self.out_path}")

    def run(self):
        """Main application loop."""
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)
        self._refresh()

        print(f"\nLoaded {len(self.td.tracks)} tracks, {self.total_frames} frames")
        print(f"Output: {self.out_path}\n")

        while True:
            self._display()
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                if self.dirty:
                    resp = input("\n  Unsaved changes. Save before quit? (y/n): ")
                    if resp.lower() == "y":
                        self.td.save_mot(self.out_path)
                        print(f"  Saved to {self.out_path}")
                break
            elif key != 255:
                self._handle_key(key)

        self.cap.release()
        cv2.destroyAllWindows()

    def _on_mouse(self, event, x, y, flags, param):
        """Mouse callback — placeholder, implemented in Task 5."""
        self.mouse_x = x
        self.mouse_y = y
```

- [ ] **Step 2: Add CLI argument parser**

Replace the `--test-roundtrip` block. Keep the test code but add the real CLI entry point:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOT track correction tool")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--mot", required=True, help="Path to MOT-format track file")
    parser.add_argument("--out", default=None,
                        help="Output path for corrected MOT file (default: <mot>_corrected.txt)")
    parser.add_argument("--test-roundtrip", action="store_true", help="Run data model tests")
    args = parser.parse_args()

    if args.test_roundtrip:
        test_in = "1,1,100.00,200.00,50.00,60.00,0.9500,0,1.00\n1,2,300.00,400.00,70.00,80.00,0.8700,6,1.00\n2,1,105.00,205.00,50.00,60.00,0.9300,0,1.00\n5,1,120.00,220.00,50.00,60.00,0.9100,0,1.00\n"
        Path("/tmp/test_mot_in.txt").write_text(test_in)
        td = TrackData()
        td.load_mot("/tmp/test_mot_in.txt")
        assert len(td.tracks) == 2
        assert td.max_frame == 5
        td.reassign_id(2, 1)
        assert 2 not in td.tracks
        td.undo()
        assert 2 in td.tracks
        td.update_box(1, 1, 110.0, 210.0, 55.0, 65.0)
        assert td.tracks[1][1][0] == 110.0
        td2 = TrackData()
        td2.load_mot("/tmp/test_mot_in.txt")
        td2.interpolate_track(1)
        assert 3 in td2.tracks[1]
        td2.save_mot("/tmp/test_mot_out.txt")
        print("All tests passed.")
        sys.exit(0)

    out_path = args.out or str(Path(args.mot).stem + "_corrected.txt")
    app = App(args.video, args.mot, out_path)
    app.run()
```

- [ ] **Step 3: Test keyboard navigation manually**

Run with any video + a real or synthetic MOT file:

```bash
python annotation_tools/correct_tracks.py \
    --video /path/to/any/clip.mp4 \
    --mot /path/to/clip_mot.txt
```

Verify:
- Video frame displays with boxes and track IDs
- Arrow keys / h,l navigate frames
- H,L skip 10 frames
- Track panel shows on the right with all track IDs and frame ranges
- `q` quits

- [ ] **Step 4: Commit**

```bash
git add annotation_tools/correct_tracks.py
git commit -m "feat: add App class — main loop, keyboard nav, frame display"
```

---

## Task 5: App — Mouse Interaction (select, drag, panel click)

**Files:**
- Modify: `annotation_tools/correct_tracks.py`

Implement the full mouse callback: clicking boxes to select, clicking panel to select + jump, dragging to move/resize boxes.

- [ ] **Step 1: Add hit-test helper to App**

Add this method to the `App` class, before `_on_mouse`:

```python
    def _hit_test(self, x: int, y: int) -> tuple[int | None, str | None]:
        """Find which track box contains (x, y). Returns (track_id, handle).

        handle is "move" if clicking center, "br" if near bottom-right corner (resize).
        """
        entries = self.td.get_frame_tracks(self.frame_idx)
        # Check in reverse so topmost drawn box wins
        for tid, box in reversed(entries):
            bx, by, bw, bh = box[:4]
            # Check bottom-right corner for resize (within 12px)
            if abs(x - (bx + bw)) < 12 and abs(y - (by + bh)) < 12:
                return tid, "br"
            # Check if inside box
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return tid, "move"
        return None, None
```

- [ ] **Step 2: Implement _on_mouse callback**

Replace the placeholder `_on_mouse` method:

```python
    def _on_mouse(self, event, x, y, flags, param):
        """Handle mouse events: select, drag-move, drag-resize, panel click."""
        self.mouse_x = x
        self.mouse_y = y
        frame_w = self.cached_frame.shape[1] if self.cached_frame is not None else 0

        # Click on panel (right side)
        if event == cv2.EVENT_LBUTTONDOWN and x >= frame_w:
            panel_y = y
            tid = self.renderer.panel_track_at_y(panel_y)
            if tid is not None:
                self.selected_tid = tid
                # Jump to first frame of this track
                f_start, _ = self.td.track_frame_range(tid)
                self._goto_frame(f_start)
            return

        # Scroll panel
        if event == cv2.EVENT_MOUSEWHEEL and x >= frame_w:
            if flags > 0:
                self.renderer.panel_scroll = max(0, self.renderer.panel_scroll - 3)
            else:
                self.renderer.panel_scroll += 3
            return

        # Click on frame area
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

        # Drag
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

        # Release drag
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

            # Only update if actually moved
            if abs(dx) > 2 or abs(dy) > 2:
                self.td.update_box(self.drag_tid, self.frame_idx, nx, ny, nw, nh)
                self.dirty = True

            self.dragging = False
            self.drag_tid = None
            self.drag_handle = None
            self.drag_orig_box = None
```

- [ ] **Step 3: Test mouse interaction manually**

```bash
python annotation_tools/correct_tracks.py \
    --video /path/to/clip.mp4 \
    --mot /path/to/clip_mot.txt
```

Verify:
- Click a box on the frame: box gets highlighted (thicker border), selected in panel
- Click a track in the panel: jumps to that track's first frame, selects it
- Drag a box center: moves the box
- Drag bottom-right corner: resizes the box
- Yellow preview box shown during drag
- Release: box updates, marked as keyframe
- Scroll wheel on panel: scrolls track list

- [ ] **Step 4: Commit**

```bash
git add annotation_tools/correct_tracks.py
git commit -m "feat: add mouse interaction — select, drag-move, drag-resize, panel click"
```

---

## Task 6: Integration Test + Documentation

**Files:**
- Modify: `annotation_tools/correct_tracks.py` (minor)
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: End-to-end workflow test**

Generate a MOT file from the tracker, then open it in the correction tool:

```bash
# Step 1: generate lenient tracker output
conda run -n boat-tracker python track/track_video_predict.py \
    --weights weights/best.pt \
    --source /path/to/one/clip/ \
    --out /tmp/test_annotation \
    --conf 0.15 --iou 0.5 --ema-alpha 1.0 \
    --max-coast 30 \
    --enable-nms --nms-iou-thresh 0.5 \
    --save-mot

# Step 2: open in correction tool
python annotation_tools/correct_tracks.py \
    --video /path/to/clip.mp4 \
    --mot /tmp/test_annotation/mot/clip_name.txt \
    --out /tmp/test_annotation/gt/clip_name_gt.txt
```

Test the full workflow:
1. Navigate frames with arrow keys
2. Click a track in the panel — jumps to first appearance
3. Press `r`, type a new ID — verify track is reassigned
4. Drag a box to move it — verify keyframe is created
5. Press `k` to interpolate — verify gaps are filled
6. Press `u` to undo — verify state is restored
7. Press `s` to save — verify output file is valid MOT format
8. Press `q` — verify unsaved changes prompt works

- [ ] **Step 2: Add annotation_tools to README directory structure**

In `README.md`, add to the directory structure:

```
├── annotation_tools/
│   └── correct_tracks.py          # MOT track correction tool (OpenCV-based)
```

- [ ] **Step 3: Add annotation_tools to CLAUDE.md directory structure**

In `CLAUDE.md`, add to the repository layout:

```
├── annotation_tools/
│   └── correct_tracks.py          # MOT track correction tool
```

- [ ] **Step 4: Commit**

```bash
git add annotation_tools/correct_tracks.py README.md CLAUDE.md
git commit -m "feat: MOT correction tool — annotation workflow for tracking GT"
```

---

## Key Bindings Reference

| Key | Action |
|-----|--------|
| `h` / left arrow | Previous frame |
| `l` / right arrow | Next frame |
| `H` (shift+h) | Back 10 frames |
| `L` (shift+l) | Forward 10 frames |
| `<` | Back 30 frames |
| `>` | Forward 30 frames |
| `g` | Jump to selected track's first frame |
| `r` | Reassign selected track to a new ID (terminal prompt) |
| `d` | Delete selected track's box at current frame |
| `x` | Delete entire selected track |
| `k` | Interpolate selected track between keyframes |
| `u` | Undo last edit |
| `s` | Save corrected MOT file |
| `q` | Quit (prompts to save if dirty) |
| Click box | Select track |
| Click panel | Select track + jump to first frame |
| Drag box center | Move box |
| Drag box bottom-right | Resize box |
| Scroll wheel on panel | Scroll track list |
