# MOT Track Correction Tool

OpenCV-based tool for correcting tracker output into ground truth annotations. Loads a video + MOT-format track file, lets you fix track IDs and bounding boxes, and exports corrected GT.

## Workflow

### Step 1 — Generate baseline tracks with lenient settings

Run the tracker with low confidence and high max-age to over-detect. The goal is to capture all objects even at the cost of false positives — it's easier to delete wrong tracks than annotate from scratch.

```bash
conda run -n boat-tracker python track/track_video_predict.py \
    --weights weights/best.pt \
    --source data/eval/clips \
    --out results/annotation_baseline \
    --conf 0.15 --iou 0.5 --ema-alpha 1.0 \
    --max-coast 30 \
    --enable-nms --nms-iou-thresh 0.5 \
    --save-mot
```

This produces MOT text files in `results/annotation_baseline/mot/`.

### Step 2 — Open in correction tool

```bash
conda run -n boat-tracker python annotation_tools/correct_tracks.py \
    --video data/eval/clips/clip_001.mp4 \
    --mot results/annotation_baseline/mot/clip_001.txt \
    --out data/eval/gt/clip_001/gt.txt
```

### Step 3 — Review and correct

The tool displays the video with bounding boxes and a track panel on the right.

**Typical correction workflow:**

1. Scroll through the track panel to find all track IDs
2. Click a track in the panel — jumps to its first frame
3. Scrub forward with `l`/`L` to follow the track visually
4. If the track switches ID mid-way (fragmentation), select the wrong ID and press `r` to reassign it to the correct one
5. If a bounding box is wrong, drag it to reposition or resize from the bottom-right corner
6. If you correct boxes at sparse keyframes, press `k` to linearly interpolate between them
7. Delete false positive tracks with `x`, or individual boxes with `d`
8. Press `s` to save periodically

### Step 4 — Output

The corrected file is saved in MOTChallenge format at the `--out` path:

```
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class_id>,<visibility>
```

Place GT files in the structure expected by `eval/eval_tracking.py`:

```
data/eval/gt/
├── clip_001/
│   └── gt.txt
└── clip_002/
    └── gt.txt
```

## Key Bindings

| Key | Action |
|-----|--------|
| `h` / `l` | Previous / next frame |
| `H` / `L` | Back / forward 10 frames |
| `<` / `>` | Back / forward 30 frames |
| `g` | Jump to selected track's first frame |
| `r` | Reassign selected track to a new ID (terminal prompt) |
| `d` | Delete selected track's box at current frame |
| `x` | Delete entire selected track |
| `k` | Interpolate selected track between keyframes |
| `u` | Undo last edit (up to 50 levels) |
| `s` | Save corrected MOT file |
| `q` | Quit (prompts to save if unsaved changes) |

## Mouse

| Action | Effect |
|--------|--------|
| Click box on frame | Select that track |
| Click track in panel | Select + jump to first frame |
| Drag box center | Move box |
| Drag bottom-right corner | Resize box |
| Scroll wheel on panel | Scroll track list |

Dragging a box marks that frame as a keyframe. Use `k` to interpolate between keyframes.

## Self-Test

Run data model tests (no video or cv2 display needed):

```bash
python annotation_tools/correct_tracks.py --test
```

## Dependencies

No additional dependencies — uses OpenCV and numpy from the `boat-tracker` conda environment.
