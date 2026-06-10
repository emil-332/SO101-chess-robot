# Perception Data Collection

How to collect the data that trains and validates the **board perception**
preprocessing. This is **separate** from the LeRobotDataset of
teleoperated robot demonstrations — perception data is just **board
photos + labels**; no robot is needed.

Perception approach: a YOLO-nano
4-corner detector + homography grounding on the **overhead** camera, and a
lightweight per-square CNN on the **side/oblique** camera. Geometry is zero-shot;
piece classification is **few-shot** (~2 starting-position photos per new board).

## The key trick: auto-label from a known position

You do **not** hand-label pieces. Set up a *known* position (a FEN), photograph
it, and the FEN labels all 64 squares. Combined with the homography grounder
(`grid_from_corners`) and the crop extractor (`extract_square_crops`), every
full-board photo yields 64 labeled square crops for free. Hand-work is reduced
to: (a) clicking 4 board corners per camera once per fixed rig/board, and (b)
typing each arrangement's FEN. `camera_utils.labeled_square_crops(image, grid,
board)` does the auto-labeling.

## The three datasets

### A — Corner-detection set (geometry / YOLO-nano; must be zero-shot)

- Camera: **overhead** only.
- Sample = one overhead image + 4 board-corner pixel coords (→ `BoardCorners`).
- Aim for **diversity over volume**: ~**100–200 images** spanning many board
  styles/colours/materials, lighting, backgrounds, and small camera-pose jitter,
  pieces present and absent. Breadth across board *types* is what makes geometry
  zero-shot.
- Annotate with `scripts/annotate_corners.py` (click a1, h1, h8, a8).

### B — Per-square piece-classifier set (CNN; few-shot per board)

- Camera: **side/oblique** (~45–60°) — type and colour are readable at an angle.
- Sample = a cropped square image + label ∈ {`empty`} ∪ {`{white,black}_{pawn,
  knight,bishop,rook,queen,king}`} (13 classes).
- Generate cheaply: calibrate the **side** camera's 4 corners once (fixed rig),
  then for each known-position photo run `labeled_square_crops` →
  ~**20–50 photos** ≈ **1,000–3,000 labeled crops**.
- Oblique-crop nuance: tall pieces lean into the square behind them, so the crop
  is extended **upward** (`top_pad_ratio` in `extract_square_crops` /
  `crop_box`). Tune per rig.
- Few-shot deployment: for any *new* board, take **2 starting-position photos**
  and fine-tune the classifier.

### C — Held-out perception eval set (for 1b.3 numbers)

- Cameras: **overhead + side**, synchronized, of board types **not** used in A/B.
- Sample = a `PerceptionSample` (loaded by `load_perception_samples`): `frames`
  (overhead+side), `ground_truth_board` (from FEN), `ground_truth_grid` (from
  hand-clicked overhead corners), `held_out=true`, and a few `capture_targets`.
- A handful of board types × several positions each, including some capture
  positions. `evaluate_perception` consumes these to produce the real
  `square_grounding / occupancy / piece_classification / capture_detection /
  zero_shot_board_generalization` numbers.

## On-disk layout

```text
perception_data/
  corners/            # Dataset A
    img_0001.jpg
    manifest.jsonl    # written by scripts/annotate_corners.py
  pieces/             # Dataset B (raw photos; crops generated at train time)
    img_0001.jpg
    manifest.jsonl
  eval_heldout/       # Dataset C
    boardX/pos1_overhead.jpg
    boardX/pos1_side.jpg
    manifest.jsonl    # read by eval/perception_metrics.load_perception_samples
```

### Manifest record schemas (JSONL — one JSON object per line)

Corner manifest (A), written by `annotate_corners.py`:

```json
{"image": "corners/img_0001.jpg", "camera": "overhead", "board_type": "woodA",
 "corners": {"a1": [x, y], "h1": [x, y], "h8": [x, y], "a8": [x, y]}}
```

Eval manifest (C), read by `load_perception_samples`:

```json
{"overhead": "eval_heldout/boardX/pos1_overhead.jpg",
 "side": "eval_heldout/boardX/pos1_side.jpg",
 "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
 "overhead_corners": {"a1": [x, y], "h1": [x, y], "h8": [x, y], "a8": [x, y]},
 "board_type": "boardX", "held_out": true, "capture_targets": ["e4", "d5"]}
```

## Collection procedure

1. **Fix the rig:** mount overhead + side cameras, fix the board base, keep them
   stationary. Discover camera indices with `lerobot-find-cameras opencv`; the
   stable keys are `observation.images.overhead` and `observation.images.side`.
2. **Per board type:** click the 4 corners once for each camera (constant while
   nothing moves) via `scripts/annotate_corners.py`.
3. **Vary positions:** arrange several known FENs (start, mid-game, a few
   captures), photograph overhead+side synchronized, record the FEN. Repeat
   across many board styles for A/B; across *new* styles for C.
4. **Vary nuisance factors** for zero-shot: lighting, background, board
   material/colour, piece set, minor camera jitter.
5. **Train on the cloud GPU**, export both models to **ONNX/OpenVINO**, run
   inference on the laptop iGPU.

### Diversity checklist (drives zero-shot quality)

board colour/material · piece set & style · lighting/shadows · background clutter
· small viewpoint variation · occupied vs sparse positions · capture positions.

## Tooling

- `scripts/annotate_corners.py` — click 4 corners, append to the corner manifest
  (needs `pip install -e ".[tools]"` for the GUI).
- `chess_robot.perception.camera_utils` — `corners_from_points`, `crop_box`,
  `extract_square_crops`, `square_label`, `labeled_square_crops`.
- `chess_robot.eval.perception_metrics` — `load_perception_samples`,
  `evaluate_perception`, and the five metrics.
- `chess_robot.chess.board_state.BoardState.from_fen` — FEN -> occupancy.
