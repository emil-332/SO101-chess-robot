# Piece classification CNN (two-stage)

This is the learned per-square piece classifier from the perception design
(`docs/architecture.md`): geometry is grounded zero-shot by homography, and piece
identity is read few-shot by a small CNN on the side/oblique camera. It feeds the
structured board state used for capture detection and for cross-checking metadata.
Manipulation (pi0.5) never sees raw pixels of the board for this purpose.

## Why two stages

We follow chesscog (Wölflein & Lange, *Determining Chess Game State From an
Image*, 2021), which beat prior work by splitting the problem:

1. **Occupancy** — binary empty/occupied on a near-square crop.
2. **Piece identity** — 12 classes (`{white,black}_{pawn,knight,bishop,rook,queen,king}`)
   on a taller crop that includes the piece rising above its square.

Splitting keeps the empty/occupied decision (which drives capture detection, a
safety-relevant signal) simple and accurate, and lets the identity head focus on
the harder 12-way problem on occupied squares only. CVChess (2025) shows a single
13-class head also works (~99% in-domain on ChessReD); we kept the two-stage split
per the chosen design and because the occupancy signal stands on its own.

## How it fits the pipeline

Both runtime classifiers implement `PieceClassifier.classify(frames, grid)` and
drop into `ComposedBoardPerception`. The grounded grid (from the corner
detector + homography) gives the 64 square quads; the classifier crops the
configured camera, runs occupancy over all grounded squares, then runs identity
over the occupied ones, and returns a `BoardState`.

```
overhead frame ── corner detector + homography ──► grounded grid
side frame ───────────────────────────────────────┐
                                                   ▼
                            occupancy CNN (per square: empty/occupied)
                                                   ▼
                            piece CNN (per occupied square: 12 classes)
                                                   ▼
                                              BoardState
```

## Modules

Torch-free (run on the laptop):

- `perception/piece_cnn_config.py` — `TwoStageConfig` + YAML loader; per-stage
  backbone, crop input size, and upward crop padding (`top_pad_ratio`).
- `perception/piece_dataset.py` — class orderings, label <-> `Piece` conversion,
  crop extraction (train + inference), numpy bilinear resize, ImageNet
  normalization, prediction-to-`BoardState`, and the `.npz` dataset artifact.
  This is shared by the torch and ONNX paths so they preprocess identically.
- `perception/piece_cnn_onnx.py` — `OnnxTwoStageClassifier` (onnxruntime), the
  laptop inference path for the exported models.

Torch (run on the cloud GPU, `perception-train` extra):

- `perception/piece_cnn.py` — torchvision backbones with a fresh head, two-phase
  fine-tune (head only, then the whole network), evaluation, ONNX export, and
  `TorchTwoStageClassifier`.

Scripts:

- `scripts/prepare_piece_dataset.py` — build a crop dataset.
  - `--source renderer`: random boards from the synthetic renderer. No extra data,
    runs on the laptop. Validates the prep -> train plumbing only; the flat
    renderer does not teach real piece appearance.
  - `--source manifest`: a JSON list of labelled photos
    (`{"image", "corners", "fen"}`). This is the path for the chesscog synthetic
    set and, later, our own captured boards.
- `scripts/train_piece_cnn.py` — train the stage(s). `--dry-run` is torch-free and
  prints the plan; `--smoke` runs 1 epoch per phase for a cheap end-to-end check.

Config: `configs/perception/piece_cnn.yaml`.

## Backbone choice (deviation from chesscog)

chesscog uses ResNet for occupancy and InceptionV3 for identity. We default to
ResNet for both (occupancy `resnet18`, piece `resnet34`) because it exports to
ONNX cleanly and trains cheaply within the ~$100 GPU budget. The backbone is a
config key (`resnet18`, `resnet34`, `mobilenet_v3_small`, `efficientnet_b0`), so
InceptionV3 can be added if exact replication is wanted. This is the only
deviation from the cited recipe.

## Data plan

1. **Pretrain on chesscog synthetic.** The chesscog dataset provides rendered
   boards with per-image corner and FEN labels, which map directly to the
   `manifest` source. Pretraining gives a strong prior for "a chess piece on a
   square".
2. **Few-shot fine-tune on our set.** Our pieces are 3D-printed, so they will not
   match chesscog's Staunton pieces. Photograph the starting position from the
   side camera a few times, cut 64 auto-labelled crops per photo from the known
   FEN, augment, and fine-tune. This is the chesscog transfer trick and matches
   the few-shot decision in `docs/architecture.md`.
3. (Optional) Because we own the piece meshes, we can render our exact pieces in
   Blender for a larger synthetic set of the real geometry, reducing the number of
   real photos needed.

Steps 2 and 3 need captured data or meshes and are not implemented yet. Everything
in steps 1's tooling (prep, train, export, inference, evaluation) is implemented
and tested.

## Cloud runbook (vast.ai)

Rent an RTX 4090/5090 box (same as the pi0.5 smoke), clone the repo, then:

```bash
pip install -e ".[perception-train]"   # torchvision + onnx (torch from the image)
```

### Step 1: renderer smoke (verify the torch path, minutes, ~cents)

```bash
bash scripts/piece_cnn_smoke.sh
```

This builds a tiny rendered crop set and runs `train_piece_cnn.py --smoke` (1 epoch
per phase, both stages). Success = two checkpoints + two ONNX files under
`outputs/piece_cnn_smoke` and finite val accuracy printed. The flat renderer does
not teach real appearance; this only confirms prep -> train -> export runs.

### Step 2: download a split and verify orientation

```bash
pip install osfclient
osf -p xf3ka list   # -> osfstorage/{train.zip,train.z01,val.zip,test.zip}

# val first (small): unzips to a val/ dir of PNG + JSON
osf -p xf3ka fetch osfstorage/val.zip val.zip && unzip val.zip

# convert + orientation preview
python scripts/chesscog_to_manifest.py --chesscog-dir val \
    --out manifests/val.json --preview 4
#   ^ inspect manifests/preview/ FIRST. Each `*_<square>_<piece>.png` crop must
#     show the named piece (e.g. *_e1_white_king.png shows a white king). If a crop
#     shows the wrong piece, the corner->square mapping is flipped: stop and report.
```

### Step 3: train

```bash
# quick real-data check: train on val with an internal split (minutes, no big download)
python scripts/prepare_piece_dataset.py --source manifest \
    --manifest manifests/val.json --images-root val --out datasets/piece_val.npz
python scripts/train_piece_cnn.py --data datasets/piece_val.npz \
    --out-dir outputs/piece_cnn_val --stage both

# full pretrain: download + merge the train split, convert, build, train, eval on val
osf -p xf3ka fetch osfstorage/train.zip train.zip
osf -p xf3ka fetch osfstorage/train.z01 train.z01
zip -s 0 train.zip --out train_full.zip && unzip train_full.zip   # -> train/
python scripts/chesscog_to_manifest.py --chesscog-dir train --out manifests/train.json
python scripts/prepare_piece_dataset.py --source manifest \
    --manifest manifests/train.json --images-root train --out datasets/piece_train.npz
python scripts/train_piece_cnn.py --data datasets/piece_train.npz \
    --val-data datasets/piece_val.npz --out-dir outputs/piece_cnn --stage both
```

`--val-data` evaluates on the separate val set (honest held-out accuracy); without
it the script uses an internal split of `--data`. The full train set is large
(~4,400 images -> a multi-GB `.npz` built in memory); if prepare runs out of RAM,
cap it with `--limit` on the converter or build in chunks. Free disk after
unzipping by deleting `train.zip`, `train.z01`, `train_full.zip`.

Outputs: `occupancy.pt` / `piece.pt` (checkpoints) and `occupancy.onnx` /
`piece.onnx` (for the laptop). Copy the ONNX models to the laptop, store them under
`models/<name>/` (kept out of git, see `models/README.md`), and point the
perception config at them.

### Step 4 (later, needs our data): few-shot fine-tune on the 3D-printed set

This adapts the chesscog base to the real pieces. It warm-starts from the chesscog
checkpoint instead of ImageNet, so a handful of photos is enough.

```bash
# 1) photograph the starting position from the side camera (a few boards),
#    label corners with scripts/annotate_corners.py, write a small manifest
#    (image, corners, fen), then build crops:
python scripts/prepare_piece_dataset.py --source manifest \
    --manifest manifests/ours.json --images-root photos/ --out datasets/piece_ours.npz

# 2) fine-tune from the chesscog base (--init-dir), at a lower learning rate
#    (set head_lr/full_lr lower in configs/perception/piece_cnn.yaml):
python scripts/train_piece_cnn.py --data datasets/piece_ours.npz \
    --init-dir models/piece_cnn_chesscog_2026-06-11 \
    --out-dir outputs/piece_cnn_ours --stage both
```

Then copy the new `*.onnx` into `models/<name>/` and update
`configs/perception/perception.yaml` to point at them. Nothing else changes.

## Running it in the pipeline

`configs/perception/perception.yaml` wires the trained classifier and the board
grounding into a `BoardPerception` (the factory is
`chess_robot.perception.pipeline.build_board_perception`). The config holds the
model paths and the per-camera board-corner calibration; swapping models or
recalibrating is a config edit, no code change.

```bash
pip install -e ".[perception]"     # laptop: onnxruntime + pillow
# calibrate the board corners into the config first (annotate_corners.py), then:
python scripts/run_perception.py --overhead frames/overhead.png --side frames/side.png
```

It prints the perceived position as FEN. `--metadata-fen <placement>` cross-checks
the reading against a known position. The config ships uncalibrated (null corners)
and the pipeline refuses to run an uncalibrated piece camera rather than guess, so
fill `calibration.side` (and optionally `calibration.overhead`) once the board is
built. The trained-model wiring itself is verified: the exported ONNX models load
and run through this factory on the laptop with onnxruntime.

### Troubleshooting

- `ModuleNotFoundError: No module named 'onnxscript'` at export: recent torch
  routes `torch.onnx.export` through the dynamo exporter, which needs onnxscript.
  It is in the `perception-train` extra; if the image predates that, run
  `pip install onnxscript` and re-run. (Training checkpoints `*.pt` are saved
  before export, so only the `*.onnx` files are missing.)
- ImageNet backbone weights fail to download: set `pretrained: false` in
  `configs/perception/piece_cnn.yaml` (fine for a plumbing smoke; for the real run
  keep it true).

## Evaluation

The per-square and board-level metrics already exist in
`eval/perception_metrics.py` (`evaluate_perception`). Track empty-vs-occupied
accuracy specifically: it drives capture detection and therefore the remove/place
submove split.
