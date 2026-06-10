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

## Train on the cloud GPU

```bash
# on the vast.ai box, in the repo
pip install -e ".[perception-train]"   # torchvision + onnx (torch from the image)

# 1) build a crop dataset (manifest = chesscog synthetic, or renderer for a smoke)
python scripts/prepare_piece_dataset.py --source manifest \
    --manifest labels.json --images-root images/ --out datasets/piece.npz

# 2) train both stages and export ONNX
python scripts/train_piece_cnn.py --data datasets/piece.npz \
    --out-dir outputs/piece_cnn --stage both
```

Outputs: `occupancy.pt` / `piece.pt` (checkpoints) and `occupancy.onnx` /
`piece.onnx` (for the laptop). Copy the ONNX models to the laptop and load them
with `OnnxTwoStageClassifier.from_paths(config, occ_onnx, piece_onnx)`.

## Evaluation

The per-square and board-level metrics already exist in
`eval/perception_metrics.py` (`evaluate_perception`). Track empty-vs-occupied
accuracy specifically: it drives capture detection and therefore the remove/place
submove split.
