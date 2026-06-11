#!/usr/bin/env bash
# Renderer smoke for the two-stage piece CNN on a vast.ai GPU box.
# Verifies prepare -> train -> ONNX export run end to end. Plumbing only: the flat
# synthetic renderer does not teach real piece appearance (use the chesscog
# manifest source for that). See docs/perception_piece_cnn.md.
set -euo pipefail

DATA=${DATA:-datasets/piece_smoke.npz}
OUT=${OUT:-outputs/piece_cnn_smoke}

echo "[1/2] building a tiny rendered crop dataset -> $DATA"
python scripts/prepare_piece_dataset.py --source renderer --num-boards 40 \
    --image-size 240 240 --out "$DATA"

echo "[2/2] smoke-training both stages (1 epoch per phase) -> $OUT"
python scripts/train_piece_cnn.py --data "$DATA" --out-dir "$OUT" --stage both --smoke

echo "smoke done -> $OUT"
ls -la "$OUT"
