# models/

Trained model weights live here. They are **not** committed to git (see the
repo `.gitignore`); only this README is tracked. Keep the actual `.pt` / `.onnx`
files out of version control and back them up separately (laptop folder, the
Hugging Face Hub, etc.).

## Layout

One subdirectory per trained model, named with the source and date:

```
models/
  piece_cnn_chesscog_2026-06-11/      # two-stage piece classifier, chesscog pretrain
    occupancy.onnx   occupancy.pt     # occupancy stage (ResNet-18)
    piece.onnx       piece.pt         # piece-identity stage (ResNet-34)
```

The `.onnx` files are what the laptop runs at inference (via onnxruntime); the
`.pt` files are the PyTorch checkpoints, used to warm-start the few-shot
fine-tune on the cloud GPU.

## How the pipeline finds these

`configs/perception/perception.yaml` points to the model files by path. To swap
in a fine-tuned model, change those paths (or drop the new files in a new
subdirectory and update the config). See `docs/perception_piece_cnn.md`.

## Provenance of `piece_cnn_chesscog_2026-06-11`

Two-stage piece classifier pretrained on the chesscog synthetic dataset
(OSF 10.17605/OSF.IO/XF3KA). Held-out chesscog validation accuracy: occupancy
0.9994, piece 0.9963 per square. These are in-distribution numbers on Staunton
pieces; the few-shot fine-tune on our 3D-printed set is what adapts it to the
real board.
