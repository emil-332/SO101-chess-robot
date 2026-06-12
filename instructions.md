# Instructions

This guide shows the exact commands to teach the chess board reader your own
board and pieces. Run everything from the repo folder, in order. The board reader
is the part that looks at a camera and says which piece sits on each square.

You need Python 3.10 or newer, your chess board with an overhead camera and a side
camera, and the two pictures you took of the board.

## 1. Install the project

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[camera,tools,perception,perception-train]" huggingface_hub
```

On mac or linux use `source .venv/bin/activate` instead of the second line.

## 2. Get the base model

This downloads a model already trained on a public chess set. Your training starts
from it, so you only need a handful of your own pictures.

```powershell
hf download emil-332/chess-piece-cnn-chesscog --local-dir models/piece_cnn_chesscog_2026-06-11
```

## 3. Get your pictures in place

You need one picture from the overhead camera and one from the side camera for
calibration, plus a few side pictures of the board in the full starting position
for training.

If your cameras are plugged in, grab a frame from each. Camera numbers usually
start at 0, so try 0, 1, 2 until each saved picture shows the right view.

```powershell
python scripts/capture_frames.py --camera overhead=0 --camera side=2 --out-dir frames
```

If you already took your pictures by hand, just copy them in. Put one overhead
picture as `frames/overhead.png` and one side picture as `frames/side.png`, and put
your starting position side pictures in a folder called `photos/start`.

## 4. Calibrate the board

A window opens for each picture. Click the four board corners in this order, a1,
h1, h8, a8, and it saves where the board sits in the picture.

```powershell
python scripts/calibrate_corners.py --frame side=frames/side.png --frame overhead=frames/overhead.png
```

## 5. Build the training set from your pictures

The long line of letters is the standard starting position. It tells the program
which piece is on each square in your pictures, so it can label them for you.

```powershell
python scripts/build_piece_manifest.py --images-dir photos/start --camera side --fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR --out manifests/ours.json
python scripts/prepare_piece_dataset.py --source manifest --manifest manifests/ours.json --images-root photos/start --out datasets/piece_ours.npz
```

## 6. Train

This starts from the base model and adjusts it to your pieces. It prints two
accuracy numbers when it finishes and takes a few minutes on a laptop. If you have
a GPU, drop `--device cpu` to use it.

```powershell
python scripts/train_piece_cnn.py --data datasets/piece_ours.npz --init-dir models/piece_cnn_chesscog_2026-06-11 --out-dir outputs/piece_cnn_ours --stage both --device cpu
```

Your trained files land in `outputs/piece_cnn_ours` as `occupancy.onnx` and
`piece.onnx`.

## 7. Use your new model

Open `configs/perception/perception.yaml` and set the two model paths under
`models` to your new files, `outputs/piece_cnn_ours/occupancy.onnx` and
`outputs/piece_cnn_ours/piece.onnx`. Then read the board.

```powershell
python scripts/run_perception.py --overhead frames/overhead.png --side frames/side.png
```

It prints the position it sees as a short line of letters. If that matches your
board, the board reader is trained and working.

## What this does and does not cover

This trains the board reader only. Moving the arm to make a chess move is a later
step that needs teleoperation demos and a trained arm policy, which are not part of
this guide.
