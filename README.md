# vla-for-chess

This project teaches an SO-101 robot arm to move chess pieces on a full 8x8
board from natural-language commands such as `move knight from b1 to c3`. The
board is read by a separate perception stage before anything reaches the policy,
so the policy only has to handle manipulation. A pi0.5 vision-language-action
model, fine-tuned through LeRobot, produces the arm actions. The research aim is
to compare two ways of improving that base policy after supervised training:
residual policy learning and human-in-the-loop reinforcement learning.

The repository currently contains the full deterministic core, the perception
interfaces with a working geometry path, the dataset and evaluation tooling, the
safety layer, and the pi0.5 training and inference wrappers. The pi0.5 training
integration has been run end-to-end on a rented GPU. The learned perception
models and the reinforcement-learning phases are the remaining work, and both
depend on data collection.

## 1) Architecture

The system runs as a pipeline. Perception happens first and produces a structured
board state. Deterministic logic then resolves the move. The policy receives a
preprocessed observation and outputs actions, which pass through a safety filter
before the robot executes them.

```text
raw camera frames        natural-language command
        |                          |
 board perception           command parser
 (grid + occupancy)              |
        +-------------+-----------+
                      |
            move resolver (deterministic):
              locate source and target squares,
              detect captures, split a capture into
              remove-then-place submoves,
              mark the relevant squares
                      |
          observation builder (highlighted frames)
                      |
                 pi0.5 policy
                      |
          optional residual policy + action composer
                      |
              safety layer (fail-closed)
                      |
                SO-101 execution
                      |
            rollout logging and evaluation
```

Key design points:

- Perception is a preprocessing step that sits outside the pi0.5 backbone. A
  learned vision model reads the square grid and the per-square occupancy and
  piece identity. The policy never learns to read the board.
- Two cameras. The overhead camera grounds the squares. The side or oblique
  camera reads piece identity, where type and color are easier to see.
- Captures are deterministic. When the target square is occupied, the resolver
  splits the move into two submoves: remove the captured piece to an off-board
  location, then place the instructed piece. This is move sequencing, and it does
  not plan collision-free paths through other pieces.
- Every real-robot action passes a safety layer. The layer is fail-closed and
  exposes a hardware-readiness gate that stays closed while limits are unset.
- Compute is split. The laptop handles the robot, teleoperation, data recording,
  evaluation, and the inference client. A rented vast.ai GPU handles training.
  The laptop has no usable CUDA, so perception models run there through
  ONNX/OpenVINO once trained.

## 2) Repository layout

```text
configs/                      experiment configs (robot, camera, task, dataset,
                              policy, safety, eval)
scripts/                      entry points and tooling
  collect_demos.py            record teleoperated demos into a LeRobotDataset
  inspect_dataset.py          dataset inspection (planned)
  annotate_corners.py         click 4 board corners for calibration
  verify_pipeline.py          run mock data through every offline stage
  evaluate_perception_synthetic.py   perception accuracy on rendered boards
  make_smoke_dataset.py       tiny mock LeRobotDataset (cloud)
  cloud_setup.sh              vast.ai box bootstrap for the pi0.5 smoke
  train_pi05.py               build and run the lerobot-train command
  run_policy.py               policy client (mock or remote server)
  collect_rollouts.py, train_residual_rl.py, train_hil_rl.py   RL (planned)
  evaluate_policy.py          metrics report from rollouts
src/chess_robot/
  chess/      command_parser, board_state (with FEN), board_mapper (quads),
              move_resolver (capture split-moves)
  perception/ board_perception, square_grounding (homography), piece_locator,
              camera_utils (crops, highlighting), board_renderer (synthetic)
  data/       schema, validation, lerobot_dataset (wrapper + preprocessing),
              synthetic (mock data + pipeline verification)
  policies/   pi05_policy (train config, command builder, inference wrappers)
  rl/         residual_learning, hil_rl, rewards, replay_buffer (planned)
  robot/      so101_interface, observations, actions (planned)
  safety/     safety_layer, limits
  eval/       metrics, evaluator, failure_labels, perception_metrics
  utils/      logging (rollout logger), config
tests/                        unit and integration tests
docs/                         architecture, data collection, perception data,
                              safety, training, RL, evaluation, research, cloud
```

## 3) Models

Perception uses two learned models, with deterministic geometry between them.

- Square grounding. A YOLO-nano detector finds the four board corners on the
  overhead image, and a homography maps every square to an image region. The
  geometry generalizes across boards without retraining. The homography and a
  hand-calibration fallback are implemented and tested. The corner detector is
  the training follow-up and needs an annotated dataset.
- Piece classification. A lightweight per-square CNN reads occupancy and piece
  identity from the side camera. This part is few-shot, fine-tuned on about two
  starting-position photos per new board. The interface is implemented; the CNN
  needs data. A metadata path supplies occupancy in the meantime, and a synthetic
  color-based classifier validates the geometry and crop pipeline.

The policy is pi0.5, a vision-language-action model from Physical Intelligence,
used through LeRobot and fine-tuned on teleoperated demonstrations. The smoke run
trained the action expert (about 693M of the model's 4B parameters) while
freezing the vision-language backbone. The residual policy and the HIL-RL variant
sit on top of the frozen base and are planned.

Status summary:

- Implemented and tested: the deterministic chess core, perception interfaces and
  homography grounding, the metadata and synthetic perception paths, the dataset
  wrapper, the safety layer skeleton, the evaluation harnesses, the rollout
  logger, and the pi0.5 training and inference wrappers.
- Verified on a GPU: the pi0.5 fine-tuning integration ran two steps and saved a
  checkpoint with LeRobot 0.5.2.
- Not yet trained: the corner detector, the piece CNN, and a real pi0.5
  fine-tune. The residual and HIL-RL phases are skeletons.

## 4) Research

The approach builds on the following work.

- pi0.5 (Physical Intelligence), used through LeRobot. The pretrained base is
  `lerobot/pi05_base`. Reference: https://huggingface.co/docs/lerobot/pi05
- Residual Policy Learning, Silver et al., arXiv:1812.06298. A small residual
  action is learned on top of the frozen base controller.
- Human-in-the-Loop reinforcement learning (HIL-SERL), arXiv:2410.21845, with
  the LeRobot HIL-SERL guide. Used as the second comparison point.
- Chess-from-image perception, in particular chesscog (Wölflein and Lange, 2021),
  which trains on rendered boards and transfers to real ones. This informs the
  corner-detection and per-square-classifier design and the use of synthetic
  rendering.

A perception design decision came out of this review. Board geometry is treated
as zero-shot, while piece identity is treated as few-shot, since reading
arbitrary piece styles from pixels is the harder part. The full discussion lives
in `docs/architecture.md` and `docs/research.md`.

## 5) Pipeline example

Take a capturing move on the starting position: `move rook from a1 to a8`, where
a8 holds a black rook.

1. Parse the command. `parse_command` returns piece `rook`, source `a1`, target
   `a8`.
2. Read the board. `perceive(frames)` returns a `PerceivedBoard` with the
   occupancy, the grounded grid per camera, and a source tag (metadata or
   perception).
3. Resolve the move once. `MoveResolver.resolve` sees that a1 holds a rook and
   that a8 is occupied, so it marks the move as a capture and produces two
   submoves: remove the black rook from a8 to the off-board tray, then place the
   white rook from a1 to a8.
4. Build each submove's observation. For each submove, the relevant squares are
   highlighted on the grounded cameras and the structured record is written. The
   move is resolved once and the submoves are built from that fixed plan, so the
   placement step does not get confused after the captured piece is removed.
5. Run the policy. pi0.5 takes the highlighted observation and the instruction
   and returns a base action. A residual policy can add a bounded correction.
6. Filter and execute. The safety layer checks the action and either passes it or
   blocks it. The SO-101 performs the pick and place for that submove.
7. Log and evaluate. The rollout logger records observations, actions, rewards,
   interventions, success or failure, and the submove index and role. The
   evaluation harness aggregates these into metrics.

`scripts/verify_pipeline.py` runs this whole chain on mock data with no hardware
and reports a pass or fail for each stage.

## 6) Setup

Requires Python 3.10 or newer. Developed on 3.12.

Local development and testing (laptop, no GPU needed):

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Verification (run after changes):

```bash
ruff check .
mypy
pytest -q
```

Offline checks that need no hardware:

```bash
python scripts/verify_pipeline.py                 # full pipeline on mock data
python scripts/evaluate_perception_synthetic.py   # perception on rendered boards
python scripts/train_pi05.py --smoke --dry-run     # print the training command
```

Cloud training (vast.ai GPU). The laptop has no usable CUDA, and pi0.5 is too
large to train there, so fine-tuning runs on a rented RTX 4090 or 5090. The only
credential is a Hugging Face read token, set on the box and never committed. Full 
instructions, including how to accept the gated PaliGemma license, are in 
`docs/cloud_smoke_test.md`. The short version, run on the instance:

```bash
export HF_TOKEN=hf_your_read_token
bash scripts/cloud_setup.sh
```

Hardware data collection uses the LeRobot recording tools. Discover ports and
cameras with `lerobot-find-port` and `lerobot-find-cameras opencv`, and keep
lab-specific values in the config files as placeholders such as `<FOLLOWER_PORT>`
and `<CAMERA_INDEX>`. See `docs/data_collection.md` and
`docs/perception_data_collection.md`.


## TODO NEXT 

read
https://universe.roboflow.com/chessred-vc-task/chessred-pnvwd/dataset/9 / https://github.com/tmasouris/end-to-end-chess-recognition

CVChess: A Deep Learning Framework https://arxiv.org/pdf/2511.11522

other dataset: https://osf.io/xf3ka/wiki

Create 3D chess set. 
    piece set: https://makerworld.com/de/models/406463-dubrovnik-1960-bobby-fischer-chess-set?from=search#profileId-308422
    board: https://makerworld.com/de/models/544967-modular-chess-board?from=search#profileId-473512

