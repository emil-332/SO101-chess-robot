# Data Collection & Dataset Design

This document expands the Dataset Design section of `  `.

Use **LeRobotDataset** format throughout.

Teleoperation is performed on a **full 8x8 board with all chess pieces
present**, across **all piece types** (pawn, knight, bishop, rook, queen,
king). Demonstrations are of the manipulation skill on **preprocessed
observations** (board already perceived, source/target squares highlighted --
see `architecture.md`).

Initial dataset contains **only successful teleoperation demonstrations**.
Failed attempts, autonomous failures, and human interventions are stored
**separately** for RL/HIL data.

## Natural-language task format

Constrained sentence pattern, now over all piece types:

```text
move {piece_type} from {start_square} to {target_square}
```

Examples:

```text
move knight from b1 to c3
move pawn from e2 to e4
move bishop from c1 to f4
move queen from d1 to h5
```

`piece_type` is one of: pawn, knight, bishop, rook, queen, king.

## Captures and submoves

A move whose target square is occupied is a **capture**. The deterministic move
resolver splits it into two submoves (see `architecture.md`):

```text
submove 1: remove the captured piece to the off-board capture location
submove 2: place the instructed piece on the target square
```

Each submove is recorded as its own manipulation segment with a `submove_index`
and `submove_role` (`remove` or `place`). Teleoperate and record both submoves
when collecting capture demonstrations. Non-capturing moves have a single
segment.

## Dataset progression

```text
Stage 1:
  full board present, single piece type, fixed move
  goal: smoke-test full-board data collection + preprocessing + pi0.5 training

Stage 2:
  full board, single piece type, multiple target squares
  goal: test target-square conditioning with distractor pieces present

Stage 3:
  full board, multiple piece types, multiple source/target squares,
  including captures (split submoves)
  goal: generalization across pieces, squares, and capture handling

Stage 4:
  autonomous rollouts, success/failure labels, human interventions,
  corrected actions, reward annotations
  goal: RL fine-tuning
```

## Episode schema

Each episode should contain, where available:

```text
instruction: natural-language command
observation.images.*           # preprocessed (highlighted) observation
observation.state
action
timestamp
episode_index
piece_type
start_square
target_square
board_state                    # per-square occupancy / piece identity
is_capture                     # bool
submove_index                  # 0 for single-segment moves
submove_role                   # "move" | "remove" | "place"
captured_piece_type            # if is_capture, else null
success_label
failure_type
intervention_flag
corrected_action
reward
```

Natural-language instruction is **mandatory**. `board_state` and the
capture/submove fields are strongly recommended even when the model consumes
only the preprocessed image + instruction, because evaluation and the
deterministic resolver depend on them.

## Supervised dataset rules (pi0.5 fine-tuning)

* Use only successful demonstrations.
* Keep camera setup fixed; keep robot base fixed.
* Board *type* may vary (the perception model is zero-shot), but record which
  board was used so perception can be validated separately.
* Keep task instruction accurate; keep episode/submove structure consistent.
* Remove corrupted or failed demos.
* Do **not** mix failed rollouts into the supervised demonstration dataset.
* Store dataset documentation with each dataset (including board type and
  whether occupancy was metadata-supplied or perception-derived).

## RL/HIL dataset rules

* Store autonomous rollouts separately.
* Include failure labels, intervention labels, corrected actions, safety-
  violation labels, scalar reward (if available), terminal success/failure
  labels, and the submove index/role.

Recommended feedback object:

```python
@dataclass
class Feedback:
    success_label: bool | None
    scalar_reward: float | None
    intervention_flag: bool
    corrected_action: Any | None
    safety_violation_flag: bool
    failure_type: str | None
    submove_index: int = 0
    preference_label: Any | None = None
```

Failure types use stable names (extended for the full-board / capture setting):

```text
bad_grasp
missed_piece
dropped_piece
wrong_square
wrong_piece
release_failure
collision
unsafe_motion
capture_removal_failure
timeout
perception_error
camera_failure
unknown
```

## Collection tooling

Dataset collection is built around LeRobot recording tools or compatible
wrappers, with the perception preprocessing applied so recorded observations
match what the policy will see at inference. Do not assume exact robot ports --
use placeholders (`<FOLLOWER_PORT>`, `<LEADER_PORT>`, `<CAMERA_INDEX>`,
`<HF_USER>`, `<DATASET_REPO_ID>`).

```bash
lerobot-find-port
lerobot-find-cameras opencv
```
