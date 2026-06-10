# Architecture

## Design principle: perception is preprocessing, not the VLA's job

The board is read **before** anything reaches pi0.5. A perception preprocessing
stage converts a raw camera image of any 8x8 board into a structured board
state and a preprocessed observation (with the relevant squares highlighted).
pi0.5 then focuses purely on manipulation and never has to learn to read the
board, ground square names, or classify pieces from raw pixels.

Two distinct jobs live in preprocessing, using two different methods:

1. **Learned board perception (zero-shot generalization).** A vision model
   takes a raw board image and outputs the square grid and per-square
   occupancy / piece identity. This is what generalizes zero-shot to unseen
   board types. It runs as preprocessing -- never inside the pi0.5 backbone.
2. **Deterministic move logic.** Given the perceived board state plus the
   parsed instruction, deterministic code resolves source/target squares,
   detects captures, splits capturing moves into submoves, and produces the
   highlighted observation passed to the VLA.

## End-to-end flow

```text
raw camera image                      language command
        |                                     |
board perception (learned, zero-shot)   command parser
        |                                     |
        +--------------+----------------------+
                       |
            move resolver (deterministic):
              - locate source & target squares
              - detect capture (target occupied?)
              - split capturing move into submoves
              - highlight relevant squares on image
                       |
            preprocessed observation builder
                       |
                  pi0.5 policy
                       |
            optional residual RL policy
                       |
                safety / action filter
                       |
              SO-101 robot execution
                       |
                 rollout logging
                       |
        evaluation + feedback + RL updates
```

## Conceptual modules

```text
CommandParser:
  "move knight from b1 to c3"
  -> piece_type="knight", start_square="b1", target_square="c3"

BoardPerception (learned, zero-shot, preprocessing):
  raw board image
  -> square grid (grounded square names -> image regions)
  -> per-square occupancy and piece identity
  Must generalize zero-shot to unseen 8x8 board types.
  Runs OUTSIDE the pi0.5 backbone.

BoardMapper:
  square name + perceived/calibrated grid
  -> image region and/or target board coordinate / pose

MoveResolver (deterministic, preprocessing):
  parsed command + perceived board state
  -> source square, target square
  -> capture? (is the target square occupied)
  -> ordered submoves (see Capture handling)
  -> highlighted squares for the observation

ObservationBuilder:
  raw image + highlighted source/target squares + instruction
  -> preprocessed observation for the VLA

Pi05Policy:
  preprocessed observation + language instruction
  -> base action (manipulation only)

ResidualPolicy:
  observation + base action
  -> delta action

ActionComposer:
  base action + residual action
  -> final action

SafetyLayer:
  clips actions; checks joint limits, workspace bounds,
  emergency-stop state; blocks unsafe execution

RobotInterface:
  sends action to SO-101; reads state, gripper, camera observations

RolloutLogger:
  records observations, actions, rewards, interventions,
  success labels, failure labels, board state, submove index
```

## Capture handling (deterministic split-move)

When the move resolver sees the target square is occupied by another piece (a
capture), it splits the move into two ordered submoves:

```text
submove 1: remove the captured piece from the target square
           (move it to a designated off-board / capture-tray location)
submove 2: move the instructed piece from its source square to the target square
```

Rules:

- This logic is deterministic and lives in preprocessing. It uses the perceived
  (or metadata-supplied) occupancy to decide whether a capture is happening.
- Each submove is a normal manipulation task handed to the VLA in sequence, each
  with its own highlighted observation.
- The off-board capture/tray location is configurable (do not hard-code).
- Log the submove index so rollouts and evaluation can distinguish submove 1
  (removal) from submove 2 (placement).
- This is capture **sequencing**, not collision-aware path planning. Avoiding
  knocking over intermediate pieces along a trajectory remains a Non-Goal

### Square grounding

Primary: a **learned vision model** grounds square names to image regions and
generalizes **zero-shot** to unseen 8x8 board types. It always runs as
**preprocessing** before the VLA, never inside the pi0.5 backbone.

**Chosen approach** a learned **YOLO-nano 4-corner detector** on
the overhead camera locates the board corners; a deterministic homography then
grounds all 64 squares to image regions (`grid_from_corners` in
`perception/square_grounding.py`). The detector is trained on the cloud GPU and
exported to **ONNX/OpenVINO** so perception runs on the laptop's Intel Iris Xe
at record/inference time (no CUDA assumed). Board geometry generalizes zero-shot;
deterministic per-board calibration (supplying the four corners by hand, via
`FixedCornerDetector`) is the bootstrap/fallback while the detector matures.

### Piece identity / occupancy -- primary path

Primary: the system is given the board occupancy (and the
instruction names the piece), which makes capture detection trivial and
deterministic. Keep a clean interface so occupancy can come from either metadata
(bootstrap) or the perception model (target), and they can be cross-checked.

**Chosen approach:** a lightweight **per-square CNN**
(MobileNetV3 / EfficientNet-class) classifies each grounded square as empty or a
piece (type + colour) from the oblique/side camera, trained on the cloud GPU and
exported to ONNX/OpenVINO for the laptop. **Refinement of the earlier strict
zero-shot decision:** board *geometry* generalizes zero-shot, but *piece
classification* is **few-shot** -- the classifier is fine-tuned on ~2
starting-position photos per new board type (cf. chesscog, arXiv:2104.14963).
True zero-shot piece ID remains an optional fallback. This trades a small
per-board calibration step for materially higher accuracy; the metadata path
still covers any board with no photos. See `perception/piece_locator.py`.

### Camera setup

Chosen: **overhead + side/oblique** (two cameras). The **overhead** camera feeds
square grounding (the grid is easiest to detect from above); the **side/oblique**
(~45-60 deg) camera feeds piece classification (type and colour are far easier to
read at an angle than from straight above). The wrist camera is optional and not
used by perception. Camera keys are stable -- never rename silently:

```text
observation.images.overhead   # square grounding
observation.images.side       # piece classification
observation.images.wrist      # optional; unused by perception
```

### Residual conditioning (still TBD)

Whether the residual policy is conditioned on the base action
(`residual_policy(obs, instruction, base_action)`) or only on `obs`/instruction
is an open question. Verify against Silver et al. (arXiv:1812.06298) before
committing.

## Interface and validation notes

- **BoardPerception is a separately-validated sub-module.** If its occupancy
  output is wrong, the deterministic capture logic downstream silently does the
  wrong thing. Validate perception accuracy independently (see `evaluation.md`)
  and support a metadata-supplied occupancy fallback so manipulation work is not
  blocked on perfect perception.
- Keep robot-specific code (`robot/`, `safety/`) separate from learning code
  (`policies/`, `rl/`) and from perception (`perception/`).
- Keep real-robot execution separate from offline training and evaluation.
- When any module boundary, the perception interface, policy composition, robot
  interface, RL setup, or board/piece assumptions change, update this document.

## Perception & preprocessing robustness

Hardening from a pipeline review:

- **Per-camera grounding.** Grounding is per camera (`PerceivedBoard.grids`): the
  overhead grid grounds/highlights squares, and the **side/oblique grid** is what
  the per-square classifier crops from (`ComposedBoardPerception` grounds both and
  passes the side grid to the classifier). Each camera has its own homography.
- **Exact quads, not just AABBs.** Grounding produces the true projected
  quadrilateral per square (`SquareQuad`) plus its axis-aligned bbox
  (`ImageRegion`). Highlighting draws the quad polygon, so adjacent squares don't
  blur together under oblique views. (Per-square *crops* are still rectangular
  slices; perspective-warping each cell is a future improvement.)
- **Board orientation is validated** `BoardCorners` must be given in
  `a1, h1, h8, a8` order and validates a non-degenerate convex quad (catching
  swapped/garbage corners); `BoardCorners.rotated()` relabels for a known board
  rotation. Identifying which physical corner is a1 remains a calibration step
  (auto-detection via a marker/landmark is future work).
- **Resolve captures once.** A capturing move is resolved a single time (while the
  target is still occupied) via `data.lerobot_dataset.plan_move`; each submove's
  observation is built from that fixed plan with `build_observation`. Re-resolving
  per submove is wrong (after removal the target is empty, so the resolver would
  no longer see a capture).
- **Cross-check.** `perception.cross_check_occupancy` reports per-square
  disagreements between two perceptions (e.g. metadata vs the learned model) so
  perception errors are caught, not silently trusted.
- **Loud highlighting.** If highlighting is requested but no camera is grounded,
  a warning is logged (it is never a silent no-op).
- **Board size scope.** The board is **8x8** (chess); NxN boards are out of scope
  (CLAUDE.md). "Zero-shot generalization" means board *style/appearance*, not
  dimensions. Different *physical* board sizes are handled for free — the
  homography is scale-/perspective-invariant.
- **Piece-identity generalization is few-shot, not zero-shot.** Geometry is
  shape-agnostic, but reading piece type/colour from a new piece set needs a
  ~2-photo per-board fine-tune (or metadata). Evaluations must report this.
