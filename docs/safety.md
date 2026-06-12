# Safety

**Real-robot safety is mandatory. Never remove safety checks to make
experiments easier.**

## Pre-flight checklist (before running any robot code)

Always verify:

* emergency stop is available
* robot is in a safe workspace
* board and arm base are fixed
* camera feed is live
* action limits are loaded
* gripper limits are loaded
* reset/home pose is valid
* human operator is ready to intervene

## Code-level safety

All real-robot actions must pass through a safety layer.

The safety layer must check:

* joint limits
* action magnitude
* gripper range
* workspace bounds (if available)
* max velocity / max delta
* episode timeout
* NaNs/infs in model output
* stale observations
* camera dropout
* robot disconnects

If a safety check fails:

* stop action execution
* log the failure
* return to a safe state if possible
* do **not** continue the rollout silently

### Implementation status

`safety/safety_layer.py` enforces all of the above **except** workspace bounds:

* **Enforced now:** NaN/inf, action shape, stale observation, camera dropout,
  robot disconnect, episode timeout, and the numeric-magnitude checks — per-joint
  limits, action magnitude, gripper range, and per-step delta / velocity. Each
  numeric check enforces only once its limit is set; while a limit is `<TBD>` the
  check is recorded in `SafetyResult.skipped` (never silently passed) and
  `is_hardware_ready()` reports it.
* **Workspace bounds:** Cartesian, so they need forward kinematics and are **not**
  enforced on a joint-space action. `default_limits.yaml` ships this check
  `enabled: false` (surfaced in `disabled_safety_checks()`, never silent); the
  per-joint limits bound the reachable space. Enable it only alongside an
  FK-based pose check.
* **Hardware gate:** `default_limits.yaml` ships every numeric limit as `<TBD>`,
  so `is_hardware_ready()` is `False` and `SO101Robot.connect()` refuses to
  connect. Fill the values from the real SO-101 and get a safety review (the
  robot-safety-agent) before opening the gate. Filling a value must never disable
  a check.
* **Routing:** every action reaches the arm only through
  `RobotInterface.send_action`, which calls `SafetyLayer.enforce` first
  (fail-closed). `MockRobot` runs the same routing for tests/dry-runs.

## Deployment scripts

* explain architectural changes before editing
* preserve safety checks
* keep old configs unless explicitly migrated
* update README and this doc
* provide a clear change summary

## No unsafe shortcuts

Never:

* disable action clipping without explicit instruction
* bypass emergency stop checks
* bypass workspace bounds
* run untested policies at full action scale
* run residual policy without residual magnitude limits
* execute real-robot scripts without a dry-run mode or clear confirmation path
* silently overwrite datasets or checkpoints

## When to update this doc

Update whenever the safety layer, action limits, robot execution scripts, or
deployment process change.
