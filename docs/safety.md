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

## Deployment scripts

Must:

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
