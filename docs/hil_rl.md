# Human-In-The-Loop RL

This document expands Phase 3 of the Learning Strategy. It is the **second** RL
solution, implemented after residual policy learning so the two can be
compared.

References:

* "Precise and Dexterous Robotic Manipulation via Human-in-the-Loop
  Reinforcement Learning", arXiv:2410.21845
  (https://arxiv.org/abs/2410.21845)
* LeRobot HIL-SERL docs: https://huggingface.co/docs/lerobot/hilserl

## Status

The exact implementation may vary depending on what works in practice. The
integration with pi0.5 is an open implementation question.

**Do not assume full online fine-tuning of all pi0.5 weights** unless it is
explicitly implemented and tested. Directly updating all pi0.5 weights online
is not automatically practical.

## Possible variants

1. HIL-RL trains a separate policy initialized from or conditioned on pi0.5.
2. HIL-RL trains a residual policy with human interventions.
3. HIL-RL fine-tunes parts of the pi0.5 stack.
4. HIL-RL fine-tunes adapters / LoRA modules rather than full pi0.5 weights.

Build clean interfaces so these variants can be compared against residual
policy learning.

## Required capabilities

HIL-RL must support:

* human intervention
* corrected actions
* success/failure labels
* reward classifier or manual reward
* actor / learner split
* safe abort
* rollout logging
* comparison against residual policy learning

## Compute split

* Learner process and reward-classifier training run on the cloud GPU (the
  vast.ai instance — see `docs/training_pi05.md`).
* The actor / data collection / intervention interface runs on the laptop with
  the SO-101.

## Files

* Config: `configs/policy/hil_rl.yaml`
* Training loop: `src/chess_robot/rl/hil_rl.py`, `scripts/train_hil_rl.py`
* Shared RL pieces: `src/chess_robot/rl/{rewards.py, replay_buffer.py}`
