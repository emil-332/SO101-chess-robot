# Residual Policy Learning

Reference: Tom Silver et al., "Residual Policy Learning", arXiv:1812.06298
(https://arxiv.org/abs/1812.06298).

## Concept

```python
base_action = pi05_policy(obs, instruction)
delta_action = residual_policy(obs, instruction, base_action)
final_action = safety_layer(base_action + delta_action)
```

The residual policy learns **small corrections** on top of the frozen pi0.5
base.

> **Open question (TBD):** whether the residual is conditioned on the base
> action or only on `obs`/instruction. Verify against Silver et al. before
> committing. See `architecture.md`.

## Implementation principles

* Keep pi0.5 frozen initially.
* Train a lightweight residual policy.
* Residual output must be action-bounded.
* Residual must be clipped before execution.
* Log base action, residual action, and final action separately.
* Residual policy must never bypass the safety layer.
* Start residual magnitude small.
* Make residual scale configurable.
* Support disabling the residual policy through config.

## Required logs

```text
base_action
residual_action
final_action
residual_norm
reward
success_label
failure_type
safety_violation_flag
```

## Files

* Config: `configs/policy/residual_rl.yaml`
* Composition logic: `src/chess_robot/policies/{residual_policy.py, action_composer.py}`
* Training: `src/chess_robot/rl/residual_learning.py`, `scripts/train_residual_rl.py`
* Rollout collection: `scripts/collect_rollouts.py`
