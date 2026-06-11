# Evaluation

This document expands the Evaluation section of `  `. Evaluation must be
standardized so the three approaches can be compared fairly. The full-board /
all-piece-types setting adds perception and capture-specific metrics.

## Perception metrics (validate separately from manipulation)

The learned board-perception model is validated independently, because
downstream deterministic logic trusts its output. Track:

```text
square_grounding_accuracy        # correct square-name -> region mapping
occupancy_accuracy               # correct occupied/empty per square
piece_classification_accuracy    # correct piece type per occupied square
zero_shot_board_generalization   # accuracy on unseen board types
capture_detection_accuracy       # correctly flags target-occupied captures
```

A perception error must be attributable: when a rollout fails, record whether
the cause was perception (`perception_error`) or manipulation. Run perception
validation on held-out board types to measure zero-shot generalization.

## Primary manipulation metrics

```text
success_rate
wrong_square_rate
wrong_piece_rate
mean_target_error_cm
grasp_success_rate
drop_rate
release_failure_rate
collision_rate
intervention_rate
episode_time
safety_violation_rate
```

## Capture / submove metrics

```text
capture_success_rate             # full capture (remove + place) succeeded
removal_submove_success_rate
placement_submove_success_rate
capture_split_correctness        # resolver split when (and only when) needed
```

## Residual-learning-specific metrics

```text
base_policy_success_rate
residual_policy_success_rate
mean_residual_norm
residual_action_saturation_rate
improvement_over_base
```

## HIL-RL-specific metrics

```text
number_of_interventions
intervention_timing
correction_success_rate
reward_curve
success_rate_over_training_time
human_operator_time
```

## Comparisons to implement

```text
pi0.5 supervised
pi0.5 + residual RL
pi0.5 + HIL-RL
```

Report manipulation metrics with perception held fixed (ideally with
metadata-supplied occupancy) so policy comparisons are not confounded by
perception errors, and separately report end-to-end numbers with the perception
model in the loop.

Do **not** add ACT or SmolVLA baselines unless explicitly requested.

## Files

* Metrics: `src/chess_robot/eval/metrics.py`
* Evaluator: `src/chess_robot/eval/evaluator.py`
* Failure labels: `src/chess_robot/eval/failure_labels.py`
* Config: `configs/eval/chess_eval.yaml`
* Script: `scripts/evaluate_policy.py`

Evaluation scripts should be runnable from the laptop (as a client) even when
heavy evaluation jobs run on the cloud GPU.
