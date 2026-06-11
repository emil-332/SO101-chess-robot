# Research References

Grounding references for the project. Use these to verify that implementations
match the intended papers.

## pi0.5 / LeRobot

```text
https://huggingface.co/docs/lerobot/pi05
```

Purpose in this project:

* main pretrained VLA model
* used through LeRobot if possible
* fine-tuned on chess-piece teleoperation data

## Residual Policy Learning

```text
https://arxiv.org/abs/1812.06298
```

Tom Silver et al., "Residual Policy Learning".

Purpose in this project:

* learn an RL residual on top of the pi0.5 base policy
* do not train from scratch
* compare base policy vs. base + residual

Open question to resolve against this paper: whether the residual is
conditioned on the base action or only on the observation/instruction.

## HIL-SERL / Human-In-The-Loop RL

```text
https://arxiv.org/abs/2410.21845
https://huggingface.co/docs/lerobot/hilserl
```

"Precise and Dexterous Robotic Manipulation via Human-in-the-Loop
Reinforcement Learning."

Purpose in this project:

* compare human-in-the-loop RL against residual policy learning
* collect interventions and corrections
* test whether HIL-RL improves real-robot success rate and precision

## Note on verifying against sources

The `research-agent` should be used (read-only) to check these references when
an implementation decision depends on what a paper actually specifies, rather
than relying on memory.
