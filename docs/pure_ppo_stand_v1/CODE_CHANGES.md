# Code changes for Pure PPO standing

This record lists changes made for this task. Existing worktree edits and
untracked user files were preserved; they are not included in these task
commits.

## Physics and state correctness

- In `scripts/uz05/model.py`, rotate free-joint local angular velocity into
  world coordinates before exposing `base_ang_vel_world`.
- In `scripts/uz05/env.py`, derive pitch rate once in the body frame and fix
  the `dof_acc` difference to use newest minus oldest velocity.
- Add physical preflight coverage for reset closure, wheel contacts, body
  contacts, wheel action signs, current and torque, zero-action leg torque,
  observation frames, and control-rate timing.

## Pure PPO task and contracts

- Add `scripts/uz05/pure_ppo_stand.py` with the fixed flat-ground task, six
  direct actor actions, zero-action `q_ref`, 38-value actor observation,
  94-value actor-plus-critic observation, fixed reward, and no randomization,
  noise, delay, curriculum, or action helper.
- Add `scripts/uz05/pure_ppo_contract.py` and
  `scripts/uz05/pure_ppo_eval.py` for versioned action/observation semantics,
  deterministic selection, held-out acceptance, and three separate stresses.
- Add action, observation, reward, reset, and ONNX actor-parity tests under
  `tests/test_pure_ppo_*.py`.

## Training and diagnostics

- Add baseline preflight, random-action smoke, PPO dry run, formal training,
  deterministic validation, held-out evaluation, actuator tracing, and
  conditional ONNX export entrypoints in `scripts/`.
- Add a fresh `0.35 rad` leg-scale training ablation and a fixed-checkpoint
  paired scale sweep. Their records remain diagnostic when they do not pass.
- Add a separate fresh `position_drift=0.50` training ablation at fixed
  `0.35 rad` action scale, plus independent evaluation for all three seeds.
  The seed-0 policy passes 100/100 ID episodes; seeds 1 and 2 do not reach the
  95% threshold, so the final ONNX exporter was not run.
- Add a local `.gitignore` scope for experiment artifacts, run logs, and
  checkpoints while leaving unrelated existing output directories intact.

## Reports and experiment evidence

- `BASELINE.md` records original commit, runtime, model, audit findings, and
  known deployment-estimation limits.
- `DESIGN.md` records the frozen task, action, reward, PPO, and acceptance
  contracts.
- `TRAINING_REPORT.md` records measured preflight, test, training, evaluation,
  stress, and failure results.
- `ABLATIONS.md` and `REPRODUCE.md` record the paired diagnostics, follow-up
  runs, commands, and retained artifact paths.

All training results are saved locally under the experiment-specific
`runs/`, `checkpoints/`, and `artifacts/` directories. These binary outputs
are intentionally not staged as source commits.
