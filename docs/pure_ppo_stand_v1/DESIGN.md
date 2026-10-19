# Pure PPO stand experiment design

## Fixed task

- Task ID and protocol: `pure_ppo_stand_v1` / `stand_offset_v1`.
- Flat MuJoCo ground, no motion command, reset support height `0.23825 m`, nominal leg length `0.18413 m`.
- Initial roll/yaw are zero. Pitch is uniformly sampled in `[-3°, 3°]`; pitch rate is uniformly sampled in `[-0.1, 0.1] rad/s`.
- Episode cap is 1000 policy steps = 8 seconds. Fixed training termination is 25° absolute roll/pitch, chassis/gimbal contact, base height outside `[0.12, 0.45] m`, planar drift beyond `0.40 m`, or numerical failure.
- Training has no pushes, action/observation delay, observation noise, domain randomization, state scheduling, action freeze, course changes, or curriculum.

## Action and observations

Normalized policy action has six values in `[-1,1]` in order `[wheel_left, wheel_right, leg_left_1, leg_left_2, leg_right_1, leg_right_2]`. Wheel scale is `8.0 rad/s`; each leg target is the fixed `q_ref` plus `0.5 rad` times its action, clipped to `±1.2 rad`. Action zero means wheel target zero and leg target `q_ref`.

The PPO policy receives 38 actor values and emits six actions. A 94-value flat Gym observation is used by SB3: the critic sees the same 38 actor values plus 56 privileged simulator values. The ONNX actor input is the same 38-value prefix; action mapping is unchanged.

## Reward

One reward is computed per 8 ms policy step and is not multiplied by `dt`:

```text
upright + alive - pitch_rate - horizontal_velocity - position_drift
         - height_error - leg_length_error - small_action_rate
         - small_effort - terminal_failure
```

The constants are fixed in `StandRewardConfig` and serialized in the run configuration. Raw measurements and weighted contributions are logged. There is no reward term that compares the policy action to a reference action.

## PPO lock

| Setting | Value |
|---|---|
| Actor/Critic MLP | `[256,128,64]` / `[256,128,64]` |
| Activation | ELU |
| Optimizer | Fresh SB3 default Adam |
| Learning rate | `1e-4`, constant |
| Environments | 8 |
| Steps/env/update | 512 |
| Batch size / epochs | 512 / 5 |
| Gamma / GAE lambda | 0.99 / 0.95 |
| Clip range / target KL | 0.2 / 0.015 |
| Entropy coefficient | 0 |
| Initial log standard deviation | -1.5 |
| Seeds | 0, 1, 2, independently initialized |
| Maximum budget | 800 updates per seed; analyze at 400 |

No schedule changes any of these values during a run. Best checkpoint selection is based only on deterministic 50-episode validation at fixed seeds 10000–10049.

## Evaluation and acceptance

Final held-out ID seeds are 20000–20099 for training seed 0, 21000–21099 for seed 1, and 22000–22099 for seed 2. ID test distribution matches training. All success conditions must hold per episode: complete 1000 steps; pitch RMS ≤3°; peak absolute pitch ≤10°; maximum planar drift ≤0.10 m; each wheel contact fraction ≥95%; no chassis contact; current/torque saturation fraction ≤1% and no more than three consecutive saturated policy steps. The seed passes only at ≥95% episode acceptance. Wilson 95% intervals and pitch-sign bins are reported.

Stress modes are reported separately: initial pitch in ±5°, initial pitch rate in ±0.25 rad/s, and an instantaneous ±0.20 rad/s body pitch-rate perturbation at 4 seconds. These are evaluation-only and do not alter training.
