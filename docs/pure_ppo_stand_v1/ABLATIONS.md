# Pure PPO stand failure analysis and ablations

## Seed 0 actuator diagnosis

The formal `pure_ppo_stand_v1` seed 0 best checkpoint passed posture, contact,
drift, and survival thresholds but had `0/100` accepted ID episodes because of
actuator saturation. The step trace at
`runs/pure_ppo_stand_v1/seed0/test_trace/step_trace.csv.gz` contains 100,000
held-out policy steps and logs targets, pre/post wheel speeds, currents,
torques, joint positions/velocities, contacts, and body state.

On leg joint 4, `q_target - q_actual_before_step` averaged `-0.09769 rad`; on
torque-saturated steps it averaged `-0.12348 rad`. At `kp=100`, `kd=4`, those
steps had mean proportional demand `-12.35 Nm` and mean damping demand
`-8.89 Nm`, clipping the requested `-21.24 Nm` to the configured `-20 Nm`.
That leg channel saturated on `48.812%` of steps. The right wheel separately
reached its `±10 A` current limit on `1.653%` of steps; actual speed absolute
P95 was `7.827 rad/s` while target absolute P95 was `0.527 rad/s`.

The first physical preflight covered zero-action leg torque and low-amplitude
wheel probes, but it did not sweep large leg actions through the PD controller.
The real training/evaluation trace exposed that missing check. No motor limits,
MJCF values, or acceptance thresholds were changed.

## Paired action-scale counterfactual

The experiment in `scripts/ablate_pure_ppo_action_scale.py` reuses the same
seed 0 checkpoint and the same 100 ID test seeds (20000–20099), with
deterministic actions. It changes only `LEG_POSITION_SCALE_RAD` from `0.50` to
`0.35 rad`; it does not train or alter checkpoint weights.

| Measurement | Baseline 0.50 | Counterfactual 0.35 |
|---|---:|---:|
| Accepted ID episodes | 0/100 | 100/100 |
| Mean pitch RMS | 0.3266° | 1.2514° |
| Maximum pitch peak | 3.0389° | 2.9886° |
| Maximum drift | 0.04861 m | 0.06497 m |
| Fourth-leg torque saturation, mean | 48.812% | 0% |
| Right-wheel current saturation, mean | 1.653% | 0.035% |

This paired inference-only result isolates the mapping scale as a plausible
cause of the seed 0 saturation mode. It is not a trained policy or an export
candidate. Seed 1 also passed the baseline mapping at `0.50`, so the data do
not establish that `0.50` prevents learning; they show that PPO seeds can
converge to different actuator-use patterns. A fresh, named `0.35 rad`
experiment is the controlled training follow-up. It keeps all other task,
reward, actuator, observation, and PPO settings fixed and starts from random
weights.

## Fresh training ablation

The experiment ID is `pure_ppo_stand_leg_scale_035_v1`. Its training entry
point reruns physical preflight, random-action/VecEnv smoke, and the PPO dry
run after locking the `0.35 rad` map. Its config records the changed factor and
training entrypoint SHA256; validation uses seeds 11000–11049, separate from
the baseline validation seeds.

```bash
$SIM_PYTHON scripts/train_pure_ppo_leg_scale_ablation.py --seed 0 --updates 800 --num-envs 8 --device cuda
$SIM_PYTHON scripts/eval_pure_ppo_leg_scale_ablation.py --checkpoint checkpoints/pure_ppo_stand_leg_scale_035_v1/seed0/best.zip --training-seed 0 --mode all --episodes 100
```

The independent test seed range for this experiment starts at 23000. A fresh
training run and its ID/stress evaluation are required before this scale can be
considered a learned Pure PPO result.

## Fresh 0.35 rad training result

The experiment ID is `pure_ppo_stand_leg_scale_035_v1`; seed 0 started from new PPO weights and ran all 800 updates. Its independent seeds were 23000–23099. It passed posture and full-duration survival but returned `0/100` accepted because the policy drifted and exceeded actuator limits:

- Pitch RMS mean `0.3720°`, drift P95 `0.1660 m`, maximum drift `0.1836 m`.
- Mean wheel contacts `0.511/0.568`; all episodes failed both wheel contact gates.
- Mean wheel current saturation `0.439/0.412`; every episode failed saturation fraction and duration.
- Leg channel 3 saturation averaged `0.269`; every episode exceeded that channel's saturation limits.
- No chassis contact; all episodes ran 1000 steps.

The held-out 10,000-step trace is `runs/pure_ppo_stand_leg_scale_035_v1/seed0/test_trace/step_trace.csv.gz`. Wheel target absolute P95 was `3.51/5.33 rad/s`; actual speed absolute P95 was `16.75/18.13 rad/s`, with a peak of `24.65 rad/s`. Wheel current was clipped at `±10 A` on `43.8%/41.3%` of steps. The actor's two negative leg channels were at `-1` on `19.2%/31.6%` of steps, consistent with the measured leg torque saturation.

### Paired scale sweep on the same fresh checkpoint

This sweep keeps the actor weights and held-out IDs fixed (23000–23099) and changes only the leg target scale. It is diagnostic inference, not retraining:

| Leg scale | Accepted | Pitch RMS mean | Drift P95 | Contact mean L/R | Wheel current saturation max |
|---:|---:|---:|---:|---:|---:|
| 0.35 rad | 0/100 | 0.372° | 0.1660 m | 0.511 / 0.568 | 45.1% |
| 0.30 rad | 0/100 | 0.656° | 0.2227 m | 0.567 / 0.624 | 38.5% |
| 0.25 rad | 0/100 | 1.069° | 0.2327 m | 0.655 / 0.706 | 29.2% |

Lowering leg scale reduces wheel saturation and improves contact for this actor, but it increases drift and none of the scales passes. The full per-episode paired records and script hash are in `runs/pure_ppo_stand_leg_scale_035_v1/seed0/scale_levels_counterfactual/comparison.json`.

## Fresh 5x position-drift reward run

The data showed the 0.35 rad actor staying upright while moving, so the next fresh experiment held the action map and every other setting fixed and changed only the per-step `position_drift` coefficient from `0.10` to `0.50`. The task ID is `pure_ppo_stand_leg_scale_035_drift_weight_5x_v1`. Each training seed used fresh PPO initialization, 800 updates, 8 environments, and validation seeds 12000–12049. Held-out ID seeds were 24000–24099, 24100–24199, and 24200–24299.

| Seed | Best update | Validation | Independent ID | Result |
|---:|---:|---:|---:|---|
| 0 | 800 | 50/50 | 100/100 | Pass; Wilson 95% `[0.9630, 1.0000]` |
| 1 | 650 | 18/50 | 49/100 | Fail; 51 wheel saturation-duration failures and 8 drift failures |
| 2 | 550 | 47/50 | 92/100 | Fail; 8 episodes had four consecutive saturated wheel steps, above the fixed three-step limit |

Seed 0 also completed 100/100 episodes in each stress set. Pitch RMS/drift P95 were `0.702°/0.0980 m` at ±5° pitch, `0.584°/0.0697 m` at out-of-range pitch rate, and `0.588°/0.0611 m` after the mid-run impulse. No stress episode had chassis contact; wheel and leg saturation stayed near zero.

Seed 1's independent failures are strongly pitch-sign dependent: negative initial pitch passed `0/31`, near-zero passed `16/33`, and positive pitch passed `33/36`. The actor has little leg saturation, but wheel saturation duration and drift remain. Seed 2 has no leg saturation, good contact and drift, and very low saturation fractions; its eight failures come only from a saturation run of four consecutive steps, while acceptance allows at most three.

These results show that stronger drift reward can produce one robust Pure PPO policy while not making the training outcome reliable across random initializations. The acceptance threshold, termination conditions, model, and real actuator limits were not altered. The follow-up report is in `TRAINING_REPORT.md`. No final ONNX was exported because only one of these three fresh training seeds passed independent ID acceptance.
