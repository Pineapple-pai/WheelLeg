# Pure PPO stand training report

## Status at this report revision

The physical preflight, 1000-transition random-action smoke test, 17 automated tests, and PPO startup dry run pass. The original `pure_ppo_stand_v1` three-seed experiment is complete: seed 1 passed independent ID acceptance, while seeds 0 and 2 failed. Two evidence-driven, one-factor follow-ups are also complete. A fresh 0.35 rad leg-scale run exposed a moving, saturated policy; a separate 5x position-drift reward run produced one seed that passed all independent ID episodes, while its other two seeds failed the 95% criterion.

The task is **not reliable across the required three training seeds**. No final ONNX was exported. The best follow-up checkpoint is seed 0 of `pure_ppo_stand_leg_scale_035_drift_weight_5x_v1` (100/100 independent ID episodes), but seed 1 passed 49/100 and seed 2 passed 92/100. These are real outcomes under the unchanged acceptance thresholds; the two failing seeds are retained in full.

Every formal run records source fingerprint `77531bd427e59fff9dd7f53ac4bd1f4c992d86cff8fe82a9311b4da8eadb7799`. After those runs, a redundant blank line at end of `scripts/uz05/pure_ppo_eval.py` was removed to make the committed source pass whitespace checks. The resulting current fingerprint is `36864f138974c210cafcb33e52e9db14d337939d72c387d54d7c305bdc9c2e45`; the change is formatting-only and does not alter evaluation behavior. Run and checkpoint contracts preserve the exact fingerprint used during training.

## Preflight evidence

- Model SHA256: `22fc08a5f900ffc20d977c29851618fb0ca03c1d6b01dd2da0f32209ba505c54`.
- 100 fresh compile/reset trials passed; maximum four-loop residual `9.75e-15 m`, all left/right wheel contacts present, no chassis contact, no observation noise, and no domain randomization.
- All 8 dual/single wheel physical probes at normalized ±0.1/±0.3 remained finite and below the 10 A current limit. Common positive action gave `body_vx_delta=-0.00238 m/s` at 0.1 and `-0.00723 m/s` at 0.3; common negative action reversed those signs.
- The measured one-policy-step wheel speeds were also recorded: common `+0.1` mapped to `[0.8, 0.8] rad/s` targets and `[0.181, 0.125] rad/s` actual speeds after 8 ms; common `+0.3` mapped to `[2.4, 2.4]` and `[1.186, 1.123] rad/s`. Negative probes reversed both wheel-speed and body-velocity signs; isolated left/right probes changed only the corresponding wheel target/current/torque. The speed values are short transient measurements, not claims of steady tracking.
- Zero action maps to wheel speed zero and the fixed `q_ref`, producing zero initial leg torque. The previous absolute-zero coordinate would request about 19 Nm on each leg joint at this pose.
- Pitch ±5°, pitch rates ±0.1 rad/s, nonzero roll/yaw, and angular-velocity world/body round trips passed.
- An offline 2000-step reward sample was run with zero actions and with initial-policy-scale Gaussian actions (`sigma=exp(-1.5)`). Both sets fell in this deliberately unassisted task: 27 and 32 body-contact terminations, respectively. Mean weighted pitch-rate cost was `-3.46` / `-4.05`, versus mean upright reward `0.55` / `0.50`; all raw and weighted quantiles are retained in `artifacts/pure_ppo_stand_v1/smoke.json`. Reward coefficients are fixed before training; the rate term is a state outcome penalty and is not an action target.

## Automated tests and smoke

```text
unittest: 17 tests, OK
random-action smoke: 1000 transitions, 17 expected body-contact terminations, no numerical failures
SB3 DummyVecEnv: observation shape (2, 94), finite rewards and observations
ONNX random-policy parity test: 1000 observations, max error <= 1e-4
```

The random-action terminations are retained as smoke evidence and are not PPO failures or acceptance outcomes.

## Formal training results

### Seed 0

- Fresh PPO initialization; 800/800 updates, 3,276,800 environment steps, CUDA; no checkpoint was loaded. Fixed validation selected `best.zip` at update 650; `last.zip` is update 800.
- Late training produced 1000-step episodes. At update 800, rollout reward mean was `1.045`, episode return mean `1053.26`, mean episode length `1000`, pitch RMS `0.00635 rad`, and explained variance `0.973`; PPO approximate KL was `0.00976`, clip fraction `0.111`, and policy standard deviations remained finite.
- Independent ID test: `0/100` accepted (Wilson 95% interval `[0, 0.03699]`). All 100 survived the full 8 seconds, with mean pitch RMS `0.3266°`, P95 `0.6082°`, maximum peak pitch `3.0389°`, maximum planar drift `0.04861 m`, both wheel contact fractions effectively 100%, and zero chassis-contact episodes.
- The gate failures were actuator saturation. Each episode exceeded the leg-torque saturation fraction limit on leg joint 4: mean `48.812%`, P95 `49.7%`, maximum `49.7%`; a wheel-saturation fraction or duration limit also failed in 88/100 episodes. The most common failure tuple was wheel fraction plus leg fraction (77/100); 12/100 failed only leg fraction, 9/100 also exceeded wheel saturation duration, and 2/100 failed wheel duration plus leg saturation.
- The final actor's mean normalized action over the 100 ID episodes was `[-0.0298, 0.0328, 0.0181, -0.1339, -0.2084, -0.4447]`. A 100,000-row trace (`runs/pure_ppo_stand_v1/seed0/test_trace/step_trace.csv.gz`) confirms the leg mechanism: on the fourth leg joint, mean `q_target - q_actual_before_step` was `-0.09769 rad`; on saturated steps it was `-0.12348 rad` and mean speed was `-2.223 rad/s`. With the configured `kp=100`, `kd=4`, the measured P and D demands averaged `-12.35 Nm` and `-8.89 Nm` on those steps, totaling `-21.24 Nm` before the fixed `-20 Nm` clamp. It saturated at `-20 Nm` for `48.812%` of all traced policy steps. This directly ties the learned negative leg target and its motion to saturation.
- The same trace records both wheel speeds around each 8 ms policy step. Right-wheel actual speed had absolute P95 `7.827 rad/s` and peak `18.630 rad/s`, while its target absolute P95 was only `0.527 rad/s`; speed error reached the fixed `±10 A` controller limit for `1.653%` of steps (left: `0.084%`). This is a separate wheel speed-tracking saturation issue, not evidence that the actor emits large wheel targets (mean targets were `[-0.239, 0.263] rad/s`).
- Independent stresses (reported separately, never counted as ID success): 100/100 episodes in each distribution ran the full duration. At ±5° initial pitch, mean pitch RMS was `0.5616°` and drift P95 `0.09991 m`; at out-of-range pitch rate they were `0.3522°` and `0.06216 m`; with a mid-run ±0.20 rad/s pitch-rate impulse they were `0.3327°` and `0.03636 m`. These do not change the ID failure.
- The 400-update validation already showed survival but intermittent drift and wheel-saturation failures. By update 650, all 50 validation episodes survived with mean pitch RMS `0.310°` and maximum drift `0.0506 m`, but none passed because leg-torque saturation remained about 49% in one channel. Update 800 traded this for some wheel saturation again; `best.zip` therefore remains the deterministic-validation selection, not the last weights.

### Seed 1

- Fresh PPO initialization; 800/800 updates and 3,276,800 timesteps with the same frozen config. At update 400, fixed validation was 0/50 due to drift (`max 0.148 m`) and wheel saturation (`max fraction 1.3%`, up to 13 consecutive steps); all 50 survived, with pitch RMS mean `0.9996°`.
- Validation changed sharply over updates: 30/50 accepted at 650, 0/50 at 700, then 50/50 at both 750 and 800. Best selection chose update 750 by lower pitch RMS: `0.3504°` mean, `0.6037°` P95, peak maximum `2.9554°`, drift maximum `0.06232 m`, wheel saturation maximum `0.3%` and one consecutive step.
- At update 800, rollout reward mean was `1.056`, mean episode length `1000`, pitch RMS mean `0.00969 rad`, contact fractions `0.9718/0.9863`, approximate KL `0.0181`, clip fraction `0.1279`, and explained variance `0.7703`.
- Independent ID test on the selected update-750 checkpoint: `100/100` accepted (Wilson 95% interval `[0.9630, 1.0000]`); all survived 1000 steps, pitch RMS mean `0.3590°` / P95 `0.5821°`, maximum peak `2.9714°`, drift maximum `0.05587 m`, both wheel contact fractions `100%`, and zero body-contact episodes. Leg saturation maxima were at most `0.4%` with one-step maximum duration; wheel saturation maximum was `0.4%` and one step. This passes seed 1 ID acceptance.
- Independent stress evaluations each ran 100/100 episodes to the 8-second limit: ±5° pitch had pitch RMS mean `0.5272°`, drift P95 `0.1050 m` / max `0.2073 m`; out-of-range pitch rate had `0.3656°`, drift P95 `0.0531 m` / max `0.1095 m`; the mid-run impulse had `0.3602°`, drift P95 `0.0444 m` / max `0.0559 m`. Stress outcomes remain separate from ID acceptance and are not deployment evidence.

### Seed 2

- Fresh PPO initialization; 800/800 updates and 3,276,800 timesteps. Fixed validation never passed an episode. At update 400, the policy was still falling; by 600 it survived 49/50 episodes but had 0/50 accepted due to drift and wheel saturation. Update 700 was selected as `best.zip` because it had the lowest validation pitch RMS among checkpoints with zero accepted episodes, even though it still failed drift and actuator gates.
- At update 800, rollout reward mean was `0.601`, 5/5 completed episodes ran 1000 steps, pitch RMS mean was `0.02718 rad`, and explained variance was `0.938`; however, mean wheel contact fell to `0.4588/0.4722` and mean wheel saturation to `60.93%`.
- Independent ID test of the deterministic-validation `best.zip`: `0/100` accepted (Wilson 95% interval `[0, 0.03699]`). Termination reasons were 77 drift-limit failures and 23 time limits. Pitch RMS mean was `0.6861°` / P95 `0.9577°`, but drift P95 was `0.40120 m`; mean wheel contact was `0.4648/0.4363`; wheel saturation averaged `48.15%/51.21%` with maxima of 29/32 consecutive steps. Leg saturation averaged `16.49%, 4.81%, 21.56%, 3.06%` across the four joints. No body contacts occurred. This seed learned an upright but moving, partly airborne behavior that fails standing acceptance.
- The three stress tests showed the same failure mode: 69/100 drift-limit terminations at ±5° pitch, 73/100 at out-of-range pitch rate, and 76/100 after the mid-run impulse. Drift P95 remained about `0.401 m` in all three distributions; these results do not change the ID failure.

### Current failure analysis

The policy has learned to hold the robot upright on the nominal task, but it has not learned a control solution inside the actuator saturation limits. The initial physical preflight verified small wheel probes and zero-action leg torque, but did not sweep the full leg-action scale and therefore missed the high-action PD torque saturation risk. The held-out result is real and not hidden by posture averages. Step-level data locates the leg clipping at the combination of a learned target offset, a measurable tracking error, and velocity damping; it also locates a distinct right-wheel speed error during current clipping. Any follow-up training change will use a separately named experiment and alter one factor only; the three frozen seed runs remain intact.

### One-factor diagnostic: leg target scale

Using the same seed 0 `best.zip`, the same 100 held-out IDs (20000–20099), and deterministic actions, an inference-only counterfactual changed only `LEG_POSITION_SCALE_RAD` from `0.50` to `0.35`. It yielded 100/100 accepted episodes, pitch RMS mean `1.2514°`, P95 `1.3807°`, peak maximum `2.9886°`, drift maximum `0.06497 m`, zero leg torque saturation, and right-wheel current saturation mean `0.035%` (baseline `1.653%`). Baseline was 0/100 accepted with fourth-leg saturation mean `48.812%`. The counterfactual is **not** a trained checkpoint and is not eligible for export; applying the scale change to baseline weights cannot establish training convergence. It does support the single-factor hypothesis that the first leg target scale made the learned policy exceed motor authority. A separate task ID, fresh initialization, separate gates/contracts, and a full training run at `0.35` are now prepared to test that hypothesis.

### Fresh one-factor run: leg target scale 0.35 rad

The paired, inference-only `0.50 → 0.35 rad` change on the original seed 0 checkpoint removed its fourth-leg saturation and passed the same 100 IDs, but was not a trained checkpoint. A fresh seed 0 run was therefore trained for 800 updates at `0.35 rad` with every other task and PPO setting fixed. Its independent test used seeds 23000–23099 and returned `0/100` accepted. All 100 survived 8 seconds and had mean pitch RMS `0.3720°`, but drift P95 was `0.1660 m`, wheel contact averaged `0.511/0.568`, wheel current saturation averaged `43.9%/41.2%`, and leg channel 3 torque saturation averaged `26.9%`.

The 10,000-step trace at `runs/pure_ppo_stand_leg_scale_035_v1/seed0/test_trace/step_trace.csv.gz` shows the actuator mechanism. Wheel target absolute P95 was `3.51/5.33 rad/s` while actual speed absolute P95 was `16.75/18.13 rad/s` and reached `24.65 rad/s`; the wheel current hit its fixed `±10 A` limit for `43.8%/41.3%` of steps. The actor saturated its two negative leg action channels on `19.2%/31.6%` of steps. The lower mapping scale alone did not force a newly trained policy into the acceptable actuator mode.

A paired inference-only sweep reused this exact checkpoint and all 100 IDs while changing only the leg target scale:

| Leg scale | Accepted | Pitch RMS mean | Drift P95 | Wheel contact mean L/R | Wheel saturation max |
|---:|---:|---:|---:|---:|---:|
| 0.35 rad | 0/100 | 0.372° | 0.1660 m | 0.511 / 0.568 | 45.1% |
| 0.30 rad | 0/100 | 0.656° | 0.2227 m | 0.567 / 0.624 | 38.5% |
| 0.25 rad | 0/100 | 1.069° | 0.2327 m | 0.655 / 0.706 | 29.2% |

Reducing the scale lowered some current saturation but increased drift; none of the three mappings passed. These rows are inference-only diagnostics, not trained policies or export candidates. Exact data is in `runs/pure_ppo_stand_leg_scale_035_v1/seed0/scale_levels_counterfactual/comparison.json`.

### Fresh one-factor run: 5x position-drift reward

The next run held the 0.35 rad leg map fixed and changed only `reward.position_drift` from `0.10` to `0.50`. All other reward terms, physics, action mapping, observations, termination, and PPO settings remained fixed. Seeds 0, 1, and 2 each used fresh PPO initialization, 800 updates, and the unchanged 95% independent acceptance requirement. Validation seeds were 12000–12049; held-out ID seeds began at 24000, 24100, and 24200 respectively.

| Training seed | Best update | Validation accepted | Independent ID accepted | Wilson 95% interval | Pitch RMS mean | Drift P95 | Independent failure evidence |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 800 | 50/50 | 100/100 | [0.9630, 1.0000] | 0.577° | 0.0611 m | None; all 100 also survived each stress mode |
| 1 | 650 | 18/50 | 49/100 | [0.3942, 0.5865] | 2.074° | 0.1051 m | 51 episodes exceeded wheel saturation duration; 8 exceeded drift |
| 2 | 550 | 47/50 | 92/100 | [0.8500, 0.9589] | 0.363° | 0.0466 m | All 8 failures exceeded wheel saturation duration by one step (4 vs allowed 3) |

Seed 0 is a real successful Pure PPO result for its own independent test distribution. Its stress evaluations each ran 100/100 episodes to 8 seconds: ±5° initial pitch had RMS `0.702°` and drift P95 `0.0980 m`; out-of-range pitch rate had RMS `0.584°` and drift P95 `0.0697 m`; the mid-run impulse had RMS `0.588°` and drift P95 `0.0611 m`. No stress episode had body contact, and actuator saturation stayed near zero.

Seed 1's failures are concentrated at negative initial pitch: `0/31` negative, `16/33` near-zero, and `33/36` positive-pitch episodes passed. It has no meaningful leg torque saturation, but its stress drift P95 reached `0.216 m` at ±5° pitch and `0.138 m` under rate OOD. Seed 2 had excellent posture, contacts, drift, and very low saturation fractions; however, eight episodes had a four-step wheel-current saturation run, above the fixed three-step limit. All eight are retained and counted as failures. No acceptance limits were relaxed.

The higher drift weight changed the dominant failure from continuous actuator saturation to rare or seed-specific drift and saturation-duration failures. It materially improves one random initialization and does not establish reliability across random seeds. The complete run logs, checkpoints, contracts, and evaluation rows are under `runs/pure_ppo_stand_leg_scale_035_drift_weight_5x_v1/` and `checkpoints/pure_ppo_stand_leg_scale_035_drift_weight_5x_v1/`.

### Overall determination

Across the original experiment only seed 1 passed. Across the final one-factor follow-up only seed 0 passed. Thus one of three fresh seeds passed in each training configuration, and **the requested cross-seed convergence/reliability criterion is not met**. The successful seed 0 checkpoint is not promoted as a final model. No ONNX was exported because the multi-seed result fails the project acceptance standard. A next experiment should first examine the seed 1 negative-pitch failures and the seed 2 four-step wheel-current bursts with paired data, then change one justified factor; current evidence does not justify changing real actuator limits, model physics, or acceptance criteria.
