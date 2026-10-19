# Pure PPO control-path audit

## Policy boundary

The only training action is the six-dimensional SB3 Gaussian PPO actor output in the exact order:

```text
[wheel_left, wheel_right, leg_left_1, leg_left_2, leg_right_1, leg_right_2]
```

`AsymmetricActorCriticPolicy` gives the Actor only observation elements `[0:38]`; its value network reads all 94 observation values. `AsymmetricMlpExtractor.forward_actor()` slices the actor prefix directly. There is no action head from the critic, and critic-only features cannot change actor latents. A unit test perturbs the privileged tail and confirms actor invariance while critic features change.

## Fixed action mapping and actuators

`map_stand_offset_action()` in `scripts/uz05/pure_ppo_stand.py` is the single mapping for training and simulation evaluation. It accepts no state or observation.

- Wheel target: `omega_target = +1 × 8.0 × action[0:2] rad/s`, clipped to the physical speed limit by the existing speed controller.
- Leg targets: `clip(q_ref + 0.5 × action[2:6], ±1.2 rad)`, where `q_ref = [-0.19450, 0.19129, -0.19141, 0.19057] rad`.
- Wheel low-level path: speed error → configured proportional current demand (`1.25 A/(rad/s)`, Ki `0`) → `±10 A` current limit → `0.246 Nm/A × 0.70` efficiency → `±2.46 Nm` output torque limit → divide by gear ratio `268/17` for MuJoCo motor control.
- Leg low-level path: fixed joint position target and zero target velocity → configured joint PD (`kp=100`, `kd=4`) → `±20 Nm` torque limit → MuJoCo motor control.
- One PPO action is held through four 2 ms MuJoCo ticks, giving 125 Hz policy and 500 Hz motor updates.

The speed/position feedback above is the fixed motor-level controller permitted by the taskbook. It has no pitch input or separate balance branch. All six policy dimensions map independently to their corresponding two wheel and four leg targets. Current or torque can respond to feedback, as expected for the motor interface; the coordinate targets cannot.

## Reward and training-path checks

`compute_stand_reward()` uses uprightness, survival, pitch rate, horizontal speed, drift, height, leg length, action rate, measured actuator effort, and terminal failure. It does not receive a teacher action, target action, reference trajectory, pitch feedback gain, or controller output. Reward terms are recorded raw and as weighted contributions at each step and aggregated in episode and rollout logs.

The training environment fixes flat ground, zero motion commands, 1000 policy steps, initial pitch `Uniform(-3°, +3°)`, initial pitch rate `Uniform(-0.1, +0.1) rad/s`, and zero roll/yaw. Domain randomization and observation noise are explicitly forced off at construction and reported on every reset. It has no delay, push, action freezing, random height/leg commands, curriculum, helper controller, or time-varying reward/optimizer setting.

Training constructs `PPO(AsymmetricActorCriticPolicy, ...)` with new random weights and a fresh optimizer. `PPO.load()` occurs only in evaluation/export entry points. Best checkpoint selection uses a fixed 50-seed deterministic validation set, and never uses training returns as a selection criterion.

## Observation caveats

The 38D Actor layout is: gravity projection, body angular velocity, reset-relative heading sin/cos, body linear velocity proxy, relative height, relative planar position, terrain preview, four leg positions, four leg velocities, two wheel velocities, five fixed command values, and six previous actions. The Critic gets the 56D privileged tail (true velocity/position, leg length and rate, contacts, terrain scan, joint acceleration/torque, and zero domain-parameter values).

Noise is encoded by a per-environment `ObservationEncoder`; in this task its configured noise vector is all zeros. The Actor's linear velocity, height, planar position, and terrain measurements currently come from ideal MuJoCo proxies; they must not be described as already calibrated hardware estimates.

## Search and manual call-path review

The new training, task, evaluation, and export files were searched for `teacher`, `expert`, `residual`, `fallback`, `override`, `blend`, `warmstart`, `pretrained`, `curriculum`, `anneal`, `LQR`, `BalanceController`, and pitch PID patterns. Search hits in the task module are docstrings or the kinematic closure residual variable. Manual call-path review found no policy helper, action overwrite, teacher signal, checkpoint load during training, or balance-controller import. The only feedback loops are the wheel speed current path and leg joint PD described above.

## Reproducibility contract

The run and checkpoint metadata lock model SHA256, action protocol and order, `q_ref`, scales, signs, actor observation layout hash, actuator configuration, policy/motor rates, noise/randomization switches, seed, configuration SHA256, source fingerprint, and commit. The evaluator refuses missing/legacy/mismatched contracts. Legacy deployment code is not used by this task and no policy is sent to a real robot.
