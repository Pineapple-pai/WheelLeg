# UZ-05 deployment boundary

The frozen timing boundary is 500 Hz motor control and 125 Hz ONNX policy
inference.  One policy action is held for four motor ticks.

The reference controller exposes separate `policy_tick()` and `motor_tick()`
entry points.  The 500 Hz task must never invoke or wait for ONNX; the 125 Hz
task publishes a complete six-value action atomically, and timeout handling
holds the last target before entering the hardware safe-stop path.

The leg path is DM MIT (`p_des`, `v_des=0`, fixed `Kp/Kd`, `t_ff=0`).  The wheel
path is PPO wheel-speed target, 500 Hz PI, then C620 torque-current command.

`uz05_interface.json` separates simulation values from fields that cannot be
declared safe until hardware bench testing.  Any value containing `REQUIRED`
must block motor enable in the real controller.

The deployment runtime must consume only the first 38 actor observations.  The
remaining 56 values in the training observation are critic-only and must never
be required by the ONNX model.

Reference protocol packing is implemented in `scripts/uz05/deployment.py`.
It performs no CAN I/O and deliberately does not guess motor IDs, offsets or
installation signs.
