# Pure PPO stand baseline

## Repository state

- Baseline commit: `ff19e52cf6dd1b309adb2c07e575ddd2be4d870e`.
- Working branch: `experiment/pure_ppo_stand_v1`.
- The worktree already contained user changes before this task. They were preserved; new work is isolated by file and staged by explicit path/hunk.
- The taskbook is at the repository root: `WheelLeg_Pure_PPO_Stand_Codex_Taskbook.md`.
- `docs/sim2real_training_strategy.md` is absent from the local worktree. The referenced upstream version was reviewed; it was treated as methodology, not as a source of weights or controller code.
- PR [#1](https://github.com/Pineapple-pai/WheelLeg/pull/1) describes the free-joint angular-velocity frame correction. PR [#2](https://github.com/Pineapple-pai/WheelLeg/pull/2) describes per-environment observation-noise configuration. Both were reviewed against the worktree; the local changes are applied without cherry-picking. The per-instance `ObservationEncoder` already exists in the dirty worktree.

## Runtime lock

The first experiment uses the `sim` Conda environment and the current model file `uz05_修正后(3).xml`.

| Component | Observed value |
|---|---|
| Python | 3.10.20 |
| MuJoCo | 3.2.3 |
| Stable-Baselines3 | 2.3.2 |
| PyTorch | 2.2.2+cu118 |
| NumPy | 1.26.4 |
| Gymnasium | 0.29.0 |
| ONNX | 1.16.2 |
| ONNX Runtime | 1.19.2 |
| Device | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB |

This `sim` environment did not initially contain Matplotlib and its user-site Pandas import lacked `pytz`, preventing SB3 from importing. The runtime was made usable by adding `matplotlib==3.8.4`, `pillow==11.3.0`, and `pytz==2025.2`. PyTorch's CUDA 11.8 libraries are bundled with Isaac Sim 4.0.0, so commands must expose those libraries through `LD_LIBRARY_PATH`; this does not change the CUDA or Torch packages.

## Model and mechanics

- MJCF SHA256: `22fc08a5f900ffc20d977c29851618fb0ca03c1d6b01dd2da0f32209ba505c54`.
- Compiled model: `nq=21`, `nv=20`, `nu=6`, four equality loop constraints, `dt=0.002 s`, gravity `-9.81 m/s²`, total mass `17.122316 kg`, wheel radius `0.055 m`.
- The active standing pose is `[-0.19450, 0.19129, -0.19141, 0.19057] rad`; the passive linkage pose is solved once from the four equality loops. Across 100 fresh compile/reset trials, maximum equality residual was below `9.75e-15 m`, both wheels contacted the floor, and there was no chassis contact.
- The reset base height is `0.23825 m`; measured link lengths were `0.183904 m` and `0.184412 m`.
- Both wheel axes resolve to world `-Y`. Positive normalized wheel action produces a positive wheel target/current/torque and a measured negative body-X velocity change; negative action reverses the response. The tested `±0.1` and `±0.3` actions did not saturate the wheel current controller.
- The local visual mesh directory in this workspace is `/home/p/下载/UZ05_MJCF_real/UZ05_MJCF_real_params/meshes/stl`. Training strips only non-colliding display mesh geoms before compilation; inertial and collision definitions remain from the selected MJCF. Rendering can set `UZ05_MESH_DIR` to the mesh directory.

## Known starting issues and limits

- The old `UZ05Env.reset()` does not initialize the passive linkage coordinates at the closed-loop solution; its reset can begin with centimetre-scale loop mismatch. The Pure PPO task sets a physically closed passive pose.
- The old absolute-zero leg target gives approximately `[19.45, -19.129, 19.141, -19.057] Nm` at the tested standing pose. The new fixed coordinate origin makes zero action target `q_ref`, yielding zero initial leg PD torque.
- `UZ05Model.base_ang_vel_world` previously returned the free-joint local `qvel[3:6]` unchanged. It now rotates to world coordinates; callers convert once to body coordinates. A dependent pitch-rate read also now uses the body frame.
- The previous environment's `dof_acc` finite difference was oldest minus newest. It is corrected to newest minus oldest.
- `base_lin_vel` in the actor observation is currently a MuJoCo-derived body velocity used as an idealized proxy. It has not been established as a real sensor estimate. This work demonstrates simulation behavior only, not hardware deployment readiness.
- The wheel controller uses the configured proportional speed gain and zero integral gain (`speed_ki_a_per_rad_s2=0`). The controller retains an integral state for interface compatibility, which is reset at episode boundaries. Its gain and other actuator values are simulator configuration and have not been newly calibrated against physical motor logs.
- The zero-action policy falls during an unassisted open-loop check. This is expected; zero action is not a standing controller and is not an acceptance test.

## Reference projects reviewed

- [SCUT wheeled-legged RL](https://github.com/scutrobotlab/wheeled-legged_RL): reviewed for environment and asymmetric actor-critic methodology. The current implementation uses its own MuJoCo model and does not import pretrained policy weights or a reference action controller.
- [cheng695/RL](https://github.com/cheng695/RL): reviewed as a same-model implementation reference. No weights or behavior-cloning targets are used.

The actual code, contracts, preflight report, and tests for this task are the authoritative evidence for the experiment.
