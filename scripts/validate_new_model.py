"""Read-only dynamics/old-policy transfer audit. Never calls PPO.learn().

Run once with UZ05_MODEL_XML set to the immutable legacy XML, then with the
new default. Outputs belong in separate directories; no checkpoint is written.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from uz05.env import UZ05Env
from uz05.spec import MODEL_XML, MESH_DIR, REPO_ROOT, env_params_for_stage
from uz05.reversal_diagnostics import replace_observed_command
from uz05.m3508_diagnostics import reference_at
from run_m3508_validation import PROFILES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', default=str(REPO_ROOT/'checkpoints/joint_tracking_pos_v16/checkpoint'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'results.json').exists():
        raise RuntimeError('Use a fresh output directory; do not overwrite audit results')
    checkpoint = Path(args.checkpoint)
    source = json.loads(Path(str(checkpoint)+'_contract.json').read_text())
    # Explicit transfer experiment: verify the source belongs to the legacy
    # model, not a fabricated contract made to pass the new-model resume guard.
    legacy = REPO_ROOT/'uz05_修正后(3).xml'
    assert source['model_xml_sha256'] == digest(legacy)
    p = env_params_for_stage('joint_tracking_v16')
    assert vars(p.wheel) == source['actuator_config']['wheel']
    p.robust.pushes_enabled = False
    e = UZ05Env('joint_tracking_v16', params=p)
    m, d = e.sim.model, e.sim.data
    e.stage = replace(e.stage, moving_start_probability=0., episode_steps=3000)
    assert m.nq == 21 and m.nv == 20 and m.nu == 6 and m.neq == 4
    assert m.opt.timestep == .002
    planes = np.flatnonzero(m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    assert len(planes) == 1
    reset_rows = []
    for randomization in (False, True):
        e.params.domain_randomization.enabled = randomization
        for seed in range(50):
            obs, _ = e.reset(seed=61000+seed)
            assert np.isfinite(obs).all()
            s = e.sim
            error = float(s.reset_projector.errors(d).max())
            speed = float(np.linalg.norm((s.reset_projector.jacobian(d)@d.qvel).reshape(-1,3), axis=1).max())
            assert error < 1e-8 and speed < 1e-8
            reset_rows.append(dict(seed=61000+seed, randomization=randomization,
                                   closure_m=error, closure_speed_m_s=speed,
                                   resamples=e.reset_projection_info['resamples']))
    e.params.domain_randomization.enabled = False
    normalizer = VecNormalize.load(str(checkpoint)+'_vecnormalize.pkl', DummyVecEnv([lambda:e]))
    normalizer.training = False
    normalizer.norm_reward = False
    policy = PPO.load(checkpoint, device='cpu')
    checkpoint_hash = digest(str(checkpoint)+'.zip')
    rows = []
    started = time.perf_counter()
    for task in ('stand', 'forward2', 'reverse2', 'stop2_reverse', 'motion'):
        for seed in (51000, 51001, 51002):
            obs, _ = e.reset(seed=seed)
            samples = []
            closure_peak = mapping_peak = wheel_peak = leg_peak = current_peak = 0.
            phase = -1
            for step in range(3000):
                tick = step*p.control_dt
                index, command, _ = reference_at(PROFILES[task], tick, 'step', 0., p.control_dt, task)
                e.command[:] = command
                e.command_target[:] = command
                e.command_steps_left = 3001
                if index != phase:
                    phase = index
                    obs = e._obs(current_action=e.previous_action)
                else:
                    obs = replace_observed_command(obs, command, e._observation_encoder.clip_observations)
                action, _ = policy.predict(normalizer.normalize_obs(obs), deterministic=True)
                obs, reward, terminated, truncated, info = e.step(action)
                assert np.isfinite(obs).all() and np.isfinite(reward)
                assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
                c = e._previous_control
                actual = d.qfrc_actuator[e.sim.wheel_dof_adr]
                mapping_peak = max(mapping_peak, float(np.max(np.abs(actual-c['wheel_torque']))))
                # Every motor substep is bounded by the controller current and
                # torque clamp; reconstruct its torque from the sampled current.
                motor_torque = e.actuators.wheel.torque_from_current(c['motor_wheel_current_samples'])
                wheel_peak = max(wheel_peak, float(np.max(np.abs(motor_torque))))
                leg_peak = max(leg_peak, float(np.max(np.abs(c['motor_leg_torque_samples']))))
                current_peak = max(current_peak, float(np.max(np.abs(c['motor_wheel_current_samples']))))
                closure_peak = max(closure_peak, float(e.sim.reset_projector.errors(d).max()))
                samples.append([(step+1)*p.control_dt, info['pitch'], info['roll'],
                                info['tracking_vx_error']+command[0], command[0],
                                info['tracking_height_error'], info['wheel_saturation_fraction']])
                if terminated or truncated:
                    break
            a = np.asarray(samples)
            survived = len(a) == 3000 and not terminated
            tail = a[a[:,0] > 22.] if survived else np.empty((0,7))
            row = dict(task=task, seed=seed, steps=len(a), survived_24s=survived,
                       terminated=bool(terminated), truncated=bool(truncated),
                       pitch_rms_deg=float(np.degrees(np.sqrt(np.mean(a[:,1]**2)))),
                       pitch_peak_deg=float(np.degrees(np.max(np.abs(a[:,1])))),
                       roll_rms_deg=float(np.degrees(np.sqrt(np.mean(a[:,2]**2)))),
                       final2_vx_m_s=float(tail[:,3].mean()) if len(tail) else None,
                       final2_speed_rmse_m_s=float(np.sqrt(np.mean((tail[:,3]-tail[:,4])**2))) if len(tail) else None,
                       closure_peak_m=closure_peak, mapping_error_peak_nm=mapping_peak,
                       wheel_torque_peak_nm=wheel_peak, leg_torque_peak_nm=leg_peak,
                       current_peak_a=current_peak,
                       mujoco_warnings=[int(w.number) for w in d.warning])
            assert mapping_peak < 1e-10 and current_peak <= 10.+1e-10
            assert wheel_peak <= 2.46+1e-10 and leg_peak <= 20.+1e-10
            assert not any(row['mujoco_warnings'])
            rows.append(row)
            np.savez_compressed(out/f'{task}_{seed}.npz', samples=a,
                                columns=['time_s','pitch_rad','roll_rad','vx','command_vx','height_error','wheel_sat'])
            print(json.dumps(row), flush=True)
    assert digest(str(checkpoint)+'.zip') == checkpoint_hash
    result = dict(model=str(MODEL_XML), model_sha256=digest(MODEL_XML), mesh_dir=str(MESH_DIR),
                  mujoco_version=mujoco.__version__, checkpoint=str(checkpoint),
                  checkpoint_sha256=checkpoint_hash, source_model_sha256=source['model_xml_sha256'],
                  no_training=True, explicit_cross_model_evaluation=True,
                  actuator_model='A_legacy, unchanged source checkpoint mapping',
                  elapsed_s=time.perf_counter()-started, resets=reset_rows, episodes=rows,
                  dimensions=dict(nq=m.nq,nv=m.nv,nu=m.nu,neq=m.neq),
                  integrator=int(m.opt.integrator), cone=int(m.opt.cone), plane_count=len(planes))
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    normalizer.close()


if __name__ == '__main__':
    main()
