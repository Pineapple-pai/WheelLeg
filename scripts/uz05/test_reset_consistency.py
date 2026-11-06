"""Paired actual-environment reset and 0.1/0.5 s startup diagnostics.

Run: conda run -n sim python scripts/uz05/test_reset_consistency.py
The fixed-action controller is the existing actuator bank, not a PPO policy.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from functools import partial

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uz05.env import UZ05Env
from uz05.reset_projection import CLOSURES
from uz05.spec import MODEL_XML, env_params_for_stage


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(zip(('median', 'p95', 'max'), map(float, np.percentile(a, [50, 95, 100]))))


def snapshot(e):
    s, d, m = e.sim, e.sim.data, e.sim.model
    clearance = []
    for gid in s.wheel_geoms:
        axis_z = d.geom_xmat[gid].reshape(3, 3)[2, 2]
        radius, half = m.geom_size[gid, :2]
        extent = radius*np.sqrt(max(0., 1-axis_z**2)) + half*abs(axis_z)
        p = d.geom_xpos[gid]
        clearance.append(p[2] - extent - s.support_height(float(p[0]), float(p[1])))
    rpy = Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).as_euler('xyz')
    eq = np.asarray(d.efc_type) == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    velocity_error = (s.reset_projector.jacobian(d) @ d.qvel).reshape(-1, 3)
    force, _, body_hit, _ = s.contact_state(terrain_only=True)
    return {
        **{name+'_error_m': float(v) for (name, _, _), v in zip(CLOSURES, s.reset_projector.errors(d))},
        'max_closure_error_m': float(s.reset_projector.errors(d).max()),
        'max_closure_speed_m_s': float(np.linalg.norm(velocity_error, axis=1).max()),
        'wheel_height_difference_m': float(abs(np.diff(d.site_xpos[s.wheel_sites, 2])[0])),
        'left_wheel_center_z_m': float(d.site_xpos[s.wheel_sites[0], 2]),
        'right_wheel_center_z_m': float(d.site_xpos[s.wheel_sites[1], 2]),
        'minimum_wheel_clearance_m': float(min(clearance)),
        'max_contact_penetration_m': float(max([0.] + [-c.dist for c in d.contact])),
        'body_contact': float(body_hit),
        'roll_rad': float(rpy[0]), 'pitch_rad': float(rpy[1]),
        'qvel_norm': float(np.linalg.norm(d.qvel)),
        'active_speed_norm_rad_s': float(np.linalg.norm(d.qvel[s.hip_dof_adr])),
        'passive_speed_norm_rad_s': float(np.linalg.norm(d.qvel[s.reset_projector.dofs])),
        'base_angular_speed_rad_s': float(np.linalg.norm(d.qvel[3:6])),
        'wheel_ground_force_n': float(force.max()),
        'equality_force_norm_n': float(np.linalg.norm(d.efc_force[eq])),
    }


def write_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def benchmark_env(stage, projected):
    params = env_params_for_stage(stage)
    params.robot.closed_chain_reset = projected
    return UZ05Env(stage, params=params)


def parallel_benchmark(stage, count, seeds):
    """Actual SB3 subprocess environments, without a policy or PPO training."""
    from stable_baselines3.common.vec_env import SubprocVecEnv
    result = {'num_envs': count, 'reset_batches': seeds, 'modes': {}}
    for mode in ('legacy', 'projected'):
        vec = SubprocVecEnv([partial(benchmark_env, stage, mode == 'projected')
                            for _ in range(count)], start_method='spawn')
        try:
            for seed in range(10):
                vec.seed(seed*count); vec.reset()
            timings = []
            for seed in range(seeds):
                vec.seed(seed*count)
                started = time.perf_counter(); vec.reset()
                timings.append((time.perf_counter()-started)*1000)
            steps = []
            for _ in range(100):
                started = time.perf_counter(); vec.step(np.zeros((count, 6)))
                steps.append((time.perf_counter()-started)*1000)
            result['modes'][mode] = {'reset_batch_ms': stats(timings),
                                     'step_batch_ms': stats(steps)}
        finally:
            vec.close()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', default='joint_tracking_v16')
    p.add_argument('--seeds', type=int, default=100)
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--output', type=Path, default=Path('diagnostics/reset_consistency'))
    p.add_argument('--skip-transients', action='store_true')
    p.add_argument('--parallel-envs', type=int, default=0,
                   help='Additionally benchmark this many SB3 subprocess environments')
    args = p.parse_args()
    if args.seeds < 1:
        p.error('--seeds must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    envs = {}
    for mode in ('legacy', 'projected'):
        params = env_params_for_stage(args.stage)
        params.robot.closed_chain_reset = mode == 'projected'
        envs[mode] = UZ05Env(args.stage, params=params)
        for seed in range(10):  # Warm caches, outside timed measurements.
            envs[mode].reset(seed=seed)
    reset_rows, failures, rejections = [], [], []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        # Alternate measurement order to reduce order/temperature bias.
        for mode in (('legacy', 'projected') if seed % 2 else ('projected', 'legacy')):
            e = envs[mode]
            started = time.perf_counter()
            try:
                e.reset(seed=seed)
            except RuntimeError as exc:
                failures.append({'mode': mode, 'seed': seed, 'error': str(exc)})
                continue
            ms = (time.perf_counter() - started)*1000
            reset_rows.append({'mode': mode, 'seed': seed, 'reset_ms': ms,
                               'resamples': e.reset_projection_info['resamples'],
                               'iterations': e.reset_projection_info['iterations'], **snapshot(e)})
            if e.reset_projection_info['rejections']:
                rejections.append({'mode': mode, 'seed': seed,
                                   'reasons': e.reset_projection_info['rejections']})
    write_csv(args.output / 'reset_samples.csv', reset_rows)
    summary = {'model_xml': str(MODEL_XML), 'stage': args.stage,
               'seed_start': args.seed_start, 'seeds': args.seeds, 'failures': failures,
               'rejections': rejections,
               'notes': ['PPO training not run; reward/action mapping/physics unchanged.',
                         'Domain randomization retained; no scheduled pushes in raw startup tests.',
                         'qvel_norm mixes translational m/s and angular rad/s; grouped speeds also reported.',
                         'Equality force norm is L2 of all connect constraint scalar forces.'],
               'reset': {}, 'transients': {}}
    for mode in envs:
        rows = [r for r in reset_rows if r['mode'] == mode]
        summary['reset'][mode] = {key: stats([r[key] for r in rows])
                                  for key in rows[0] if key not in ('mode', 'seed')}
        summary['reset'][mode]['failure_rate'] = sum(f['mode'] == mode for f in failures)/args.seeds
        summary['reset'][mode]['resample_total'] = sum(r['resamples'] for r in rows)

    peaks, traces = [], {}
    warnings = []
    if not args.skip_transients:
        summary['notes'].append(
            'Startup pairs use projected reset accepted base/active/wheel coordinates and velocities; '
            'legacy passive position and velocity remain at XML zero. For seeds with rejection this '
            'replaces legacy first draw with the accepted draw, avoiding a resampling confound.')
        fixed_action = np.array([.1, .1, .01, -.01, .01, -.01])
        summary['fixed_action'] = fixed_action.tolist()
        for controller in ('zero_torque', 'fixed_action'):
            for mode, e in envs.items():
                s, d, m = e.sim, e.sim.data, e.sim.model
                count = int(round(.5/m.opt.timestep))
                all_traces = []
                for seed in range(args.seed_start, args.seed_start + args.seeds):
                    e.reset(seed=seed)
                    if mode == 'legacy':
                        paired = envs['projected']
                        paired.reset(seed=seed)
                        # Match all prescribed inputs. Only passive position and
                        # velocity differ between the two reset implementations.
                        d.qpos[:] = paired.sim.data.qpos
                        d.qvel[:] = paired.sim.data.qvel
                        d.qpos[s.reset_projector.qpos] = s.reset_projector.reference
                        d.qvel[s.reset_projector.dofs] = 0.
                        mujoco.mj_forward(m, d)
                    records = []
                    for step in range(count + 1):
                        # Refresh diagnostics at the integrated qpos/qvel, not
                        # the preceding pre-integration state in mj_step.
                        mujoco.mj_forward(m, d)
                        sample = snapshot(e)
                        records.append([*sample.values(), *d.qvel[s.hip_dof_adr],
                                        *d.qvel[s.reset_projector.dofs], *d.qvel[3:6]])
                        if step == count:
                            break
                        if controller == 'zero_torque':
                            d.ctrl[:] = 0.
                        else:
                            control = e.actuators.compute(fixed_action, state={
                                'joint_pos': s.joint_positions(), 'joint_vel': s.joint_velocities(),
                                'wheel_vel': s.wheel_velocities()}, dt=float(m.opt.timestep))
                            d.ctrl[s.hip_actuator_ids] = control['leg_ctrl']
                            d.ctrl[s.wheel_actuator_ids] = control['wheel_ctrl']
                        s.step()
                    arr = np.asarray(records)
                    keys = list(sample) + [f'active_velocity_{i}' for i in range(4)] + [
                        f'passive_velocity_{i}' for i in range(8)] + [f'base_omega_{i}' for i in range(3)]
                    for window in (.1, .5):
                        a = arr[:int(round(window/m.opt.timestep)) + 1]
                        peaks.append({'controller': controller, 'mode': mode, 'seed': seed,
                                      'window_s': window,
                                      **{k: float(np.abs(a[:, i]).max()) for i, k in enumerate(keys)}})
                    bad = {str(i): int(w.number) for i, w in enumerate(d.warning) if w.number}
                    if bad or not np.isfinite(arr).all() or abs(d.time-.5) > 1e-8:
                        warnings.append({'mode': mode, 'controller': controller, 'seed': seed,
                                         'warnings': bad, 'time': float(d.time)})
                    all_traces.append(arr)
                traces[f'{controller}_{mode}'] = np.asarray(all_traces)
                print(f'completed {controller} / {mode}: {args.seeds} seeds', flush=True)
        write_csv(args.output / 'startup_peaks.csv', peaks)
        np.savez_compressed(args.output / 'startup_traces.npz', **traces,
                            columns=np.array(keys), time_s=np.arange(count+1)*m.opt.timestep)
        for controller in ('zero_torque', 'fixed_action'):
            summary['transients'][controller] = {}
            for window in (.1, .5):
                by_mode = {}
                for mode in envs:
                    rows = [r for r in peaks if r['mode'] == mode and
                            r['controller'] == controller and r['window_s'] == window]
                    by_mode[mode] = {k: stats([r[k] for r in rows]) for k in keys}
                summary['transients'][controller][str(window)] = by_mode
    summary['numerical_warnings'] = warnings
    if args.parallel_envs:
        summary['parallel_benchmark'] = parallel_benchmark(args.stage, args.parallel_envs, args.seeds)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for mode in envs:
            a = sorted(r['max_closure_error_m']*1000 for r in reset_rows if r['mode'] == mode)
            ax.plot(a, np.arange(1, len(a)+1)/len(a), label=mode)
        ax.set(xlabel='Maximum closure position error (mm)', ylabel='Empirical cumulative fraction')
        ax.legend(); fig.tight_layout(); fig.savefig(args.output/'closure_distribution.png', dpi=160)
        plt.close(fig)
        if traces:
            fig, axes = plt.subplots(2, 4, figsize=(15, 7))
            for row, controller in enumerate(('zero_torque', 'fixed_action')):
                for col, key in enumerate(('qvel_norm', 'wheel_ground_force_n', 'equality_force_norm_n', 'max_closure_error_m')):
                    ax = axes[row, col]
                    for mode in envs:
                        a = traces[f'{controller}_{mode}'][:, :, keys.index(key)]
                        line, = ax.plot(np.arange(a.shape[1])*m.opt.timestep, np.median(a, axis=0), label=mode)
                        ax.fill_between(np.arange(a.shape[1])*m.opt.timestep,
                                        np.percentile(a, 5, axis=0), np.percentile(a, 95, axis=0),
                                        color=line.get_color(), alpha=.15)
                    ax.set(title=f'{controller}\n{key}', xlabel='Time (s)')
                    ax.axvline(.1, color='gray', linestyle=':'); ax.legend()
            fig.tight_layout(); fig.savefig(args.output/'startup_comparison.png', dpi=160)
            plt.close(fig)
    except ImportError:
        summary['plot_note'] = 'Matplotlib unavailable; CSV and NPZ contain all measurements.'
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('mode | closure median/p95/max (mm) | reset median (ms) | resamples | failures')
    for mode in envs:
        v = summary['reset'][mode]
        print(mode, [round(x*1000, 8) for x in v['max_closure_error_m'].values()],
              round(v['reset_ms']['median'], 3), v['resample_total'], v['failure_rate'])
    print('numerical warnings:', warnings)
    for e in envs.values():
        e.close()


if __name__ == '__main__':
    main()
