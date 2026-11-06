"""Optional read-only instrumentation for reset A/B experiments."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .env import UZ05Env


def fingerprint(value):
    h = hashlib.sha256()
    def walk(v):
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            h.update(str((v.shape, str(v.dtype))).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode()); walk(v[k])
        elif isinstance(v, (list, tuple)):
            for x in v: walk(x)
        else:
            h.update(repr(v).encode())
    walk(value)
    return h.hexdigest()


def initial_state_audit(model, env, path):
    p = model.policy
    result = {
        'policy_sha256': fingerprint(p.state_dict()),
        'actor_sha256': fingerprint([p.mlp_extractor.policy_net.state_dict(), p.action_net.state_dict(), p.log_std]),
        'actor_mean_sha256': fingerprint([p.mlp_extractor.policy_net.state_dict(), p.action_net.state_dict()]),
        'action_std': p.log_std.detach().exp().cpu().tolist() if p.log_std is not None else None,
        'critic_sha256': fingerprint([p.mlp_extractor.value_net.state_dict(), p.value_net.state_dict()]),
        'optimizer_sha256': fingerprint(p.optimizer.state_dict()),
        'optimizer_step_values': sorted(set(float(v['step']) for v in p.optimizer.state.values())),
        'reward_normalization_sha256': fingerprint(vars(env.ret_rms)),
        'observation_normalization_enabled': env.norm_obs,
        'observation_normalization_sha256': fingerprint(vars(env.obs_rms)) if hasattr(env, 'obs_rms') else None,
        'learning_rate_schedule_0_05_1': [float(model.lr_schedule(x)) for x in (0., .5, 1.)],
        'clip_schedule_0_05_1': [float(model.clip_range(x)) for x in (0., .5, 1.)],
        'ppo_epoch_update_count': model._n_updates,
        'source_num_timesteps_before_learn': model.num_timesteps,
        'source_progress_before_learn': model._current_progress_remaining,
        'stage': env.get_attr('stage')[0].name,
        'not_restored': ['MuJoCo episode state', 'environment RNG state', 'Python/NumPy/Torch RNG state',
                         'rollout buffer', 'VecNormalize per-environment discounted returns'],
        'learn_behavior': 'Local timesteps/progress restarted; constant LR/clip schedules unchanged; PPO epoch counter retained.',
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2)+'\n')


class AuditedResetEnv(UZ05Env):
    """Write initial states without drawing randomness or altering observations."""
    def __init__(self, *args, audit_path, **kwargs):
        self._audit_file = None
        super().__init__(*args, **kwargs)
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_file = path.open('w')
        self._audit_writer = None
        self._audit_index = 0

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        if self._audit_file is not None:
            s, d, m = self.sim, self.sim.data, self.sim.model
            rpy = Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).as_euler('xyz')
            row = {'reset_index': self._audit_index, 'explicit_seed': kwargs.get('seed'),
                   'roll_rad': rpy[0], 'pitch_rad': rpy[1],
                   **{f'active_q_{i}': q for i, q in enumerate(s.joint_positions())},
                   'wheel_z_left_m': d.site_xpos[s.wheel_sites[0], 2],
                   'wheel_z_right_m': d.site_xpos[s.wheel_sites[1], 2],
                   'initial_contacts': d.ncon,
                   'penetration_m': max([0.] + [-c.dist for c in d.contact]),
                   'wheel_ground_force_n': float(s.contact_state(terrain_only=True)[0].max()),
                   'resamples': self.reset_projection_info['resamples'],
                   'rejections': json.dumps(self.reset_projection_info['rejections']),
                   'max_closure_error_m': float(s.reset_projector.errors(d).max())}
            if self._audit_writer is None:
                self._audit_writer = csv.DictWriter(self._audit_file, fieldnames=list(row))
                self._audit_writer.writeheader()
            self._audit_writer.writerow(row); self._audit_file.flush()
            self._audit_index += 1
        return result

    def close(self):
        if self._audit_file is not None:
            self._audit_file.close()
            self._audit_file = None
        return super().close()
