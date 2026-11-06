"""Evaluation-only controls. Not imported by training, env, or actuator code."""
import numpy as np
from .spec import ACTOR_OBS_BLOCKS, OBS_SCALE


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class CircularTargetLimiter:
    """One update per policy tick; observation previous_action is executed action."""
    def __init__(self, initial_q, rate, dt):
        self.target = np.array(initial_q, dtype=float).copy()
        self.rate, self.dt = rate, dt

    def apply(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1., 1.)
        if self.rate is None:
            # Preserve the unmodified action exactly, including +pi/-pi endpoint.
            return action.copy(), np.zeros(4, dtype=bool)
        delta = wrap(action[2:]*np.pi-self.target)
        bound = self.rate*self.dt
        active = np.abs(delta) > bound+1e-12
        self.target += np.clip(delta, -bound, bound)
        result = action.copy()
        result[2:] = wrap(self.target)/np.pi
        return result, active


def command_at(schedule, time_s, duration, task):
    phase = max(i for i, e in enumerate(schedule) if time_s+1e-9 >= e[0])
    start, vx, yaw, height = schedule[phase]
    # Isolate the +2->-2 reversal. Earlier motion commands remain identical.
    # Reverse-stop additionally tests the -2 launch where finetunes often fail.
    ramped = (task == 'motion' and start == 11) or (task == 'stop2_reverse' and start in (3, 10))
    if duration and ramped and phase:
        alpha = np.clip((time_s-start)/duration, 0., 1.)
        vx = schedule[phase-1][1] + alpha*(vx-schedule[phase-1][1])
    return phase, np.array([vx, 0., yaw, height, 0.])


def replace_observed_command(obs, command, clip):
    """Command has zero sensor noise: update its block without extra RNG draws."""
    offset = 0
    result = obs.copy()
    for name, width in ACTOR_OBS_BLOCKS:
        if name == 'command':
            block = np.asarray(command)*OBS_SCALE[name]
            result[offset:offset+width] = np.clip(block, -clip, clip) if clip > 0 else block
            return result
        offset += width
    raise RuntimeError('Missing command observation')
