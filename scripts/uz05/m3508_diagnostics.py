"""Opt-in diagnostic models, never imported by training or production control.

Control current is the legacy speed-loop variable, not a calibrated measurement
of DC bus or winding current. C/D explicitly hypothesize a changed effective
output gain; they do NOT silently raise the 10 A control-variable limit.
"""
from dataclasses import asdict, dataclass
import numpy as np
from .actuators import WheelEscController


@dataclass(frozen=True)
class TorqueModel:
    name: str
    peak_nm: float
    interpretation: str
    burst_seconds: float = 0.

    def to_dict(self):
        return {**asdict(self), 'control_current_limit_a': 10.,
                'peak_output_nm_per_control_amp': self.peak_nm/10.,
                'equivalent_current_at_peak_using_0246_a': self.peak_nm/.246}


MODELS = {
    'A_legacy': TorqueModel('A_legacy', 1.722, 'Exact original I*0.246*0.70, cap 2.46, current limit 10.'),
    'B_rated': TorqueModel('B_rated', 2.46, 'Rated-point approximation I_control*0.246; current definition uncalibrated.'),
    'C_burst': TorqueModel('C_burst', 3., '0.5s post-vx-command-change gain envelope, then B. Duration/gain are hypotheses, not hardware ratings.', .5),
    'D_stall_bound': TorqueModel('D_stall_bound', 3.69, 'Optimistic constant all-speed gain ceiling ONLY for sensitivity; not sustainable or a physical speed-torque model.'),
}


class DiagnosticWheelEsc(WheelEscController):
    def __init__(self, params, model='A_legacy'):
        super().__init__(params)
        self.torque_model = MODELS[model]
        self.reset()

    def reset(self):
        super().reset()
        self.time_s = 0.
        self.boost_until = -np.inf
        self.last_goal = 0.

    def begin_control(self, time_s, requested_vx):
        self.time_s = float(time_s)
        if abs(requested_vx-self.last_goal) > 1e-9:
            self.boost_until = time_s+self.torque_model.burst_seconds
        self.last_goal = float(requested_vx)

    @property
    def effective_cap_nm(self):
        if self.torque_model.name == 'C_burst' and self.time_s+1e-12 >= self.boost_until:
            return 2.46
        return self.torque_model.peak_nm

    def torque_from_current(self, current):
        if self.torque_model.name == 'A_legacy':
            return super().torque_from_current(current)  # exact historical arithmetic
        cap = self.effective_cap_nm
        # No change to the inherited speed loop or its current clipping.
        return np.clip(np.asarray(current)*(cap/10.), -cap, cap)


def reference_at(schedule, time_s, name, previous_vx, dt, task):
    """Only named task events are shaped; unchanged pre-event history is paired."""
    phase=max(i for i,e in enumerate(schedule) if time_s+1e-9>=e[0])
    start,vx,yaw,height=schedule[phase]
    focal=(task=='motion' and start==11) or (task=='forward2' and start==3) or (task=='stop2_reverse' and start in (3,10))
    if focal and phase and name.startswith('ramp_'):
        duration=float(name.split('_')[1])
        alpha=np.clip((time_s-start)/duration,0.,1.)
        vx=schedule[phase-1][1]+alpha*(vx-schedule[phase-1][1])
    elif focal and name.startswith('accel_'):
        bound=float(name.split('_')[1])*dt
        vx=previous_vx+np.clip(vx-previous_vx,-bound,bound)
    return phase,np.array([vx,0.,yaw,height,0.]),float(schedule[phase][1])
