"""Opt-in command/reset sampling experiments; no reward, action or plant edits."""
from dataclasses import replace
import hashlib
from pathlib import Path
import numpy as np
from .env import UZ05Env

SAMPLING_NAMES=('original','low_band','launch_focus','startup_balanced','startup_low','startup_mixed_ramp','startup_stratified')
STARTUP_STAGES=('joint_tracking_startup','joint_tracking_startup_precision','joint_tracking_startup_precision_fine')
STARTUP_SAMPLERS=('startup_balanced','startup_low','startup_mixed_ramp','startup_stratified')
STARTUP_GOALS={'startup_balanced':[0.,-.1,.1,-.25,.25,-.5,.5],
               'startup_low':[0.,-.1,.1,-.25,.25]}
STARTUP_PROBABILITIES={'startup_balanced':[.25,.125,.125,.125,.125,.125,.125],
                       'startup_low':[.25,.1875,.1875,.1875,.1875]}
STARTUP_GOALS['startup_mixed_ramp']=STARTUP_GOALS['startup_balanced'].copy()
STARTUP_PROBABILITIES['startup_mixed_ramp']=STARTUP_PROBABILITIES['startup_balanced'].copy()
STARTUP_GOALS['startup_stratified']=STARTUP_GOALS['startup_balanced'].copy()
STARTUP_PROBABILITIES['startup_stratified']=STARTUP_PROBABILITIES['startup_balanced'].copy()
STRATIFIED_GOALS=(0.,0.,-.1,.1,-.25,.25,-.5,.5)
RAMP_ACCEL_M_S2=.25


def sampling_contract(name):
    if name not in SAMPLING_NAMES:raise ValueError(name)
    common={'name':name,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'physical_parameters':'unchanged','reward':'unchanged','actuator_mapping':'unchanged',
            'domain_randomization':'unchanged','pushes':'unchanged','reference':'step, no smoothing'}
    if name=='original':return {**common,'sampling':'unmodified stage command/reset distribution'}
    if name=='low_band':return {**common,
        'sampling':'original draws; nonzero vx -> sign(vx)*(0.25+0.25*clip(abs(vx)/2,0,1))',
        'zero_commands':'unchanged','height_yaw_holds_moving_starts':'unchanged'}
    if name in STARTUP_SAMPLERS:return {**common,
        'sampling':('each24s episode:0-3s zero,3-24s held goal; rank-offset cyclic eight-goal strata, two zero slots'
                    if name=='startup_stratified' else 'each24s episode:0-3s zero,3-24s held goal; independent stand episodes'),
        'required_stages':list(STARTUP_STAGES),
        'goals_m_s':STARTUP_GOALS[name],
        'goal_probabilities':STARTUP_PROBABILITIES[name],
        'height_m':.27,'yaw_rad_s':0.,'moving_start_probability':0.,
        'pushes':'disabled by startup stage; not a single-factor ablation',
        'reference':('50% step / 50% ramp at0.25m/s^2, selected once at launch; reward tracks observed reference'
                     if name=='startup_mixed_ramp' else 'step, no smoothing'),
        'interpretation':'foundation task distribution; all six PPO actions free, no action smoothing'}
    return {**common,'sampling':'each24s episode:0-3s zero,3-24s held goal',
        'goals_m_s':[-.5,-.25,.25,.5],'goal_probabilities':[.25]*4,
        'height_m':.27,'yaw_rad_s':0.,'moving_start_probability':0.,
        'remaining_initial_state_distribution':'unchanged',
        'interpretation':'bundled launch-task and reset sampling; not a speed-only ablation'}


class SamplingEnv(UZ05Env):
    def __init__(self,*args,command_sampling,sampling_rank=0,**kwargs):
        if command_sampling not in ('low_band','launch_focus',*STARTUP_SAMPLERS):raise ValueError(command_sampling)
        self.command_sampling=command_sampling
        self.sampling_rank=int(sampling_rank)
        self._sampling_episode=-1
        self._launch_goal=None
        self._launch_ramp=None
        super().__init__(*args,**kwargs)
        if not self.stage.continuous_position:raise ValueError('sampling experiments require continuous-position stage')
        if command_sampling in STARTUP_SAMPLERS and self.stage.name not in STARTUP_STAGES:
            raise ValueError('startup samplers require a startup stage')
        if command_sampling=='launch_focus':
            self.stage=replace(self.stage,moving_start_probability=0.)

    def reset(self,**kwargs):
        self._sampling_episode+=1
        self._launch_goal=None
        self._launch_ramp=None
        return super().reset(**kwargs)

    def _sample_command(self):
        if self.command_sampling=='low_band':
            super()._sample_command()
            vx=float(self.command_target[0])
            if abs(vx)>1e-12:
                vx=float(np.sign(vx)*(.25+.25*np.clip(abs(vx)/2.,0.,1.)))
                self.command_target[0]=vx
                self.command[0]=vx
            return
        if self._launch_goal is None:
            self._launch_goal=(STRATIFIED_GOALS[(self.sampling_rank+self._sampling_episode)%len(STRATIFIED_GOALS)]
                               if self.command_sampling=='startup_stratified' else
                               float(self.rng.choice(STARTUP_GOALS[self.command_sampling],
                                p=STARTUP_PROBABILITIES[self.command_sampling]))
                               if self.command_sampling in STARTUP_SAMPLERS else
                               float(self.rng.choice([-.5,-.25,.25,.5])))
        launch_step=round(3./self.params.control_dt)
        vx=0. if self.steps<launch_step else self._launch_goal
        ramping=False
        if self.command_sampling=='startup_mixed_ramp' and self.steps>=launch_step:
            if self._launch_ramp is None:self._launch_ramp=bool(self.rng.random()<.5)
            if self._launch_ramp:
                vx=float(np.sign(vx)*min(abs(vx),RAMP_ACCEL_M_S2*(self.steps-launch_step)*self.params.control_dt))
                ramping=abs(vx)<abs(self._launch_goal)
        self.command_target[:]=[0. if self.steps<launch_step else self._launch_goal,0.,0.,.27,0.]
        self.command[:]=[vx,0.,0.,.27,0.]
        self.command_steps_left=max(1,(launch_step-self.steps) if self.steps<launch_step
                                    else 1 if ramping else self.stage.episode_steps-self.steps+1)
