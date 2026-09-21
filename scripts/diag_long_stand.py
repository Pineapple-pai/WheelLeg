import numpy as np, sys
sys.path.insert(0,'.')
from scipy.spatial.transform import Rotation
from uz05.env import UZ05Env
from dataclasses import replace
for tag,dr in (("no-dr",False),("dr",True)):
    env=UZ05Env(stage='stand',stand_level=2,assist_scale=0.0,seed=0,init_scale=1.0,
                lock_stand_leg_actions=False,stand_leg_action_limit=1.0,coord_mix=1.0)
    if not dr: env.params.domain_randomization.enabled=False
    env.stage = replace(env.stage, episode_steps=7500)
    env.reset(seed=0)
    th=[];xs=[];ys=[];I=[];n=0;info={}
    for k in range(7500):
        o,r,t,tr,info=env.step(np.zeros(6))
        q=env.sim.base_quat; rpy=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_euler('xyz')
        th.append(float(rpy[1])); xs.append(float(env.sim.data.qpos[0]-env.nominal_xy[0]))
        ys.append(float(env.sim.data.qpos[1]-0.0)); I.append(float(info['wheel_current_left'])); n=k+1
        if t or tr: break
    env.close()
    th=np.asarray(th);xs=np.asarray(xs);ys=np.asarray(ys);I=np.asarray(I)
    print(f"[{tag}] n={n} (={n*0.008:.0f}s) term={info['termination_reason']} "
          f"prms={np.degrees(np.sqrt(np.mean(th**2))):.3f}deg ptp={np.degrees(th.max()-th.min()):.3f}deg "
          f"xmax={100*np.abs(xs).max():.3f}cm xfin={100*xs[-1]:+.4f}cm ymax={100*np.abs(ys).max():.3f}cm "
          f"Irms={np.sqrt(np.mean(I**2)):.3f}A dI={np.sqrt(np.mean(np.diff(I)**2)):.4f}")
