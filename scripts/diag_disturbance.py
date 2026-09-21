"""扰动恢复：整机速度冲量 + 持续外力，全部走 env 内置协同控制器。"""
import numpy as np, sys
sys.path.insert(0,'.')
from scipy.spatial.transform import Rotation
from uz05.env import UZ05Env
from dataclasses import replace

def run(imp=0.0, force=0.0, force_steps=25, steps=1500, seed=0):
    env=UZ05Env(stage='stand',stand_level=2,assist_scale=0.0,seed=seed,init_scale=1.0,
                lock_stand_leg_actions=False,stand_leg_action_limit=1.0,coord_mix=1.0)
    env.params.domain_randomization.enabled=False
    env.stage=replace(env.stage,episode_steps=steps)
    env.reset(seed=seed)
    if imp: env.sim.data.qvel[0] += imp
    th=[];xs=[];I=[];n=0;info={}
    for k in range(steps):
        if force and k<force_steps: env.sim.data.xfrc_applied[env.sim.chassis_body,0]=force
        else: env.sim.data.xfrc_applied[env.sim.chassis_body,0]=0.0
        o,r,t,tr,info=env.step(np.zeros(6))
        q=env.sim.base_quat; rpy=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_euler('xyz')
        th.append(float(rpy[1])); xs.append(float(env.sim.data.qpos[0]-env.nominal_xy[0]))
        I.append(float(info['wheel_current_left'])); n=k+1
        if t or tr: break
    env.close()
    th=np.asarray(th);xs=np.asarray(xs);I=np.asarray(I)
    rec = float(100*np.abs(xs[force_steps:]).max()) if force and n>force_steps else float('nan')
    return dict(n=n,surv=n>=steps,term=info['termination_reason'],
        prms=float(np.degrees(np.sqrt(np.mean(th**2)))),ptp=float(np.degrees(th.max()-th.min())),
        xpeak=float(100*np.abs(xs).max()),xfin=float(100*xs[n-1]),xafter=rec,
        Irms=float(np.sqrt(np.mean(I**2))))

print("=== 瞬时速度冲量 ===")
for imp in (0.0,0.05,0.10,0.20,0.40):
    for sg in (1.0,-1.0):
        d=run(imp=sg*imp)
        print(f"J={sg*imp:+.2f}m/s n={d['n']:4d} surv={int(d['surv'])} prms={d['prms']:6.3f} "
              f"ptp={d['ptp']:6.3f} xpeak={d['xpeak']:6.2f}cm xfin={d['xfin']:+6.2f}cm Irms={d['Irms']:5.2f} {d['term']}")
print("=== 持续外力 0.16 s（等效推力）===")
for f in (5.0,10.0,20.0,40.0):
    for sg in (1.0,-1.0):
        d=run(force=sg*f)
        print(f"F={sg*f:+5.1f}N n={d['n']:4d} surv={int(d['surv'])} prms={d['prms']:6.3f} "
              f"xpeak={d['xpeak']:6.2f}cm 推后峰值={d['xafter']:6.2f}cm xfin={d['xfin']:+6.2f}cm {d['term']}")
