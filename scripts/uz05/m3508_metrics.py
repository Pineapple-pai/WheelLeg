"""Predeclared posture and tracking criteria, independent of episode reward."""
import numpy as np


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def longest_run(mask, dt=.008):
    mask=np.asarray(mask,dtype=bool)
    if not mask.any():return 0.
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return float(np.max(np.flatnonzero(edges==-1)-np.flatnonzero(edges==1))*dt)


def score_interval(c,start,end,goal,thresholds,termination_reason):
    h=thresholds;t=c['time_s'];dt=.008
    mask=(t>start+1e-9)&(t<=end+1e-9)
    complete=bool(t[-1]+1e-9>=end and (termination_reason=='time_limit' or t[-1]>end+1e-9))
    out={'interval_complete':int(complete),'exposure_s':float(mask.sum()*dt),
         'posture_success':0,'tracking_success':0,'composite_success':0,
         'response_time_s':None,'response_window_start_delay_s':None,
         'speed_band_m_s':h['zero_speed_band_m_s'] if goal==0 else h['relative_speed_band']*abs(goal)}
    if not mask.any():return out
    tt=t[mask];vx=c['vx_m_s'][mask];error=vx-goal;band=out['speed_band_m_s']
    pitch=np.degrees(c['pitch_rad'][mask]);roll=np.degrees(c['roll_rad'][mask]);height=c['height_error_m'][mask]
    unsupported=(c['normal_force_n_0'][mask]<1)&(c['normal_force_n_1'][mask]<1)
    longest_pitch=longest_run(np.abs(pitch)>h['pitch_excursion_deg'])
    longest_roll=longest_run(np.abs(roll)>h['roll_excursion_deg'])
    longest_height=longest_run(np.abs(height)>h['height_excursion_m'])
    unsupported_s=longest_run(unsupported)
    posture=complete and rms(pitch)<=h['pitch_rms_deg'] and rms(roll)<=h['roll_rms_deg'] and np.abs(height).mean()<=h['height_mae_m'] and max(longest_pitch,longest_roll,longest_height)<=h['max_excursion_duration_s']+1e-9 and unsupported_s<=h['max_both_wheels_unloaded_s']+1e-9
    settle_samples=round(h['settle_window_s']/dt)
    response=None
    if len(error)>=settle_samples:
        kernel=np.ones(settle_samples)/settle_samples
        occupancy=np.convolve((np.abs(error)<=band+1e-9).astype(float),kernel,'valid')
        mean_square=np.convolve(error**2,kernel,'valid')
        window_end=tt[settle_samples-1:]
        valid=(occupancy>=h['settle_occupancy']-1e-9)&(mean_square<=band**2+1e-12)&(window_end<=min(start+h['response_deadline_s'],end)+1e-9)
        if valid.any():response=float(window_end[np.flatnonzero(valid)[0]]-start)
    steady_mask=(tt>end-h['steady_window_s']+1e-9)&(tt<=end+1e-9)
    steady_error=error[steady_mask];steady_vx=vx[steady_mask]
    steady_complete=complete and steady_mask.sum()*dt>=h['steady_window_s']-1e-9
    tracking=False
    if len(steady_error) and steady_complete:
        steady_rms=rms(steady_error);occupancy=float((np.abs(steady_error)<=band+1e-9).mean())
        steady_ok=steady_complete and steady_rms<=band+1e-9 and occupancy>=h['settle_occupancy']-1e-9
        tracking=steady_ok and response is not None
        out.update(steady_complete=int(steady_complete),steady_mean_vx_m_s=float(steady_vx.mean()),
            steady_bias_m_s=float(steady_error.mean()),steady_mae_m_s=float(np.abs(steady_error).mean()),
            steady_rmse_m_s=steady_rms,steady_std_vx_m_s=float(steady_vx.std()),steady_inband_fraction=occupancy,
            steady_tracking_success=int(steady_ok),
            steady_1p90_to_1p95=int(goal==2 and steady_complete and 1.90-1e-9<=steady_vx.mean()<=1.95+1e-9))
    out.update(posture_success=int(posture),tracking_success=int(tracking),composite_success=int(posture and tracking),
        response_time_s=response,response_window_start_delay_s=None if response is None else response-h['settle_window_s'],
        pitch_rms_deg=rms(pitch),pitch_std_deg=float(pitch.std()),pitch_peak_deg=float(np.abs(pitch).max()),
        roll_rms_deg=rms(roll),height_mae_m=float(np.abs(height).mean()),height_peak_m=float(np.abs(height).max()),
        longest_pitch_excursion_s=longest_pitch,longest_roll_excursion_s=longest_roll,
        longest_height_excursion_s=longest_height,longest_unsupported_s=unsupported_s,
        vx_goal_rmse_m_s=rms(error),vx_reference_rmse_m_s=rms(vx-c['command_vx_m_s'][mask]),
        common_x_rms_m=rms(c['common_x_m'][mask]),common_x_peak_m=float(np.abs(c['common_x_m'][mask]).max()),
        common_outside_008_longest_s=longest_run(np.abs(c['common_x_m'][mask])>h['common_offset_descriptor_m']),
        differential_x_rms_m=rms(c['differential_x_m'][mask]),
        common_steady_rms_m=rms(c['common_x_m'][mask][steady_mask]) if steady_mask.any() else None,
        pitch_rate_peak_rad_s=float(np.abs(c['pitch_rate_rad_s'][mask]).max()),
        interval_start_s=start,interval_end_s=end)
    return out
