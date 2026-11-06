"""Small, independent reward definition for the controlled simplification study."""
import numpy as np


def huber_square(x):
    x = np.abs(np.asarray(x, dtype=float))
    return np.where(x <= 1, x*x, 2*x-1)


def tracking_terms(p, *, vx_error, yaw_rate_error, height_error, roll,
                   pitch_error, heading_error, stop_distance, action_delta,
                   terminated, axle_offset, axle_enabled, dt):
    """Four groups plus one optional axle term; no gates on tracking rewards.

    Huber errors supply non-vanishing large-error slopes. The positive task and
    attitude budgets are fixed across all four arms. Failure is a one-off cost.
    """
    task = (3.0 - getattr(p,'vx_weight',1.0)*huber_square(vx_error/p.vx_scale_m_s)
            - .5*huber_square(yaw_rate_error/p.yaw_rate_scale_rad_s)
            - huber_square(height_error/p.height_scale_m)
            - .5*huber_square(stop_distance/p.stop_position_scale_m))
    attitude = (1.0 - .5*huber_square(roll/p.roll_scale_rad)
                - .5*huber_square(pitch_error/p.pitch_scale_rad)
                - .3*huber_square(heading_error/p.heading_scale_rad))
    terms = {'task_tracking': float(task)*dt, 'body_attitude': float(attitude)*dt,
             'action_smoothness': -p.smoothness_weight*float(np.square(
                 np.asarray(action_delta)/p.action_delta_scale).mean())*dt,
             'failure': -p.failure_cost if terminated else 0.0}
    if axle_enabled:
        excess = max(0., axle_offset-p.axle_deadband_m)/p.axle_scale_m
        terms['axle_alignment'] = -p.axle_weight*float(huber_square(excess))*dt
    return terms
