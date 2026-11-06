"""Evaluate frozen reset A/B runs and build paired CSVs, plots, and a report."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from stable_baselines3.common.save_util import load_from_zip_file

from ppo_reset_ab_evaluation import evaluate, COLUMNS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'diagnostics/ppo_reset_ab'
METRICS = ['success', 'survival_s', 'pitch_rms_deg', 'roll_rms_deg', 'height_mae_m',
           'axle_mean_m', 'common_x_rms_m', 'common_x_peak_m', 'differential_x_rms_m',
           'vx_rmse_m_s', 'joint_target_rate_rms_rad_s', 'joint_saturation_fraction',
           'wheel_saturation_fraction', 'joint_target_lag_rms_rad', 'common_x_outside_008_fraction']
METRICS += ['longest_joint_full_cycle_saturation_s', 'longest_wheel_full_cycle_saturation_s']
TAGS = {'reward': 'rollout/ep_rew_mean', 'episode_length': 'rollout/ep_len_mean',
        'pitch_rms_deg': 'Posture/pitch_rms_deg', 'roll_rms_deg': 'Posture/roll_rms_deg',
        'height_rmse_m': 'Tracking/height_rmse', 'policy_loss': 'train/policy_gradient_loss',
        'value_loss': 'train/value_loss', 'approx_kl': 'train/approx_kl',
        'training_survival_rate': 'Episode/survive_rate', 'fps': 'time/fps'}


def get_training(config):
    rows = []
    for run in config['runs']:
        root = Path(run['directory'])/'tensorboard'
        events = list(root.glob('PPO_*'))
        assert len(events) == 1, events
        ea = EventAccumulator(str(events[0]), size_guidance={'scalars': 0}); ea.Reload()
        values = {}
        for label, tag in TAGS.items():
            if tag not in ea.Tags()['scalars']: continue
            for point in ea.Scalars(tag):
                update = int(round(point.step/config['steps_per_update']))
                # SB3 dumps rollout statistics before train(); train() and our
                # post-update callback are therefore flushed at the next dump.
                if not tag.startswith(('rollout/', 'time/')):
                    update -= 1
                if update < 1: continue
                values.setdefault(update, {})[label] = point.value
        update = None
        for line in (Path(run['directory'])/'train.log').read_text().splitlines():
            match = re.search(r'Learning iteration (\d+)/', line)
            if match: update = int(match.group(1))
            match = re.search(r'Computation: (\d+) steps/s \(collection: ([\d.]+)s, learning ([\d.]+)', line)
            if match and update is not None:
                values.setdefault(update, {}).update(fps=float(match[1]),
                    collection_seconds=float(match[2]), learning_seconds=float(match[3]))
            for label, field in (('Mean surrogate loss', 'policy_loss'),
                                  ('Mean value_function loss', 'value_loss'),
                                  ('Mean kl loss', 'approx_kl'), ('Mean reward', 'reward'),
                                  ('Mean episode length', 'episode_length'),
                                  ('Posture/pitch_rms_deg', 'pitch_rms_deg'),
                                  ('Posture/roll_rms_deg', 'roll_rms_deg'),
                                  ('Episode/survive_rate', 'training_survival_rate')):
                match = re.search(re.escape(label)+r':\s*([+\-\d.eE]+)', line)
                if match and update is not None and field not in values.setdefault(update, {}):
                    # Last update has not been TensorBoard-flushed; terminal
                    # fallback is rounded, explicitly recorded below.
                    values[update][field] = float(match[1])
        for update, row in sorted(values.items()):
            rows.append({'train_seed': run['seed'], 'mode': run['mode'], 'update': update,
                         'environment_steps': update*config['steps_per_update'],
                         'terminal_fallback_last_update': update == 200, **row})
    return pd.DataFrame(rows)


def initial_distributions(config):
    rows, samples = [], []
    columns = ['roll_rad', 'pitch_rad', *[f'active_q_{i}' for i in range(4)],
               'wheel_z_left_m', 'wheel_z_right_m', 'penetration_m', 'initial_contacts',
               'wheel_ground_force_n', 'resamples', 'max_closure_error_m']
    for run in config['runs']:
        frames = []
        for path in (Path(run['directory'])/'resets').glob('worker_*.csv'):
            frame = pd.read_csv(path)
            frame = frame[frame['reset_index'] >= 1].copy()  # Exclude factory pre-load reset.
            frame['worker'] = path.stem
            frames.append(frame)
        a = pd.concat(frames, ignore_index=True)
        a['train_seed'] = run['seed']; a['mode'] = run['mode']
        samples.append(a)
        for key in columns:
            v = a[key].to_numpy()
            rows.append({'train_seed': run['seed'], 'mode': run['mode'], 'variable': key,
                         'count': len(v), 'mean': float(v.mean()), 'std': float(v.std()),
                         'min': float(v.min()), 'median': float(np.median(v)),
                         'p05': float(np.percentile(v, 5)), 'p95': float(np.percentile(v, 95)),
                         'max': float(v.max()), 'nonzero_fraction': float((v != 0).mean())})
    pd.DataFrame(rows).to_csv(OUT/'reset_distribution.csv', index=False)
    all_samples = pd.concat(samples, ignore_index=True)
    all_samples.to_csv(OUT/'reset_initial_samples.csv', index=False)
    return all_samples


def paired_results(evaluation, primary):
    final = evaluation[(evaluation['update'] == 200) & (evaluation['mode'] != 'baseline')]
    averaged = final.groupby(['train_seed', 'mode', 'task'])[METRICS].mean().reset_index()
    rows = []
    for seed in (1, 2, 3):
        for task in averaged.task.unique():
            a = averaged[(averaged.train_seed == seed) & (averaged.task == task)]
            for metric in METRICS:
                old = float(a[a['mode'] == 'legacy'][metric].iloc[0])
                new = float(a[a['mode'] == 'projected'][metric].iloc[0])
                rows.append({'train_seed': seed, 'task': task, 'metric': metric,
                             'legacy': old, 'projected': new, 'difference': new-old})
        for metric in METRICS:
            a = averaged[(averaged.train_seed == seed) & averaged.task.isin(primary)]
            old, new = [float(a[a['mode'] == mode][metric].mean()) for mode in ('legacy', 'projected')]
            rows.append({'train_seed': seed, 'task': 'primary_mean', 'metric': metric,
                         'legacy': old, 'projected': new, 'difference': new-old})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT/'paired_results.csv', index=False)
    return averaged, pairs


def convergence_metrics(config, training, evaluation):
    rows = []
    baseline = evaluation[(evaluation['mode'] == 'baseline') & (evaluation.task == 'stand')].success.mean()
    for run in config['runs']:
        a = training[(training.train_seed == run['seed']) & (training['mode'] == run['mode'])]
        stand = evaluation[(evaluation.train_seed == run['seed']) &
                           (evaluation['mode'] == run['mode']) & (evaluation.task == 'stand')]
        curve = stand.groupby('update').success.mean().sort_index()
        xs = np.r_[0, curve.index.to_numpy()]; ys = np.r_[baseline, curve.to_numpy()]
        row = {'train_seed': run['seed'], 'mode': run['mode'],
               'stand_success_auc_normalized': float(np.trapz(ys, xs)/200),
               'wall_seconds': json.loads((Path(run['directory'])/'completed.json').read_text())['seconds']}
        row['wall_steps_per_second'] = config['steps_per_run']/row['wall_seconds']
        for key in ('reward', 'episode_length', 'training_survival_rate', 'height_rmse_m', 'pitch_rms_deg', 'roll_rms_deg', 'approx_kl'):
            row[key+'_first50'] = float(a[a['update'] <= 50][key].mean())
            row[key+'_last50'] = float(a[a['update'] > 150][key].mean())
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'convergence_metrics.csv', index=False)
    return result


def historical_curves():
    import matplotlib.pyplot as plt
    ea = EventAccumulator(str(ROOT/'runs/joint_tracking_pos_v16/PPO_1'), size_guidance={'scalars': 0})
    ea.Reload()
    fields = [k for k in TAGS if k != 'fps']
    frames = []
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, key in zip(axes.ravel(), fields):
        tag = TAGS[key]
        if tag not in ea.Tags()['scalars']: continue
        points = ea.Scalars(tag)
        a = pd.DataFrame({'environment_steps': [v.step for v in points], 'value': [v.value for v in points]})
        a['metric'] = key; frames.append(a)
        ax.plot(a.environment_steps, a.value.rolling(50, min_periods=1).mean())
        ax.set(title=key, xlabel='v16 environment steps'); ax.grid(alpha=.2)
    fig.suptitle('Historical v16 training; 50-record rolling mean; survival is mixed-task training survival')
    fig.tight_layout(); fig.savefig(OUT/'historical_v16_curves.png', dpi=160); plt.close(fig)
    pd.concat(frames, ignore_index=True).to_csv(OUT/'historical_v16_metrics.csv', index=False)


def plots(training, evaluation, averaged, config, resets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {1: '#2878b5', 2: '#d27a15', 3: '#38935b'}
    styles = {'legacy': '--', 'projected': '-'}
    fields = ['reward', 'episode_length', 'stand_success', 'pitch_rms_deg', 'roll_rms_deg',
              'height_rmse_m', 'policy_loss', 'value_loss', 'approx_kl']
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, field in zip(axes.ravel(), fields):
        for seed in (1, 2, 3):
            for mode in styles:
                a = training[(training.train_seed == seed) & (training['mode'] == mode)]
                if field == 'stand_success':
                    b = evaluation[(evaluation.train_seed == seed) & (evaluation['mode'] == mode) & (evaluation.task == 'stand')]
                    a = b.groupby('update').success.mean().reset_index(name='stand_success')
                    baseline = evaluation[(evaluation['mode'] == 'baseline') & (evaluation.task == 'stand')].success.mean()
                    a = pd.concat([pd.DataFrame({'update': [0], 'stand_success': [baseline]}), a])
                    ax.plot(a['update'], a[field], styles[mode], color=colors[seed], marker='o',
                            label=f'seed {seed} {mode}')
                elif field in a:
                    ax.plot(a['update'], a[field].rolling(5, min_periods=1).mean(), styles[mode],
                            color=colors[seed], label=f'seed {seed} {mode}')
        ax.set(title=field, xlabel='Continuation PPO updates'); ax.grid(alpha=.2)
        ax.legend(fontsize=6)
    fig.suptitle('Each paired training seed; training scalars: 5-update rolling mean; stand: fixed projected-reset evaluation')
    fig.tight_layout(); fig.savefig(OUT/'training_curves.png', dpi=170); plt.close(fig)

    tasks = config['evaluation']['primary_tasks'] + config['evaluation']['additional_tasks']
    fields = ['success', 'survival_s', 'pitch_rms_deg', 'roll_rms_deg', 'height_mae_m',
              'axle_mean_m', 'common_x_rms_m', 'joint_target_rate_rms_rad_s', 'joint_saturation_fraction']
    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    for ax, field in zip(axes.ravel(), fields):
        for seed in (1, 2, 3):
            for mode in styles:
                a = averaged[(averaged.train_seed == seed) & (averaged['mode'] == mode)].set_index('task').reindex(tasks)
                ax.plot(np.arange(len(tasks)), a[field], styles[mode], color=colors[seed], marker='o',
                        markersize=3, label=f'{seed} {mode}')
        ax.set_xticks(np.arange(len(tasks)), tasks, rotation=35, ha='right')
        ax.set_title(field); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Final policy: each marker is one training seed, mean over five evaluation seeds; failures included')
    fig.tight_layout(); fig.savefig(OUT/'evaluation_comparison.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    for row, seed in enumerate((1, 2, 3)):
        for mode in styles:
            path = OUT/f'seed_{seed}'/mode/'eval_200'/'stop2_reverse_51000.npz'
            a = np.load(path)['samples']
            for col, field in enumerate(('common_x_m', 'pitch_rad', 'joint_saturation')):
                axes[row, col].plot(a[:, 0], a[:, COLUMNS.index(field)], styles[mode], label=mode)
                axes[row, col].set(title=f'train seed {seed}: {field}', xlabel='Time (s)')
                axes[row, col].axvline(3, color='gray', linestyle=':')
                axes[row, col].axvline(10, color='gray', linestyle=':')
                axes[row, col].legend(); axes[row, col].grid(alpha=.2)
    fig.suptitle('Representative paired evaluation seed 51000: reverse to -2 m/s at 3 s, stop at 10 s; lines end on failure')
    fig.tight_layout(); fig.savefig(OUT/'joint_motion_comparison.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(14, 11))
    for col, mode in enumerate(styles):
        a = np.load(OUT/'seed_1'/mode/'eval_200'/'reverse10_51000.npz')['samples']
        for j in range(4):
            axes[j, col].plot(a[:, 0], a[:, COLUMNS.index(f'actual_q_{j}')], label='actual')
            axes[j, col].plot(a[:, 0], a[:, COLUMNS.index(f'target_q_{j}')], '--', label='target')
            axes[j, col].set(title=f'{mode}: joint {j} (rad)', xlabel='Time (s)'); axes[j, col].legend()
    fig.suptitle('Targets and actual unwrapped joint angles; train seed 1, evaluation seed 51000, reverse10')
    fig.tight_layout(); fig.savefig(OUT/'joint_targets.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, key in zip(axes.ravel(), ('roll_rad', 'pitch_rad', 'active_q_0', 'active_q_1', 'wheel_z_left_m', 'wheel_z_right_m')):
        for seed in (1, 2, 3):
            for mode in styles:
                a = resets[(resets.train_seed == seed) & (resets['mode'] == mode)][key]
                counts, edges = np.histogram(a, bins=20, density=True)
                ax.plot((edges[:-1]+edges[1:])/2, counts, styles[mode], color=colors[seed], label=f'{seed} {mode}')
        ax.set_title(key); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT/'reset_distributions.png', dpi=170); plt.close(fig)


def report(config, training, evaluation, averaged, pairs, resets, convergence):
    # Only final success metrics enter the decision; rules were frozen before training.
    select = lambda task, metric: pairs[(pairs.task == task) & (pairs.metric == metric)]
    primary = select('primary_mean', 'success')
    common = select('primary_mean', 'common_x_rms_m')
    all_task_success = pairs[(pairs.metric == 'success') & (pairs.task != 'primary_mean')]
    no_task_retreat = all(all_task_success.groupby('task').difference.median() > -.10+1e-9)
    superior = (primary.difference >= .10-1e-9).all() and no_task_retreat and (
        common.projected <= .9*common.legacy).all()
    retreat = (primary.difference <= -.10+1e-9).all()
    for task in ('stand', 'stop2_reverse'):
        a = select(task, 'success')
        retreat = retreat or (a.difference.median() <= -.20+1e-9 and (a.difference < 0).all())
    decision = 'A' if superior else ('C' if retreat else 'B')
    decision_explanation = {
        'A': 'Projected 在预先固定的任务与共同摆腿指标上跨三个种子稳定改善，可保留并考虑完整训练；小样本仍有不确定性。',
        'B': '没有达到跨三个种子稳定改善或明显退化的预设标准。保留修复作为物理一致性改进，继续排查控制和奖励，不盲目延长训练。',
        'C': 'Projected 达到预设的跨种子退化标准，暂不完整重训，优先分析初始化分布、接触和旧策略兼容性。',
    }[decision]
    audits = [json.loads((Path(r['directory'])/'initial_state.json').read_text()) for r in config['runs']]
    assert all(a == audits[0] for a in audits), 'Initial training states differ!'
    final_counters = []
    for run in config['runs']:
        data, params, _ = load_from_zip_file(run['checkpoint']+'.zip', device='cpu')
        steps = sorted(set(float(v['step']) for v in params['policy.optimizer']['state'].values()))
        assert data['num_timesteps'] == config['steps_per_run']
        assert data['_n_updates'] == audits[0]['ppo_epoch_update_count'] + 200*5
        assert steps == [audits[0]['optimizer_step_values'][0]+200*5*8]
        final_counters.append({'train_seed': run['seed'], 'mode': run['mode'],
                               'timesteps': data['num_timesteps'], 'ppo_epoch_updates': data['_n_updates'],
                               'adam_steps': steps})
    integrity = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in config['hashes'].items()}
    assert all(integrity.values()), integrity
    (OUT/'resume_state_verification.json').write_text(json.dumps({'all_six_states_equal': True,
        'source_integrity': integrity, 'initial_state': audits[0], 'final_counters': final_counters}, indent=2)+'\n')
    text = ['# UZ05 短程 PPO Reset A/B 实测报告', '', f'最终决策：**{decision}**。{decision_explanation}', '',
            '3 个配对训练种子，各组 200 更新、819200 步，总计 4915200 步。'
            '这是成熟 v16 策略的短程续训，不是从零训练；无法据此判断从零学习的初期收敛。', '',
            '两组只改变闭链一致 Reset 开关；投影组还包含接触拒绝后的有限重采样。'
            '奖励、网络、学习率、动作和随机化沿用 v16。没有运行完整 4000 更新重训。', '',
            '## 续训状态与历史', '',
            '六组训练前 Actor/Critic、Adam、归一化、学习率/clip 调度及 PPO epoch 计数指纹完全一致。'
            '观察归一化关闭，保留相同奖励归一化统计。源 PPO epoch 更新计数 157500，Adam step 1260000。'
            '未恢复 MuJoCo 状态、环境 RNG、Python/NumPy/Torch RNG、rollout buffer；'
            'learn 的局部 timesteps/progress 从头计数，固定 LR=3e-5、clip=0.2 不受影响。'
            '课程固定为 v16，未改变采样分布或加入课程切换。', '',
            '最终六组均核验为 819200 个新增环境步、158500 个累计 PPO epoch 更新、Adam step 1268000。', '',
            '源训练历史最后一次：日志 episode reward 8963.19、episode length 2166.79；'
            '历史训练不等同于本次统一 Projected Reset 下的固定协议评估。'
            '源模型及源码哈希见 config_comparison.json；六组初始状态校验见 resume_state_verification.json。', '',
            '## 配对成功率', '',
            '成功沿用现有评估：24 秒到 time_limit；它只表示存活，速度、高度和姿态误差另报。'
            '每个数是五个评估种子的均值。所有策略都用 Projected Reset、无推力/动力学随机化、'
            '无移动起步，保留原观测噪声。命令在 v16 范围内，精确切换时序为独立固定协议。', '',
            '| 任务 | seed 1 Legacy→Projected | seed 2 | seed 3 | 配对差值中位数 [范围] |',
            '|---|---:|---:|---:|---:|']
    for task in config['evaluation']['primary_tasks']+config['evaluation']['additional_tasks']:
        a = select(task, 'success').sort_values('train_seed')
        cells = [f'{r.legacy:.0%}→{r.projected:.0%}' for r in a.itertuples()]
        text.append('| '+task+' | '+' | '.join(cells)+f' | {a.difference.median():+.0%} [{a.difference.min():+.0%}, {a.difference.max():+.0%}] |')
    text += ['', '## 配对连续指标', '',
             '下面先对每项主要任务的五个评估回合取均值，再对五项主要任务等权平均；失败回合也包含在内。'
             '失败越早可能产生更小的姿态/RMS 误差，因此必须结合存活时间解释。', '',
             '| 指标 | seed 1 Legacy→Projected | seed 2 | seed 3 | 配对差值中位数 [范围] |',
             '|---|---:|---:|---:|---:|']
    for metric in METRICS[1:]:
        a = select('primary_mean', metric).sort_values('train_seed')
        cells = [f'{r.legacy:.5g}→{r.projected:.5g}' for r in a.itertuples()]
        text.append('| '+metric+' | '+' | '.join(cells)+f' | {a.difference.median():+.5g} [{a.difference.min():+.5g}, {a.difference.max():+.5g}] |')
    text += ['', '## 初始化分布', '',
             '| seed/mode | 实际训练 reset 数 | 至少一次重采样 | 平均重采样次数 | 初始穿透 >0.01mm |',
             '|---|---:|---:|---:|---:|']
    for (seed, mode), a in resets.groupby(['train_seed', 'mode']):
        text.append(f'| {seed}/{mode} | {len(a)} | {(a.resamples>0).mean():.2%} | {a.resamples.mean():.4f} | {(a.penetration_m>1e-5).mean():.2%} |')
    text += ['', '所有初始 roll/pitch、四主动角度、轮心高度、接触与拒绝信息均保存在 reset_initial_samples.csv；'
             '分位数与分布图见 reset_distribution.csv/reset_distributions.png。'
             '这些是策略运行实际遇到的 episode 起点，样本数也受存活时间影响。'
             '两组轨迹和后续 RNG 消耗不相同；不能把任何差异完全归因于被动位置投影，'
             '投影一致性、速度一致性及接触拒绝共同构成本次处理。', '',
             '## 共同摆腿与控制记录', '',
             '所有最终评估回合的时间序列位于各 run 的 eval_200/*.npz。列名在文件 columns 中：'
             '四关节实际/目标/速度、pitch/pitch rate、轮速实际/目标、轮输出侧实际命令力矩，'
             '以及机体系轮心相对髋轴 xL/xR、共同偏移 (xL+xR)/2 和差值 xL-xR。'
             '目标变化率使用周期差，滞后误差也使用周期差；额外区分数字上 >π 的跳变及仅坐标包装跳变。', '',
             'evaluation_metrics.csv 包含每回合共同偏移 RMS/峰值/超 8cm 占比、目标滞后、电机饱和、'
             '共同偏移与 pitch/加速度在 ±0.496s 内的最大绝对相关及其 lag。'
             'lag 为正表示第二个信号滞后于共同偏移；这是描述性相关，不是因果证据。'
             '持续饱和时长要求某电机在连续完整控制周期内每个物理子步均达到 >=99% 限制，'
             '比只在一个子步触及限幅更严格。', '',
             '| 任务/seed/mode | 共同偏移峰值均值 m | 左右差 RMS m | 目标滞后 RMS rad | 腿持续饱和最长均值 s | 数值跳变次数总计 |',
             '|---|---:|---:|---:|---:|---:|']
    final = evaluation[(evaluation['update'] == 200) & (evaluation['mode'] != 'baseline')]
    control = final.groupby(['task','train_seed','mode']).mean(numeric_only=True).reset_index()
    control.to_csv(OUT/'control_diagnostics.csv', index=False)
    for task in ('reverse10', 'stop2_reverse'):
        for seed in (1,2,3):
            for mode in ('legacy','projected'):
                a = final[(final.task==task) & (final.train_seed==seed) & (final['mode']==mode)]
                text.append(f'| {task}/{seed}/{mode} | {a.common_x_peak_m.mean():.4f} | '
                    f'{a.differential_x_rms_m.mean():.4f} | {a.joint_target_lag_rms_rad.mean():.4f} | '
                    f'{a.longest_joint_full_cycle_saturation_s.mean():.3f} | {int(a.target_numeric_jump_count.sum())} |')
    text += ['', '## 对八个问题的回答', '']
    def diffs(task, metric):
        return select(task, metric).sort_values('train_seed').difference.to_numpy()
    def fmt(a): return '['+', '.join(f'{v:+.5g}' for v in a)+']'
    auc = convergence.pivot(index='train_seed', columns='mode', values='stand_success_auc_normalized')
    auc_diff = (auc.projected-auc.legacy).to_numpy()
    text += [
        f'1. 训练初期收敛：续训 0–200 更新的固定站立成功率归一化 AUC 差值为 {fmt(auc_diff)}（seed 1/2/3，Projected−Legacy）。'
        + ('方向全部改善，但这只是成熟 v16 的续训。' if np.all(auc_diff > 1e-9) else '未观察到三个种子一致提高。')
        + '从零训练初期收敛尚未测试；见 convergence_metrics.csv 与逐种子曲线。',
        f'2. 站立稳定性：最终成功率差值 {fmt(diffs("stand", "success"))}；'
        f'pitch RMS 差值 {fmt(diffs("stand", "pitch_rms_deg"))}°，roll RMS 差值 {fmt(diffs("stand", "roll_rms_deg"))}°。'
        '成功率与姿态误差应共同解释，微小 RMS 差异不代表新能力。',
        f'3. 共同摆腿：主要任务共同偏移 RMS 差值 {fmt(diffs("primary_mean", "common_x_rms_m"))} m；'
        f'峰值差值 {fmt(diffs("primary_mean", "common_x_peak_m"))} m。'
        + ('RMS 在三个种子均下降，仍需结合峰值、存活和轨迹。' if np.all(diffs('primary_mean', 'common_x_rms_m') < 0)
           else '没有三个种子一致的 RMS 改善。'),
        f'4. 高速换向：motion 成功率差值 {fmt(diffs("motion", "success"))}；'
        f'补充 reverse10 为 {fmt(diffs("reverse10", "success"))}。双方都失败的协议存在地板效应，无法证明等效。',
        f'5. 后退停车：stop2_reverse 成功率差值 {fmt(diffs("stop2_reverse", "success"))}；'
        f'高位 height2_reverse 为 {fmt(diffs("height2_reverse", "success"))}。',
        '6. 退化指标：主要任务平均高度 MAE 差值 '+fmt(diffs('primary_mean','height_mae_m'))+' m；'
        '目标变化率 RMS 差值 '+fmt(diffs('primary_mean','joint_target_rate_rms_rad_s'))+' rad/s；'
        '腿电机饱和比例差值 '+fmt(diffs('primary_mean','joint_saturation_fraction'))+'。'
        '这些成本正差值代表退步，完整任务/种子结果见 paired_results.csv，失败样本未剔除。',
        f'7. 种子一致性：主要任务平均成功率差值 {fmt(primary.sort_values("train_seed").difference.to_numpy())}；'
        '各种子波动范围已逐项列出。n=3 很小，五个固定评估种子不能替代更多训练重复。',
        f'8. 完整重训：{decision}。{decision_explanation}', '',
        '## 训练效率', '',
        '| seed | Legacy 秒 / steps/s | Projected 秒 / steps/s |', '|---|---:|---:|']
    for seed in (1,2,3):
        a = convergence[convergence.train_seed == seed].set_index('mode')
        text.append(f'| {seed} | {a.loc["legacy","wall_seconds"]:.1f} / {a.loc["legacy","wall_steps_per_second"]:.1f} | '
                    f'{a.loc["projected","wall_seconds"]:.1f} / {a.loc["projected","wall_steps_per_second"]:.1f} |')
    text += ['',
             '运行顺序交替以减少时间顺序影响，但系统负载仍可能波动。上述为带相同审计记录的实际端到端速度，'
             '不是孤立 Reset 微基准；评估在全部训练结束后执行。', '',
             '训练标量优先读取 TensorBoard：rollout reward/length 使用当轮编号，'
             'train/姿态统计按 SB3 下轮才 dump 的行为回移一轮对齐。最后一轮尚未刷新的字段从终端取舍入值；'
             '最后一轮 height RMSE 没有终端对应值，CSV 保留缺失，不补造。', '',
             '## 文件与复现', '',
             '训练命令见 commands.sh，训练顺序为 s1 Legacy/Projected、s2 Projected/Legacy、s3 Legacy/Projected。'
             '每组日志、TensorBoard、初始状态审计和 reset 分布分别保留；checkpoint 在 checkpoints/ppo_reset_ab_20261104_*。', '',
             '```bash', 'conda run --no-capture-output -n sim python scripts/run_ppo_reset_ab.py --resume --execute',
             'conda run --no-capture-output -n sim python scripts/analyze_ppo_reset_ab.py --evaluate', '```', '',
             '训练 resume 只跳过已完成组，拒绝覆盖部分运行；分析复用已有评估，避免重复运行。', '',
             '![训练曲线](training_curves.png)', '![评估对照](evaluation_comparison.png)',
             '![共同摆腿](joint_motion_comparison.png)']
    (OUT/'report.md').write_text('\n'.join(text)+'\n')
    (OUT/'decision.json').write_text(json.dumps({'decision': decision,
        'primary_success_differences': primary.difference.tolist(),
        'common_offset_rms_differences': common.difference.tolist()}, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluate', action='store_true')
    args = p.parse_args()
    config = json.loads((OUT/'config_comparison.json').read_text())
    assert all((Path(r['directory'])/'completed.json').exists() for r in config['runs']), 'Training incomplete'
    evaluation = []
    settings = config['evaluation']
    if args.evaluate:
        jobs = [(config['source_checkpoint'], settings['primary_tasks']+settings['additional_tasks'],
                 settings['seeds'], str(OUT/'baseline'), True)]
        labels = [(0, 'baseline', 0)]
        for run in config['runs']:
            for update in settings['checkpoints'][1:]:
                checkpoint = run['checkpoint']+f'_iter_{update}' if update < 200 else run['checkpoint']
                tasks = settings['intermediate_tasks'] if update < 200 else settings['primary_tasks']+settings['additional_tasks']
                jobs.append((checkpoint, tasks, settings['seeds'], str(Path(run['directory'])/f'eval_{update}'), update == 200))
                labels.append((run['seed'], run['mode'], update))
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
            for (seed, mode, update), rows in zip(labels, pool.map(evaluate_job, jobs)):
                evaluation += [dict(train_seed=seed, mode=mode, update=update, **r) for r in rows]
        pd.DataFrame(evaluation).to_csv(OUT/'evaluation_metrics.csv', index=False)
    evaluation = pd.read_csv(OUT/'evaluation_metrics.csv')
    training = get_training(config)
    stand = evaluation[(evaluation.task == 'stand') & (evaluation['mode'] != 'baseline')].groupby(
        ['train_seed', 'mode', 'update']).success.mean().reset_index(name='stand_success')
    training = training.merge(stand, on=['train_seed', 'mode', 'update'], how='left')
    training.to_csv(OUT/'training_metrics.csv', index=False)
    resets = initial_distributions(config)
    averaged, pairs = paired_results(evaluation, settings['primary_tasks'])
    convergence = convergence_metrics(config, training, evaluation)
    plots(training, evaluation, averaged, config, resets)
    historical_curves()
    report(config, training, evaluation, averaged, pairs, resets, convergence)
    from summarize_ppo_reset_findings import supplement
    supplement(OUT)


def evaluate_job(args):
    return evaluate(*args)


if __name__ == '__main__': main()
