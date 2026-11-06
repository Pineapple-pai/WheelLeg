"""Per-speed acceptance for explicit, frozen calibration interventions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from check_transfer_progress import evaluate_progress
from review_completed_transfer import read_verified
from run_speed_calibration_diagnostic import OUT, ARMS, FIT, ROOT

def main():
    plan=json.loads((OUT/'plan.json').read_text())
    assert hashlib.sha256(FIT.read_bytes()).hexdigest()==plan['fit_source_sha256']
    outputs={}
    for mode,arms in [('nominal',ARMS),('holdout_dr',('control','both'))]:
        if not all((OUT/mode/arm/'results.json').exists() for arm in arms):continue
        data={arm:read_verified(OUT/mode/arm/'results.json') for arm in arms}
        baseline=data['control'];records=[];fits={};gates={}
        for arm,d in data.items():
            assert d['config']['protected']==baseline['config']['protected']
            assert d['config']['model_xml_sha256']==json.loads((ROOT/'diagnostics/low_speed_physics_pair_20261106/current/config.json').read_text())['model_xml_sha256']
            meta=d['config']['diagnostic_intervention']
            g,b=plan['coefficients']
            assert meta['gain']==(g if arm in ('gain','both') else 1.)
            assert meta['bias_m_s']==(b if arm in ('bias','both') else 0.)
            gates[arm]=evaluate_progress(baseline,d)
            for r in d['task_metrics']:
                rec=dict(arm=arm,**r)
                rec['goal']=d['config']['schedules'][r['task']][-1][1]
                rec['failed_response']=r['response_time_s'] is None
                rec['failed_rmse']=not r.get('steady_complete',False) or r.get('steady_rmse_m_s',np.inf)>r['speed_band_m_s']+1e-9
                rec['failed_occupancy']=not r.get('steady_complete',False) or r.get('steady_inband_fraction',0)<.95-1e-9
                records.append(rec)
            seg=[r for r in d['segments'] if r['end_s']==24 and r['vx_goal']!=0]
            if all(r['complete'] and r['steady_vx'] is not None for r in seg):
                gain,bias=np.polyfit([r['vx_goal'] for r in seg],[r['steady_vx'] for r in seg],1)
                fits[arm]=dict(gain=float(gain),bias=float(bias))
        df=pd.DataFrame(records);df.to_csv(OUT/mode/'per_episode.csv',index=False)
        summary=df.groupby(['arm','goal']).agg(n=('seed','size'),posture=('posture_success','sum'),
            combined=('composite_success','sum'),vx=('steady_mean_vx_m_s','mean'),
            rmse=('steady_rmse_m_s','mean'),std=('steady_std_vx_m_s','mean'),
            occupancy=('steady_inband_fraction','mean'),failed_response=('failed_response','sum'),
            failed_rmse=('failed_rmse','sum'),failed_occupancy=('failed_occupancy','sum'))
        summary.to_csv(OUT/mode/'per_speed.csv')
        outputs[mode]=dict(gates=gates,fits=fits,all_full_pass={arm:gates[arm]['fully_passed_all_tasks'] for arm in arms})
        print(mode,summary.to_string(),json.dumps(fits),sep='\n')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for arm in arms:
            z=summary.loc[arm];z=z[z.index!=0]
            axes[0].plot(z.index,z.vx,'o-',label=arm)
            axes[1].plot(z.index,100*z.rmse/np.abs(z.index),'o-',label=arm)
        axes[0].plot([-.5,.5],[-.5,.5],'k--');axes[1].axhline(5,color='k',ls='--')
        for ax in axes:ax.set_xlabel('Original command (m/s)');ax.grid(alpha=.2);ax.legend()
        axes[0].set_ylabel('Body vx (m/s)');axes[1].set_ylabel('Steady RMSE / |command| (%)')
        fig.tight_layout();fig.savefig(OUT/mode/'comparison.png',dpi=150);plt.close(fig)
    (OUT/'acceptance.json').write_text(json.dumps(outputs,indent=2)+'\n')
    if 'nominal' in outputs:
        nominal=outputs['nominal']
        table=pd.read_csv(OUT/'nominal/per_speed.csv')
        lines=['# 偏置消除与增益补偿：独立种子逐速度验收','',
            '本实验是显式输入干预诊断，不是PPO训练或可直接续训的候选权重。策略网络、物理、执行器、奖励和评分目标均未改动；仅在评估时改变Actor看到的非零vx命令。零命令完全不变。','',
            '系数仅由此前66300–66302数据确定：gain=1.122050727，bias=+0.040197784 m/s。此次66400–66402为未用于拟合的种子。四组各21回合，共84回合；24秒，3秒起步，±0.1/±0.25/±0.5，高度0.27m，标称物理，观测噪声开启。','',
            '验收沿用5%速度带、稳态2秒RMSE、至少95%带内占比、5秒内形成合格的1秒响应窗口，以及完整存活和姿态约束。未放宽门槛，不按平均误差选择幸运种子。','',
            '|组别|非零段拟合gain|bias m/s|完整任务全部通过|',
            '|---|---:|---:|---|']
        for arm in ARMS:
            f=nominal['fits'].get(arm,{})
            lines.append(f"|{arm}|{f.get('gain',float('nan')):.6f}|{f.get('bias',float('nan')):+.6f}|{nominal['all_full_pass'][arm]}|")
        lines+=['','## 各速度联合通过回合数','',
                '|命令 m/s|control|bias|gain|both|','|---|---:|---:|---:|---:|']
        for goal in sorted(table.goal.unique()):
            cells=[f'{goal:+g}']
            for arm in ARMS:
                r=table[(table.arm==arm)&(table.goal==goal)].iloc[0]
                cells.append(f'{int(r.combined)}/{int(r.n)}')
            lines.append('|'+ '|'.join(cells)+'|')
        lines+=['','## 同时补偿组的精度与失败类型','',
                '|命令|实际均值|RMSE|标准差|带内占比|响应失败|RMSE失败|占比失败|',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
        for _,r in table[table.arm=='both'].iterrows():
            lines.append(f'|{r.goal:+g}|{r.vx:+.6f}|{r.rmse:.6f}|{r["std"]:.6f}|{r.occupancy:.2%}|{int(r.failed_response)}/3|{int(r.failed_rmse)}/3|{int(r.failed_occupancy)}/3|')
        lines+=['','![逐速度比较](nominal/comparison.png)','',
            '失败类型可重叠。均值接近目标不等于速度波动、带内占比或响应窗口达标；非零段拟合截距不代表零速站立漂移。', '',
            '## 是否提供长训命令','']
        if not nominal['all_full_pass']['both']:
            lines+=['未通过完整逐速度验收，因此本轮不提供长训练命令，不修改train_joint_transfer.sh，不将补偿包装成已学会的策略。按照预先计划，不进入条件性的域随机化验收。', '',
                '即使某项进展门槛通过，也只说明诊断干预改善，并不表示存在通过验收、可续训的checkpoint。下一步需处理残余动态波动和响应误差，再将有效机制落实到可训练策略并重新验收。']
        else:
            lines+=['标称诊断通过。仍需独立域随机化验证，并落实到可续训的策略权重；输入干预本身不能用于宣称PPO已学会，也不足以签发长训命令。']
        lines+=['','## 验证与复现','',
            'analysis脚本核对四组checkpoint哈希相同、物理哈希与当前模型相同、协议与种子匹配、系数与冻结plan一致，并重新计算进展和全任务门槛。输入变换审计确认恒等变换逐值一致、零命令不变、其它观测维度不变。三个既有进展门槛测试通过。','',
            '`conda run --no-capture-output -n wheelleg-train python scripts/analyze_speed_calibration_diagnostic.py`', '',
            '原始轨迹和评分位于nominal/{control,bias,gain,both}/；per_episode.csv包含逐种子失败标记，per_speed.csv包含逐速度汇总，acceptance.json保留全部门槛结果。原始评估使用run_speed_calibration_diagnostic.py，已有目录拒绝覆盖。']
        (OUT/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
