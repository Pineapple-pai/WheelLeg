"""Render the complete pure-PPO experiment record, including rejected arms."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/pure_ppo_precision_20261106'

def main():
 df=pd.read_csv(OUT/'per_speed.csv');fits=json.loads((OUT/'fits.json').read_text());gates=json.loads((OUT/'gates.json').read_text())
 lines=['# 纯PPO速度精度学习：奖励、探索与长训就绪验证','',
 '本轮所有训练及验收均使用原始PPO策略输出，未使用教师、监督动作目标、命令映射、动作补偿、额外增益、残差控制器或动作滤波。之前的输入补偿仅存在于单独诊断，未进入本轮训练或验收。','',
 '## 实验设置','',
 '初始化为当前0.001物理上的成熟v16迁移权重。每条初训线200轮×8环境×512步=819200条转移；batch512，5epochs，固定学习率1e-5，entropy=0，重置奖励RMS，保留相同Actor均值/Critic/Adam。观测噪声与物理随机化开启，六维动作均自由学习。','',
 '四组：control保持原precision速度尺度0.05和轮探索std0.02；wheel_fine仅改轮初始std0.003；reward_fine仅改实际速度误差奖励尺度0.01；both_fine同时改两项。腿初始std均0.02，所有std之后由PPO学习。奖励尺度只存在于训练目标，不是部署时的补偿增益。','',
 '## 逐训练种子的冻结结果','',
 '|训练种子/方案|车速增益|偏置m/s|归一化误差改善|进展门槛|全部任务通过|','|---|---:|---:|---:|---|---|']
 for key,g in gates.items():
  f=fits.get(key,{'gain':float('nan'),'bias':float('nan')})
  lines.append(f"|{key}|{f['gain']:.6f}|{f['bias']:+.6f}|{100*g['relative_improvement']:+.2f}%|{g['accepted_for_further_training']}|{g['fully_passed_all_tasks']}|")
 lines+=['','拟合只使用非零末2秒窗口，截距不能解释为零速漂移。评估使用66600–66602三个相同种子，24秒、3秒阶跃、±0.1/±0.25/±0.5及站立；评分仍为原5%带、95%占比、响应与姿态联合约束。','',
 '## 初始reward_fine候选逐速度结果','',
 '|命令|实际速度|RMSE|速度标准差|带内占比|综合通过|','|---|---:|---:|---:|---:|---:|']
 for _,r in df[(df.train_seed==711)&(df.arm=='reward_fine')].iterrows():
  lines.append(f'|{r.goal:+g}|{r.vx:+.6f}|{r.rmse:.6f}|{r["std"]:.6f}|{r.occupancy:.2%}|{int(r.passed)}/{int(r.n)}|')
 lines+=['','## 随机化与继续训练','']
 for path in sorted(OUT.glob('readiness_*.json')):
  d=json.loads(path.read_text())
  if 'recipe' not in d:continue
  lines.append(f"- {d['recipe']}：允许有监测的长训={d['ready_for_bounded_long_training']}；完整任务全部通过={d['fully_passed_all_tasks']}。")
  for name,g in d['evidence'].items():
   lines.append(f"  - {name}：误差改善{g['relative_improvement']:.2%}，进展通过={g['accepted_for_further_training']}，已成功任务退化={g['regressions']}。")
 lines+=['','wheel_fine在标称训练种子711/712均有进展，但预选711在独立随机化66701的-0.25任务丢失原有成功，因此保留拒绝记录。随后预先选择更有效的reward_fine，使用新训练种子713/714复现，未用新种子结果反向挑选来源。','',
 'continuation_s711保存了reward_fine原711候选追加200轮的独立试验：保留学到的std/Adam/RMS，使用新的环境种子1711。这是额外的纯PPO续训，不是监督校准。其结果需单独比较200轮来源，不得仅与未训练来源比较而掩盖退化。','',
 '## 长训与部署的区别','',
 '允许长训只意味着策略表现出可重复、可监测的学习进展，不意味着所有任务已收敛。continue_pure_ppo.py要求经过核验的readiness文件，并在每段结束后同种子配对验收。候选退化或无足够进展时停止，保留前一个可接受来源；不自动修改用户现有源权重。达到标称全任务通过后停止，等待独立随机化验证。','',
 '## 代码与验证','',
 '- train_uz05.py新增显式--wheel-action-std，覆盖阶段初始轮探索值，之后仍由PPO学习；与--preserve-action-std互斥。',
 '- 新增joint_tracking_startup_precision_fine阶段，相对原precision只收紧真实速度误差奖励尺度0.05→0.01；动作、物理、噪声和命令范围不变。',
 '- 真实CLI训练检查证实轮std覆盖生效且四腿std不变；续训参数测试、8项startup测试及3项进展门槛测试通过。',
 '- 分析脚本检查实际配置、初始网络均值/Critic/Adam/RMS一致、checkpoint哈希、PPO评估无diagnostic_intervention；所有失败候选保留。','',
 '详细结果见per_episode.csv、per_speed.csv、fits.json、gates.json和readiness文件。脚本：analyze_pure_ppo_precision.py、check_pure_ppo_readiness.py、report_pure_ppo_precision.py。']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
