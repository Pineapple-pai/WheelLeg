# 速度精度1000轮复盘与候选验收工具

[完整报告](../diagnostics/precision_completed_review/report.md)、[保存点统计](../diagnostics/precision_completed_review/checkpoint_summary.csv)、[起步干预图](../diagnostics/precision_completed_review/launch_interventions.png)。

`joint_precision_20261105_184905/checkpoint`在新版MJCF上训练1000轮：13个独立标称种子中+0.25m/s综合12/13通过，零速姿态12/13；±0.1仍有速度偏差，-0.25与±0.5起步未解决，域随机化精度也未通过。保留该权重，不用未经验证的短训结果覆盖。

命令缓升能显著改善部分起步的存活，轮目标速率限制没有解决±0.5，冻结腿目标让反向更差。新增可选`startup_mixed_ramp`供复现；三种子短训发现正反向能力交换，未采用为默认。原采样及全部6维PPO动作保持可用。

本轮没有选出可推荐的新训练配方，不直接追加1000或6000轮。新增实验入口继承本次最终Actor/Critic/Adam、动作探索尺度及奖励归一化；lr1e-5，原阶跃采样，4环境×256步、batch256。默认仅25轮=25600个新转移，源与候选分别对同一组65600–65604种子自动跑35回合原阶跃验收。**25轮实测也未形成整体升级，下面命令仅供复现和后续参数研究，不是本轮推荐的性能修复。**

```bash
cd /home/p/下载/WheelLeg
SOURCE=checkpoints/joint_precision_20261105_184905/checkpoint \
VERSION="joint_refine_$(date +%Y%m%d_%H%M%S)" \
SEED=0 bash scripts/train_joint_refine.sh
```

训练与评估日志分别在`logs/<VERSION>.log`和`logs/<VERSION>_evaluation.log`，新权重在`checkpoints/<VERSION>/checkpoint`，评估在`diagnostics/<VERSION>/evaluation/results.json`，源评估在`diagnostics/<VERSION>_source_evaluation/`。旧源文件不会覆盖；不要把末轮自动视为最佳权重。逐任务逐种子保护源模型已有姿态/综合成功，且要求新增非零综合成功；不满足时写入`refinement_gate.json`并退出3，提示保留源权重。这是验收拒绝，训练文件仍完整保留。通过也只允许进入后续独立验证，不自动替换源模型。

续训100轮的三种子补测结果见完整报告与`diagnostics/precision_conservative/`。25轮是为尽早暴露退化所设的预算上限，未被证明是最佳训练长度。若+0.25速度精度退化，不能用-0.25个别成功、奖励上涨或回合变长代替验收，不应继续叠加到该候选。需要验证停止、反转或更大速度时，应单独扩充验收任务；目前结论只适用于这里列出的站立和起步协议。
