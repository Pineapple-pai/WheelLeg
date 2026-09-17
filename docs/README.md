# WheelLeg 文档

当前项目只有一条主线：**MuJoCo 版 UZ-05 轮腿训练**。

- 入口与用法见根目录 [README.md](../README.md)
- 重构说明、接口、坐标系、执行器、与开源对齐、资源上限：
  [UZ-05 重构说明与训练配置](uz05_refactor.md) ★
- 机械/驱动事实与实物部署阻塞项：[模型与实物部署审计](model_deployment_audit.md)
  （写于 Isaac 时期，里面的 `omni_drones/...usd` 路径已失效，但质量/减速比/力矩等硬件结论仍有效）
- 待办与设计参考：[修改建议书](../修改.md)

> 说明：仓库里的 `omni_drones/`、`cfg/`、`tools/`、`setup.py` 与
> `scripts/{train,play,sweep_*,diagnose_*}.py` 已在 2026-09 全部移除，
> 它们是 Isaac Sim 时期的旧轮腿实现，当前 MuJoCo 训练链不引用任何一个。
> 需要时用 `git checkout HEAD -- <路径>` 找回。

## 最小训练命令

```bash
conda run --no-capture-output -n sim python -u scripts/train_uz05.py \
  --stage stand --stand-level 0 --assist-adaptive --version stand_s0_v1 \
  --updates 800 --rollout-steps 256 --batch-size 512 \
  --num-envs 8 --vec-env subproc --save-interval 100
```

一键跑完整课程：`NUM_ENVS=8 TAG=v1 bash scripts/train_uz05_all.sh stand`
