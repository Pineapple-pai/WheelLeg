# UZ05 训练模型

当前默认是`uz05_20261105_armature001/uz05_rl.xml`：在用户提供的新版模型上，仅把左右轮轴附加惯量从0.01改为**0.001kg·m²**，作为待标定候选。现有减速比、电机力矩映射、阻尼、云台COM/惯量及其它物理设置保持不变。依据与验证见[本轮说明](../docs/wheel_armature001_20261105.md)。

网格复用`uz05_20261104/meshes/stl/`，默认训练与显示均不依赖外部目录。

原始新版`uz05_20261104/uz05_rl.xml`仍保留0.01，和来源`/home/p/文档/UZ05_model/uz05_rl.xml`逐字节一致；原始资产哈希见`diagnostics/new_model_validation/model_audit.json`。根目录`uz05_修正后(3).xml`保留旧0.0001模型。

环境加载时将源平面替换为环境地面，并裁去纯视觉网格；源XML不被环境改写。保持闭链Reset投影，XML的stand keyframe不自动替换训练Reset。训练控制器仍用自身电流/力矩限幅，不将XML堵转上限作为额定输出。

历史checkpoint按模型XML哈希绑定，改变armature后不能直接同模型续训；不要改写旧契约。复现此前的0.01权重需在启动Python前设置：

```bash
export UZ05_MODEL_XML="$PWD/models/uz05_20261104/uz05_rl.xml"
```

新开终端默认使用0.001；如已经导出上述变量，执行`unset UZ05_MODEL_XML`恢复当前默认。旧训练报告中的结果对应当时模型，不自动成为0.001版本的成绩。
