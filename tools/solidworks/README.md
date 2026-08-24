# UZ-05 腿装配提取

目标是从 `串联腿装配方案2.SLDASM` 提取可审计的数据，不直接使用 SolidWorks URDF Exporter 生成的开链模型。

## Windows 端操作

前提：安装能够打开源文件版本的 SolidWorks，所有引用零件位于同一 `Pack and Go` 包中。

1. 打开 `串联腿装配方案2.SLDASM`。
2. 确认 `What's Wrong` 中没有 dangling mate、missing component 或 rebuild error。
3. 确认用于实物的 configuration，并将闭链子装配设为 flexible，手动检查完整行程和装配支路。
4. 新建一个 SolidWorks VBA macro，在 VBA 编辑器中导入 `export_uz05_leg.bas`，运行 `Main`。
5. 宏会在装配体旁生成 `uz05_leg_export`：
   - `document.csv`：源装配和 configuration。
   - `components.csv`：组件路径、configuration、suppression state、世界变换和局部质量属性数组。
   - `mates.csv`：mate 类型、对齐方式、引用组件和实体参数。
6. 以同一 configuration 导出 `leg_assembly.step`，格式选择 STEP AP242，单位使用米或在交付说明中明确单位。
7. 使用 `Evaluate -> Mass Properties` 导出质量报告，并分别记录左右髋、膝输入轴和轮轴的坐标与正方向。

宏只读取当前打开的装配体。由于当前 Linux 机器没有 SolidWorks，宏源码尚未在 SolidWorks VBA 编译器中执行；首次运行若遇到 API 版本差异，应保留完整错误信息和 SolidWorks 主版本。

## 放回项目

建议目录：

```text
model_source/uz05_leg/
  components.csv
  document.csv
  mates.csv
  leg_assembly.step
  mass_properties.txt
```

复制部署模板并填写实测值：

```bash
cd /home/p/下载/WheelLeg
cp cfg/model/uz05_deployment.template.yaml cfg/model/uz05_deployment.yaml
python3 scripts/validate_uz05_deployment_data.py
```

校验器通过前，不生成闭链 USD，不开放腿部动作，也不训练跳跃。
