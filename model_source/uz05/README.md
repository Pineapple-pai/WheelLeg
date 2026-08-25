# UZ-05 SolidWorks extraction

Source: `D:\WheelLeg`, SolidWorks 2024 SP0.1, configuration `默认`, extracted read-only on 2026-08-25.

| Directory | Assembly | Components | Mate rows | Mass |
| --- | --- | ---: | ---: | ---: |
| `leg/` | `串联腿装配方案2.SLDASM` | 153 | 348 | 0.994479 kg |
| `chassis/` | `底盘总装.SLDASM` | 483 | 648 | 15.682047 kg |
| `full_assembly/` | `UZ-05-open总装.SLDASM` | 635 | 30 | 18.687170 kg |

The original `chassis/mates.csv` contains two failing concentric mates, `同心259` and `同心260`, between the left/right calf and gearbox output components. Their old calf holes were 23.0 mm radius while the current unique coaxial holes are 23.1 mm radius, which invalidated the face references. Re-adding either mate returns `swAddMateError_OverDefinedAssembly`; they are redundant with the remaining closed-loop constraints.

`solidworks_repaired/chassis.cad_repaired.SLDASM` removes these two redundant broken mates. It was closed, reopened, and extracted again with 644 mate-entity rows, zero active or suppressed mate errors, and zero component-transform drift. The original `D:\WheelLeg\底盘总装.SLDASM` was not modified. Suppressed legacy components still contain unresolved `E:\2025赛季\...` references and cause `swFileNotFoundError` during open; they do not affect the active simulation geometry but prevent a completely clean Pack and Go.

After rebuild, the repaired chassis evaluates to `15.682344 kg`. The repaired USD preserves moving-link masses and assigns the remaining complete-assembly equivalent mass/inertia to `base_link`; its aggregate mass and zero-pose COM match the full SolidWorks assembly (`18.687170 kg`, `[-0.01147962, -0.00286462, 0.11737280] m`) within `1e-6`.

Generated static repair assets:

- `omni_drones/robots/assets/twowheel_uz05/urdf/twowheel_uz05.cad_repaired.urdf`
- `omni_drones/robots/assets/twowheel_uz05/usd/twowheel_uz05.cad_repaired.usda`

The URDF intentionally remains an open tree because URDF cannot represent this mechanism's closed loops directly. The USD repairs the confirmed right-crank joint type, angular-unit error, finite simulation drives, wheel material, and two excluded revolute loop constraints. The loop anchors were recovered independently from matching mesh holes; their authored anchor residuals are below `3e-9 m` with parallel axes. Hardware limits and actuator curves remain unknown.
