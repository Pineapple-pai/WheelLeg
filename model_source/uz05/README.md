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

The active assembly contains 20T-to-20T chain stages, so those stages have a
geometric ratio of 1:1. Hardware information supplied after extraction gives a
wheel ratio of `268/17 = 15.7647059:1`; the joint-motor datasheet gives an
integrated ratio of `9:1`, with 20/40 Nm rated/peak output torque. These values
still require a motor-turn/output-turn check, and peak duration, efficiency,
backlash, direction, and encoder zero remain deployment inputs.

The M3508+C620 wheel datasheet gives 2.46/3.69 Nm rated/stall output torque and
571/587 rpm rated/no-load output speed. The C620 manual confirms a +/-20 A CAN
command range and 20 A maximum continuous controller current, so the lower
10 A motor rated current is used as the initial hardware limit. The conflicting
2.5 A claimed stall-current row is rejected. C620 cannot absorb regenerative
current; hardware requires a braking/absorption module that clamps the DC bus
to 35 V or below.

The URDF intentionally remains an open tree because URDF cannot represent this mechanism's closed loops directly. The USD repairs the confirmed right-crank joint type, angular-unit error, finite simulation drives, wheel material, and four excluded revolute loop constraints. The second closure on each side (`L45/R45` to `L234/R234`) was recovered from a common transverse hole axis independently fitted in all four link meshes. Hardware limits and actuator curves remain unknown.
