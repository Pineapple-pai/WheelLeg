"""Keyboard command mailbox; the viewer thread never writes simulation state."""
from queue import Empty, SimpleQueue
from dataclasses import dataclass

# GLFW key codes: top-row digits 48..57 and keypad digits 320..329.
DIGIT_COMMANDS = {
    7: ("mode", "stand"), 1: ("mode", "forward"),
    2: ("mode", "reverse"), 3: ("mode", "yaw"),
    4: ("mode", "yaw_right"), 5: ("height", -0.01),
    6: ("height", 0.01),
}
KEYBOARD_HELP = """点击 MuJoCo 场景使窗口获得焦点；数字小键盘和主键盘数字均可：
  1 前进        2 后退
  3 原地左转    4 原地右转
  5 降低1cm     6 升高1cm（保持当前平移/转向命令）
  7 零速度站立  R 重置回合
  - / +：前进/后退模式调速度大小，每次0.05m/s；转向模式调转速大小，每次0.1rad/s；
         站立/高度模式调高度，每次1cm。后退和右转时 + 表示加快，- 表示减慢。
  主键盘的 = 也作为 +；支持数字小键盘的加减键。
数值限制在模型训练范围内；模式切换保留高度和已调整的速度/转速，不重置机器人。
回合时长采用模型所属阶段的配置，结束后自动重置并保持模式；--once 在回合结束时退出。
"""


class KeyboardMailbox:
    def __init__(self):
        self.queue = SimpleQueue()

    def on_key(self, keycode: int):
        digit = keycode - 320 if 320 <= keycode <= 329 else keycode - 48
        command = DIGIT_COMMANDS.get(digit)
        if keycode in (82, 114):  # R / r
            command = ("reset", None)
        elif keycode in (45, 333):  # minus / GLFW_KEY_KP_SUBTRACT
            command = ("adjust", -1)
        elif keycode in (43, 61, 334, 336):  # plus, equal, keypad add/equal
            command = ("adjust", 1)
        if command is not None:
            self.queue.put(command)

    def drain(self):
        commands = []
        while True:
            try:
                commands.append(self.queue.get_nowait())
            except Empty:
                return commands


@dataclass
class KeyboardCommandState:
    mode: str = "stand"
    height: float = 0.27
    speed: float = 0.3
    turn_speed: float = 1.0

    def handle(self, kind, value, vx_range, yaw_range, height_range):
        def clamp(x, bounds):
            return min(max(float(x), bounds[0]), bounds[1])
        if kind == "mode":
            self.mode = value
        elif kind == "height":
            if self.mode == "random":
                self.mode = "height"
            self.height = clamp(self.height + value, height_range)
        elif kind == "adjust":
            if self.mode in ("forward", "reverse", "mixed", "mixed_left"):
                cap = -vx_range[0] if self.mode == "reverse" else vx_range[1]
                self.speed = clamp(self.speed + value * .05, (0, max(0, cap)))
            if self.mode in ("yaw", "yaw_right", "mixed", "mixed_left"):
                cap = -yaw_range[0] if self.mode in ("yaw_right", "mixed") else yaw_range[1]
                self.turn_speed = clamp(self.turn_speed + value * .1, (0, max(0, cap)))
            if self.mode not in ("forward", "reverse", "yaw", "yaw_right", "mixed", "mixed_left"):
                self.mode = "height" if self.mode == "random" else self.mode
                self.height = clamp(self.height + value * .01, height_range)

    def command(self):
        vx = (-self.speed if self.mode == "reverse" else self.speed
              if self.mode in ("forward", "mixed", "mixed_left") else 0.0)
        yaw = (-self.turn_speed if self.mode in ("yaw_right", "mixed") else self.turn_speed
               if self.mode in ("yaw", "mixed_left") else 0.0)
        return [vx, 0.0, yaw, self.height, 0.0]
