"""MuJoCo 训练入口（兼容 profile 版本化命令）。

实际训练实现在 :mod:`train_uz05`，本文件保留用户习惯的
``train_uz05_mujoco.py --profile vXX`` 命令形式。
"""

from train_uz05 import main


if __name__ == "__main__":
    main()
