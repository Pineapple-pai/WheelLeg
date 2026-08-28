# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from .twowheel import TwoWheelBalance, TwoWheelClosedLoop
from .isaac_env import IsaacEnv


def resolve_env_class(name: str):
    """Resolve local tasks even if Isaac Sim recreated the registry module."""
    local_tasks = {
        TwoWheelBalance.__name__: TwoWheelBalance,
        TwoWheelBalance.__name__.lower(): TwoWheelBalance,
        TwoWheelClosedLoop.__name__: TwoWheelClosedLoop,
        TwoWheelClosedLoop.__name__.lower(): TwoWheelClosedLoop,
    }
    env_class = IsaacEnv.REGISTRY.get(name) or local_tasks.get(name)
    if env_class is None:
        available = sorted(set(IsaacEnv.REGISTRY) | set(local_tasks))
        raise KeyError(f"Unknown task {name!r}. Available tasks: {available}")
    return env_class
