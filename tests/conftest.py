"""Test setup shared by all test files."""

import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Offscreen rendering without a window, like the CLI (cli._headless_gl). Without it MuJoCo picks GLFW, which aborts
# the interpreter in a shell that has no usable display.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

from xrcollect.config import ROOT, Config  # noqa: E402
from xrcollect.env import Env, stream_seed  # noqa: E402
from xrcollect.expert import Expert  # noqa: E402
from xrcollect.protocol import SRC_CONTROLLER, HandInput, InputFrame  # noqa: E402
from xrcollect.task.loader import load_bundle  # noqa: E402

TASKS = ["tasks/stack_two_blocks", "tasks/duplo_tower", "tasks/open_drawer", "tasks/stack_cups"]


@pytest.fixture
def cfg(tmp_path) -> Config:
    c = Config()
    c.record.out_dir = str(tmp_path / "data" / "episodes")
    c.record.countdown_s = 0.0
    return c


@pytest.fixture
def task_copy(tmp_path) -> Path:
    """A private copy of the block-stacking task that a test may edit."""
    dst = tmp_path / "task"
    shutil.copytree(ROOT / "tasks" / "stack_two_blocks", dst)
    return dst


_ENVS: dict = {}


def get_env(task: str, embodiment: str | None = None) -> Env:
    """Scenes are slow to build (~1 s): one per task and embodiment for the whole test run."""
    key = (task, embodiment)
    if key not in _ENVS:
        _ENVS[key] = Env(load_bundle(task, {"embodiment": embodiment} if embodiment else None))
    return _ENVS[key]


_EPISODES: dict = {}


@pytest.fixture(scope="session")
def expert_episode(tmp_path_factory):
    """(path, success) of a scripted episode of a task, recorded once per test run."""
    def make(task: str, embodiment: str | None = None, seed: int = 0):
        key = (task, embodiment, seed)
        if key not in _EPISODES:
            env = get_env(task, embodiment)
            ex = Expert(env)
            ex.reset(stream_seed("validate", seed))
            ok = ex.run()
            d = tmp_path_factory.mktemp("episodes")
            _EPISODES[key] = (ex.save(d / "episode_000000.hdf5", success=ok), ok)
        return _EPISODES[key]
    return make


def controllers(squeeze_left=0.0, squeeze_right=0.0, left_pos=(-0.2, 1.1, -0.35), right_pos=(0.2, 1.1, -0.35),
                **buttons) -> InputFrame:
    """Both Touch controllers held in front of the operator. buttons: e.g. right_primary=1.0, left_trigger=1.0."""
    from xrcollect import protocol as P

    f = InputFrame()
    for side, pos in (("left", left_pos), ("right", right_pos)):
        h = HandInput(source=SRC_CONTROLLER)
        h.pose = np.array([*pos, 0, 0, 0, 1.0])
        setattr(f, side, h)
    f.left.buttons[P.B_SQUEEZE] = squeeze_left
    f.right.buttons[P.B_SQUEEZE] = squeeze_right
    for key, v in buttons.items():
        side, name = key.split("_", 1)
        getattr(f, side).buttons[getattr(P, f"B_{name.upper()}")] = v
    return f


def ticks(rt, n, frame=None):
    for _ in range(n):
        if frame is not None:
            rt.submit_input(frame)
        rt._tick(time.monotonic())
