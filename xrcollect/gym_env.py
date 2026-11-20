"""Any task as a Gymnasium environment, for reinforcement learning on the same scene the operator records in.

    import gymnasium as gym
    import xrcollect.gym_env  # registers xrcollect/<task>-v0 for every tasks/<task>/task.yaml

    env = gym.make("xrcollect/stack_cups-v0", cameras=["left", "cam_wrist"])
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

The world is the platform's `Env` (same bundle, scene, reset randomization and task stages as a recording), so a
policy trained here runs unchanged through `xrc serve --policy`.

- Action: per arm [J joint targets (rad), gripper (0 closed, 1 open)], the layout of a recording's
  `action/joint_pos`, so recorded demonstrations are (observation, action) pairs of this env as they are. With
  `safety=True` (default) the targets go through the platform's safety filter first, like every recorded action.
- Observation: `Env.observe()` (joint_pos, joint_vel, gripper_width, ee_pose, ee_wrench, objects, fixture_joints)
  plus `images.<camera>` (uint8 RGB) for the requested cameras, rendered like an export (geom groups 0-2).
- Reward: "sparse" = 1 on success; "progress" = the share of task stages newly done this step, plus 1 on success.
  An episode terminates on success and is truncated after `max_episode_seconds`.
- `reset(options={"state": s})` starts from a saved simulator state (a recording's `sim/state[t]`): rewind to any
  moment of a take and continue from there.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from xrcollect.config import ROOT, Config, rooted
from xrcollect.env import Env
from xrcollect.safety import SafetyFilter
from xrcollect.task.loader import load_bundle

REWARDS = ("sparse", "progress")


class XRCollectEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"]}

    def __init__(self, task: str = "tasks/stack_cups", cameras: list[str] | tuple[str, ...] = (),
                 image_size: tuple[int, int] = (224, 224), max_episode_seconds: float = 30.0,
                 reward: str = "sparse", safety: bool = True, config: Config | None = None,
                 render_mode: str | None = None, render_camera: str | None = None):
        """task: task folder or YAML file (relative to the repository root); cameras: images added to the
        observation (none by default: fastest for state-based RL); image_size: (height, width)."""
        if reward not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}, got {reward!r}")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.cfg = config or Config()
        self.env = Env(load_bundle(rooted(task)))
        e, m = self.env, self.env.model
        missing = [c for c in [*cameras, *([render_camera] if render_camera else [])] if c not in e.camera_names]
        if missing:
            raise ValueError(f"unknown cameras {missing}; this scene has {e.camera_names}")
        self.cameras, self.image_size = list(cameras), tuple(image_size)
        self.reward_mode, self.render_mode = reward, render_mode
        self.render_camera = render_camera or e.camera_names[0]
        self.max_steps = max(1, round(max_episode_seconds / e.control_dt))
        self.safety = SafetyFilter(e.bundle, e.adapter, self.cfg.ik, self.cfg.safety, e.control_dt) if safety else None
        self._renderer = None
        self._render_renderer = None

        # Action: per arm [J joint targets, gripper], bounded by the joint ranges.
        n = e.b.n_arms
        self.n_joints = len(e.b.qadr) // n
        jnt_of_qadr = {int(m.jnt_qposadr[j]): j for j in range(m.njnt)}
        rng = np.array([m.jnt_range[jnt_of_qadr[int(a)]] for a in e.b.qadr]).reshape(n, self.n_joints, 2)
        low = np.concatenate([rng[:, :, 0], np.zeros((n, 1))], axis=1).ravel()
        high = np.concatenate([rng[:, :, 1], np.ones((n, 1))], axis=1).ravel()
        self.action_space = spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)

        e.reset(0)
        state = {k: np.asarray(v, np.float64) for k, v in e.observe().items()}
        obs_spaces: dict[str, spaces.Space] = {
            k: spaces.Box(-np.inf, np.inf, shape=v.shape, dtype=np.float64) for k, v in state.items()}
        for c in self.cameras:
            obs_spaces[f"images.{c}"] = spaces.Box(0, 255, shape=(*self.image_size, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(obs_spaces)
        self._t = 0
        self._progress = 0

    # -- gymnasium API -------------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        e = self.env
        e.reset(int(self.np_random.integers(2**31)))   # the layout follows gymnasium's seeding
        if options and options.get("state") is not None:
            e.set_state(np.asarray(options["state"], np.float64))
        if self.safety is not None:
            self.safety.reset(e.arm_q)
        self._t, self._progress = 0, e.progress()
        return self._obs(), self._info()

    def step(self, action):
        e = self.env
        a = np.clip(np.asarray(action, np.float64), self.action_space.low, self.action_space.high)
        a = a.reshape(e.b.n_arms, self.n_joints + 1)
        q, grippers = a[:, :-1].ravel(), a[:, -1]
        if self.safety is not None:
            q, _ = self.safety.filter(q, e.arm_q, e.data.qpos[e.b.finger_qadr])
        e.apply_action(q, grippers)
        e.step()
        self._t += 1
        success, progress = e.success(), e.progress()
        reward = float(success)
        if self.reward_mode == "progress":
            n_stages = max(1, len(e.task.stage_names))
            reward += max(progress - self._progress, 0) / n_stages
        self._progress = progress
        return self._obs(), reward, success, self._t >= self.max_steps, self._info(success)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._render_renderer is None:
            from xrcollect.scene.render import CameraRenderer
            self._render_renderer = CameraRenderer(self.env.model, *self.image_size, [self.render_camera])
        return self._render_renderer.render(self.env.data)[self.render_camera]

    def close(self):
        for r in (self._renderer, self._render_renderer):
            if r is not None:
                r.close()
        self._renderer = self._render_renderer = None

    # -- helpers -------------------------------------------------------------------------------------
    def _obs(self) -> dict[str, np.ndarray]:
        obs = {k: np.asarray(v, np.float64) for k, v in self.env.observe().items()}
        if self.cameras:
            if self._renderer is None:
                from xrcollect.scene.render import CameraRenderer
                self._renderer = CameraRenderer(self.env.model, *self.image_size, self.cameras)
            for c, img in self._renderer.render(self.env.data).items():
                obs[f"images.{c}"] = img
        return obs

    def _info(self, success: bool | None = None) -> dict[str, Any]:
        e = self.env
        return {"success": e.success() if success is None else success, "progress": e.progress(),
                "stage_flags": e.stage_flags(), "held": e.held(), "seed": e.seed}

    def get_state(self) -> np.ndarray:
        """The full simulator state (what a recording stores per step), for `reset(options={"state": ...})`."""
        return self.env.get_state()


def register_tasks() -> list[str]:
    """Register `xrcollect/<task>-v0` for every tasks/<task>/task.yaml; returns the ids."""
    ids = []
    for f in sorted((ROOT / "tasks").glob("*/task.yaml")):
        env_id = f"xrcollect/{f.parent.name}-v0"
        if env_id not in gym.registry:
            gym.register(id=env_id, entry_point="xrcollect.gym_env:XRCollectEnv",
                         kwargs={"task": str(f.parent.relative_to(ROOT))})
        ids.append(env_id)
    return ids


register_tasks()
