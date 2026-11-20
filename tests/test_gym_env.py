"""xrcollect.gym_env: every task as a Gymnasium environment, on the same scene a recording uses."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from conftest import TASKS
from gymnasium.utils.env_checker import check_env

import xrcollect.gym_env  # registers xrcollect/<task>-v0
from xrcollect.recorder import read_episode


def make(task: str = "tasks/stack_cups", **kw) -> xrcollect.gym_env.XRCollectEnv:
    return gym.make(f"xrcollect/{Path(task).name}-v0", **kw).unwrapped


@pytest.mark.parametrize("task", TASKS)
def test_every_task_is_a_valid_gym_env(task):
    env = make(task)
    check_env(env, skip_render_check=True)
    n = env.env.b.n_arms
    assert env.action_space.shape == (n * (env.n_joints + 1),)
    assert set(env.observation_space.spaces) >= {"joint_pos", "joint_vel", "gripper_width", "ee_pose", "objects"}


def test_seeded_reset_is_reproducible_and_seeds_differ():
    env = make()
    a, _ = env.reset(seed=3)
    b, _ = env.reset(seed=3)
    c, _ = env.reset(seed=4)
    assert np.array_equal(a["objects"], b["objects"]) and np.array_equal(a["joint_pos"], b["joint_pos"])
    assert not np.allclose(a["objects"], c["objects"])


def test_reset_to_a_saved_state_rewinds():
    env = make()
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(5):
        env.step(env.action_space.sample())
    state, before = env.get_state(), env.env.observe()
    for _ in range(20):
        env.step(rng.uniform(env.action_space.low, env.action_space.high))
    obs, _ = env.reset(options={"state": state})
    assert np.array_equal(obs["joint_pos"], before["joint_pos"])
    assert np.array_equal(obs["objects"], before["objects"])


def test_recorded_demo_is_a_trajectory_of_the_gym_env(expert_episode):
    """A recording's action/joint_pos, stepped through the gym env from the recording's first state, solves the
    task: demonstrations are (observation, action) pairs of this env as they are."""
    path, ok = expert_episode("tasks/stack_cups")
    assert ok
    _, data = read_episode(path)
    env = make(safety=False)  # the recorded actions already went through the safety filter
    obs, info = env.reset(options={"state": data["sim/state"][0]})
    assert np.allclose(obs["joint_pos"], data["obs/joint_pos"][0])
    terminated = False
    for t, action in enumerate(data["action/joint_pos"]):
        obs, reward, terminated, _, info = env.step(action)
        if terminated:
            break
    assert terminated and info["success"] and reward == 1.0
    assert t < len(data["action/joint_pos"])


def test_progress_reward_and_time_limit():
    env = make(reward="progress", max_episode_seconds=0.1)
    env.reset(seed=0)
    hold = np.concatenate([env.env.arm_q, [1.0]]).astype(np.float32)
    rewards, truncated = [], False
    while not truncated:
        _, r, terminated, truncated, _ = env.step(hold)
        rewards.append(r)
    assert len(rewards) == env.max_steps == 5 and not terminated and sum(rewards) == 0.0


def test_cameras_in_the_observation_and_render():
    env = make(cameras=["left", "cam_wrist"], image_size=(48, 64), render_mode="rgb_array", render_camera="right")
    obs, _ = env.reset(seed=0)
    for c in ("left", "cam_wrist"):
        img = obs[f"images.{c}"]
        assert img.shape == (48, 64, 3) and img.dtype == np.uint8 and img.std() > 0
    assert env.render().shape == (48, 64, 3)
    with pytest.raises(ValueError):
        make(cameras=["no_such_camera"])
    env.close()
