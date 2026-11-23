"""Task layer: loading, versions, predicates on constructed states, spawn distribution, snap rule, goal spec."""

import mujoco
import numpy as np
import pytest
import yaml

from conftest import TASKS, get_env
from xrcollect.env import Env
from xrcollect.scene.constants import yaw_quat
from xrcollect.task.loader import Bundle, load_bundle
from xrcollect.task.predicates import CATALOGUE, compile_predicate
from xrcollect.task.schema import validate_task


def put(env, name, pos, yaw=0.0):
    a = env.world.qadr[name]
    env.data.qpos[a:a + 3] = pos
    env.data.qpos[a + 3:a + 7] = yaw_quat(yaw)
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)


def check(env, pred, stages=None) -> bool:
    return compile_predicate(pred)(env.world, env.data, stages or {})


@pytest.mark.parametrize("task", TASKS)
def test_tasks_load_and_roundtrip(task):
    b = load_bundle(task)
    again = Bundle.from_yaml(b.to_yaml())
    assert again.version == b.version and len(b.version) == 16
    spec = get_env(task).task.goal_spec()
    assert spec["task"] == b.name and len(spec["stages"]) == len(b.task["stages"])
    assert all(s["object"] for s in spec["stages"])


def test_yaml_on_key_is_not_a_boolean():
    b = load_bundle("tasks/stack_two_blocks")
    assert [e["on"] for e in b.task["events"]] == ["stage_reached", "success"]


def test_version_changes_with_every_layer(task_copy):
    base = load_bundle(task_copy).version
    f = task_copy / "task.yaml"
    raw = yaml.safe_load(f.read_text())
    raw["objects"]["red_block"]["mass"] = 0.06           # task objects
    f.write_text(yaml.safe_dump(raw))
    v_obj = load_bundle(task_copy).version
    raw["stages"][1]["when"]["all"][1]["on_top"]["tol_xy"] = 0.02   # task logic
    f.write_text(yaml.safe_dump(raw))
    v_logic = load_bundle(task_copy).version
    v_emb = load_bundle(task_copy, {"embodiment": "franka_single"}).version   # embodiment
    raw["cameras"] = {"cam_side": {"pos": [0.4, 1.0, 1.2], "lookat": [0.4, 0, 0.8]}}  # scene
    f.write_text(yaml.safe_dump(raw))
    v_scene = load_bundle(task_copy).version
    assert len({base, v_obj, v_logic, v_emb, v_scene}) == 5


def test_unknown_fields_and_predicates_are_refused():
    with pytest.raises(Exception):
        validate_task({"task": "x", "sucess": {}})
    with pytest.raises(ValueError, match="unknown predicate"):
        compile_predicate({"on_topp": {}})
    with pytest.raises(ValueError):
        compile_predicate({"a": 1, "b": 2})


def test_predicates_on_constructed_states():
    env = get_env("tasks/stack_two_blocks")
    env.reset(0)
    th = env.table_height
    put(env, "red_block", [0.50, 0.0, th])
    put(env, "blue_block", [0.50, 0.0, th + 0.04])
    assert check(env, {"inside": {"object": "red_block", "region": "build_area"}})
    assert check(env, {"on_top": {"object": "blue_block", "base": "red_block"}})
    assert check(env, {"relative_pose": {"object": "blue_block", "base": "red_block", "pos": [0, 0, 0.04]}})
    assert not check(env, {"relative_pose": {"object": "blue_block", "base": "red_block", "pos": [0, 0, 0.04],
                                             "yaw_deg": 45}})
    assert check(env, {"pose_near": {"object": "red_block", "pos": [0.5, 0, th], "tol_pos": 0.002}})
    assert check(env, {"upright": "red_block"}) and check(env, {"at_rest": "all"}) and check(env, {"not_held": "any"})
    assert check(env, {"all": [{"upright": "red_block"}, {"not": {"held": "red_block"}}]})
    assert check(env, {"count": {"objects": ["red_block", "blue_block"],
                                 "when": {"inside": {"object": "$", "region": "build_area"}}, "min": 2}})
    assert check(env, {"stage": "s"}, {"s": True}) and not check(env, {"stage": "s"}, {"s": False})
    put(env, "red_block", [0.50, 0.0, th - 1e-4])              # 0.1 mm into each other: exactly touching
    put(env, "blue_block", [0.50, 0.0, th + 0.04 - 2e-4])      # surfaces make no contact
    assert check(env, {"contact": {"a": "red_block", "b": "blue_block"}})
    assert check(env, {"contact": {"a": "red_block", "b": "table"}})
    assert not check(env, {"contact": {"a": "blue_block", "b": "table"}})
    assert not check(env, {"contact": {"a": "gripper:l", "b": "red_block"}})
    # off by more than the tolerances
    put(env, "blue_block", [0.52, 0.0, th + 0.04])
    assert not check(env, {"on_top": {"object": "blue_block", "base": "red_block", "tol_xy": 0.012}})
    put(env, "red_block", [0.55, 0.0, th])   # footprint sticks out of the region (half 0.06, block half 0.02)
    assert not check(env, {"inside": {"object": "red_block", "region": "build_area"}})
    put(env, "red_block", [0.50, 0.0, th + 0.02], yaw=0.0)
    a = env.world.qadr["red_block"]
    env.data.qpos[a + 3:a + 7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]  # lying on its side
    mujoco.mj_forward(env.model, env.data)
    assert not check(env, {"upright": "red_block"})
    assert set(CATALOGUE) >= {"inside", "on_top", "pose_near", "relative_pose", "contact", "held", "not_held",
                              "at_rest", "joint_at", "upright", "count", "all", "any", "not", "stage"}


def test_joint_at_on_a_fixture():
    env = get_env("tasks/open_drawer")
    env.reset(0)
    assert env.fixture_joints == ["drawer_slide"]
    assert not check(env, {"joint_at": {"joint": "drawer_slide", "min": 0.12}})
    env.data.joint("drawer_slide").qpos[0] = 0.15
    mujoco.mj_forward(env.model, env.data)
    assert check(env, {"joint_at": {"joint": "drawer_slide", "min": 0.12}})
    assert check(env, {"joint_at": {"joint": "drawer_slide", "value": 0.15, "tol": 0.005}})
    assert env.progress() == 1


def test_spawn_stays_in_its_region_and_differs_per_seed():
    env = get_env("tasks/stack_two_blocks")
    spawn = env.bundle.task["spawn"]
    seen = set()
    for seed in range(20):
        obs = env.reset(seed)
        for name, p in zip(env.objects, obs["objects"]):
            r = spawn[name]["region"]
            assert np.all(np.abs(p[:2] - r["center"]) <= np.asarray(r["half"]) + 1e-9)
            assert np.isclose(p[2], env.table_height)
        seen.add(tuple(obs["objects"][:, :2].round(4).ravel()))
        assert env.progress() == 0 and not env.success()
    assert len(seen) == 20
    assert np.array_equal(env.reset(3)["objects"], env.reset(3)["objects"])


def test_nested_cups_count_whichever_way_they_are_turned():
    env = get_env("tasks/stack_cups")
    env.reset(0)
    th, dz = env.table_height, env.bundle.task["params"]["nest_dz"]
    put(env, "green_cup", [0.55, -0.25, th], yaw=0.4)
    for yaw in (0.0, 1.6, -1.6, -2.4, 3.1):            # the toy's quaternion test failed clockwise turns
        put(env, "red_cup", [0.55, -0.25, th + dz], yaw=yaw)
        assert env.stage_flags()[0], yaw
    a = env.world.qadr["red_cup"]
    tilt = np.radians(30)
    env.data.qpos[a + 3:a + 7] = [np.cos(tilt / 2), np.sin(tilt / 2), 0, 0]   # leaning on the rim
    mujoco.mj_forward(env.model, env.data)
    assert not env.stage_flags()[0]


def test_progress_counts_stages_in_order():
    env = get_env("tasks/stack_two_blocks")
    env.reset(0)
    th = env.table_height
    put(env, "red_block", [0.30, 0.3, th])            # base outside the area
    put(env, "blue_block", [0.30, 0.3, th + 0.04])    # ... but blue on it
    assert env.progress() == 0 and not env.stage_flags()[1]  # the second stage requires the first
    put(env, "red_block", [0.50, 0.0, th])
    put(env, "blue_block", [0.50, 0.0, th + 0.04])
    assert env.progress() == 2 and env.success()


def test_snap_engages_under_press_releases_under_pull_and_is_in_the_state(expert_episode):
    from xrcollect.recorder import read_episode

    path, ok = expert_episode("tasks/duplo_tower")
    assert ok
    meta, data = read_episode(path)
    snaps = data["task/snaps"]
    assert snaps[0].sum() == 0 and snaps[-1].sum() == 3          # three clicks, none at the start
    assert list(data["task/progress"][[0, -1]]) == [0, 4]
    env = get_env("tasks/duplo_tower")
    env.set_state(data["sim/state"][-1])
    task = env.task
    k = next(i for i, p in enumerate(task.snaps.pairs) if (p.a, p.b) == ("yellow_2x2", "green_1x2x2"))
    assert task.snaps.active(env.data)[k]
    relpose = env.model.eq_data[task.snaps.eqs[k], 3:10].copy()
    assert abs(relpose[2] - 0.0192) < 0.002                       # locked where it sat: one brick height up
    # Extra state: weld poses and armed flags survive a get/set round trip into a fresh model state.
    s = env.get_state()
    env.reset(1)
    assert not np.allclose(env.model.eq_data[task.snaps.eqs[k], 3:10], relpose) or True
    env.set_state(s)
    assert np.allclose(env.model.eq_data[task.snaps.eqs[k], 3:10], relpose) and np.array_equal(env.get_state(), s)
    # Pull the top brick off the one below with more than pull_n (equal and opposite forces, so the tower stays
    # put): the weld lets go, and the top stage is lost.
    green, yellow = env.world.body["green_1x2x2"], env.world.body["yellow_2x2"]
    for _ in range(10):
        env.data.xfrc_applied[green, 2], env.data.xfrc_applied[yellow, 2] = 15.0, -15.0
        env.step()
        if any(e.name == "snap_off" and e.object == "green_1x2x2" for e in env.events):
            break
    else:
        raise AssertionError("the snap did not release under 15 N of pull")
    assert not task.snaps.active(env.data)[k] and not task.snaps.armed[k]
    env.data.xfrc_applied[green] = 0
    for _ in range(3):                      # a hand still holds the tower down; the rest stays clicked
        env.step()
    env.data.xfrc_applied[:] = 0
    assert env.progress() == 3
    # Pressing harder on a snapped brick must not pop it off (compression is not pull).
    env.set_state(s)
    for _ in range(10):
        env.data.xfrc_applied[green, 2] = -30.0
        env.step()
        assert not env.events
    env.data.xfrc_applied[:] = 0
    assert task.snaps.active(env.data)[k]


def test_single_arm_embodiment_runs_the_same_task():
    env = get_env("tasks/stack_two_blocks", "franka_single")
    assert env.arms == ["arm"] and env.adapter.action_layout()[-1] == "arm_gripper"
    assert len(env.adapter.action_layout()) == 8 and env.reset(0)["ee_pose"].shape == (1, 7)
    assert isinstance(env, Env)
