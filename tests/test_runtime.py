"""The runtime, driven tick by tick without a headset: teleop, recording, replay, branch, ghost, shared control,
hot reload, edit mode and capture goal."""

import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import controllers, ticks
from xrcollect.audit import load_curation
from xrcollect.policy import HoldRunner, LayoutMismatch, env_info, load_policy
from xrcollect.recorder import SOURCE_OPERATOR, SOURCE_POLICY, RecState, read_episode
from xrcollect.runtime import Runtime
from xrcollect.task.loader import load_bundle


@pytest.fixture
def rt(cfg):
    r = Runtime(cfg)
    yield r
    r.recorder.close()


@pytest.fixture
def events(rt):
    out = []
    rt.add_listener(out.append)
    return out


def engage(rt, side="left"):
    kw = {f"squeeze_{side}": 1.0}
    ticks(rt, 3, controllers(**kw))     # Grip press toggles the clutch
    ticks(rt, 6, controllers())


def record_take(rt, n=60, save=True, **save_kw):
    rt._handle("record_toggle", {}, time.monotonic())
    ticks(rt, n, controllers())
    rt._handle("record_toggle", {}, time.monotonic())
    assert rt.recorder.state == RecState.REVIEW
    ticks(rt, 6, controllers())
    review = rt.status()["review"]
    if save:
        rt._handle("save", save_kw, time.monotonic())
        rt.recorder.flush()
    return review


def test_clutched_teleop_moves_the_arm_with_the_hand_and_never_jumps(rt, events):
    ticks(rt, 10, controllers())
    start = rt.env.tcp_poses()[:, :3].copy()
    engage(rt, "left")
    assert rt._engaged["l"] and not rt._engaged["r"]
    assert np.linalg.norm(rt.env.tcp_poses()[:, :3] - start, axis=1).max() < 0.002    # engaging moves nothing
    for k in range(50):                                   # hand 10 cm to its right (XR +x = world -y)
        ticks(rt, 1, controllers(left_pos=(-0.2 + 0.002 * (k + 1), 1.1, -0.35)))
    ticks(rt, 40, controllers(left_pos=(-0.1, 1.1, -0.35)))
    moved = rt.env.tcp_poses()[:, :3] - start
    assert abs(moved[0][1] + 0.10) < 0.01 and np.abs(moved[0][[0, 2]]).max() < 0.01
    assert np.linalg.norm(moved[1]) < 0.002                # the other arm stays
    assert any(e.get("type") == "sound" and e["name"] == "engage" and e["arm"] == "l" for e in events)
    assert any(e.get("type") == "haptic" and e["hand"] == "l" for e in events)
    # A tracking jump drops the clutch instead of yanking the arm.
    ticks(rt, 2, controllers(left_pos=(0.3, 1.1, -0.35)))
    assert not rt._engaged["l"]
    st = rt.status()
    assert st["arms"] == ["l", "r"] and st["mapping"] == {"l": "l", "r": "r"} and st["task"] == "stack_two_blocks"


def test_input_older_than_the_timeout_releases_the_arm(rt):
    engage(rt, "right")
    assert rt._engaged["r"]
    time.sleep(0.3)
    ticks(rt, 2)                                           # no new input
    assert not rt._engaged["r"]


def test_record_review_save_writes_a_replayable_episode_with_labels(rt, events):
    engage(rt, "left")
    review = record_take(rt, 80, success=False, tags=["slow", "recovery"], quality=4, note="test take")
    assert review["duration_s"] == 1.6 and review["rejected"] == "failed take" and review["tags"]
    saved = [e for e in events if e.get("type") == "saved"]
    assert len(saved) == 1
    path = Path(saved[0]["path"])
    meta, data = read_episode(path)
    assert meta["num_steps"] == 80 and meta["success"] is False and meta["source"] == "teleop"
    assert meta["input_mode"] == "controllers" and meta["task_version"] == rt.bundle.version
    assert data["teleop/engaged"][:, 0].all() and data["teleop/left/source"][0] == 1
    assert (data["control/source"] == SOURCE_OPERATOR).all()
    # Labels from the review card live in the sidecar; the episode file has none of them.
    assert load_curation(path.parent)[path.name] == {"note": "test take", "quality": 4, "tags": ["recovery", "slow"]}
    assert "tags" not in meta and "quality" not in meta
    ticks(rt, 6, controllers())                            # the status snapshot is refreshed every 5th tick
    assert rt.status()["session"]["saved"] == 1 and rt.recorder.state == RecState.IDLE


def test_buttons_send_the_commands_of_the_one_table(rt):
    from xrcollect.teleop import BUTTONS, command_table

    ticks(rt, 2, controllers())
    ticks(rt, 2, controllers(right_primary=1.0))           # A: record
    assert rt.recorder.state in (RecState.COUNTDOWN, RecState.RECORDING)
    ticks(rt, 5, controllers())
    ticks(rt, 2, controllers(right_secondary=1.0))         # B: discard
    assert rt.recorder.state == RecState.IDLE and rt.recorder.discarded_count == 1
    table = command_table()
    assert len(table) == len(BUTTONS) and {"hand", "button", "label"} <= set(table[0])
    assert any(b["command"] == "save" and b["hand"] == "l" and b["button"] == "primary" for b in table)


def test_recorded_action_is_the_executed_one_and_dart_drift_is_not_in_it(cfg):
    cfg.record.action_noise = 0.01
    rt = Runtime(cfg)
    rt._noise = np.random.default_rng(0)                   # the drift is random: make the take reproducible
    engage(rt, "left")
    record_take(rt, 150)
    path = sorted(cfg.out_dir.glob("*.hdf5"))[0]
    meta, data = read_episode(path)
    from xrcollect.embodiment.base import split_action
    from xrcollect.env import env_from_episode

    env = env_from_episode(meta)
    q_rec, grip = split_action(data["action/joint_pos"], 7)
    q_exec = env.executed_targets(data["sim/state"])
    drift = (q_exec - q_rec)[:-1]
    assert 0.001 < np.abs(drift[:, :7]).max() < 0.08       # the engaged (left) arm was perturbed ...
    assert np.abs(drift[:, 7:]).max() < 1e-9               # ... the released one was not
    # A slow drift, not per-tick noise: smoothed it moves ~0.0007 rad a tick (1 sigma), white noise of the same
    # strength would jump ~0.014.
    assert np.abs(np.diff(drift[:, :7], axis=0)).max() < 0.005
    # What was sent reproduces the recorded states exactly.
    env.set_state(data["sim/state"][0])
    for i in range(len(q_exec) - 1):
        env.apply_action(q_exec[i], grip[i])
        env.step()
        assert np.array_equal(env.get_state(), data["sim/state"][i + 1])
    rt.recorder.close()


def test_replay_seek_step_and_branch(rt, events):
    engage(rt, "left")
    record_take(rt, 100)
    path = str(sorted(rt.cfg.out_dir.glob("*.hdf5"))[0])
    _, data = read_episode(path, keys=["sim/state"])
    rt._handle("replay", {"path": path, "paused": True, "i": 40}, 0.0)
    ticks(rt, 6)
    st = rt.status()["replay"]
    assert st["i"] == 40 and st["n"] == 100 and st["paused"] and not st["foreign"]
    assert np.array_equal(rt.env.get_state(), data["sim/state"][40])
    rt._handle("replay_step", {"delta": 5}, 0.0)
    ticks(rt, 1)
    assert np.array_equal(rt.env.get_state(), data["sim/state"][45])
    rt._handle("replay_options", {"speed": 2.0}, 0.0)
    rt._handle("replay_pause", {"paused": False}, 0.0)
    ticks(rt, 10)
    assert rt._replay["i"] == 65
    # Branch: take over at a paused frame; the new take starts at exactly that state and names its origin.
    rt._handle("replay_seek", {"i": 30}, 0.0)
    rt._handle("replay_pause", {"paused": True}, 0.0)
    ticks(rt, 1)
    rt._handle("replay_branch", {}, time.monotonic())
    ticks(rt, 30, controllers())
    assert rt.recorder.recording and rt.status()["branch"]["step"] == 30
    rt._handle("save", {}, time.monotonic())
    rt.recorder.flush()
    branch = sorted(rt.cfg.out_dir.glob("*.hdf5"))[1]
    meta, bdata = read_episode(branch)
    assert meta["branched_from"] == path and meta["branch_step"] == 30
    assert np.array_equal(bdata["sim/state"][0], data["sim/state"][30])


def test_replay_of_another_tasks_episode_shows_its_own_scene(rt, events, expert_episode):
    tower, _ = expert_episode("tasks/duplo_tower")
    before = rt.scene()[0]["version"]
    rt._handle("replay", {"path": str(tower), "paused": True, "i": 10}, 0.0)
    ticks(rt, 6)
    assert rt.status()["replay"]["foreign"] and rt.status()["task"] == "duplo_tower"
    assert "green_1x2x2" in rt.scene()[0]["objects"] and rt.scene()[0]["version"] > before
    assert any(e.get("type") == "scene_changed" and e["task"] == "duplo_tower" for e in events)
    rt._handle("replay_branch", {}, time.monotonic())      # another scene: replay only
    assert rt.recorder.state == RecState.IDLE
    rt._handle("replay_stop", {}, 0.0)
    ticks(rt, 6, controllers())
    assert rt.status()["task"] == "stack_two_blocks" and "red_block" in rt.scene()[0]["objects"]


def test_ghost_frames_follow_a_reference_take(rt, expert_episode):
    from xrcollect.protocol import MSG_GHOST, parse_pose_frame

    ref, _ = expert_episode("tasks/stack_two_blocks")
    rt._handle("ghost", {"path": str(ref)}, 0.0)
    ticks(rt, 8, controllers())
    seq, frame = rt.latest_ghost_frame()
    mtype, _, _, _, poses = parse_pose_frame(frame)
    assert seq > 0 and mtype == MSG_GHOST and len(poses) == rt.env.model.nbody
    assert rt.status()["ghost"]["n"] > 100
    rt._handle("ghost", {}, 0.0)
    assert rt._ghost is None


def test_shared_control_takeover_does_not_jump_and_is_recorded(rt):
    rt.policy = HoldRunner(env_info(rt.env), wiggle=0.2)
    rt.control_mode = "shared"
    rt._handle("record_toggle", {}, time.monotonic())
    ticks(rt, 60, controllers())
    assert rt._driver == "policy"
    tcp = rt.env.tcp_poses()[:, :3].copy()
    ticks(rt, 3, controllers(squeeze_left=1.0))            # Grip: the operator takes the left arm
    ticks(rt, 20, controllers())
    assert rt._driver == "operator"
    assert np.linalg.norm(rt.env.tcp_poses()[:, :3] - tcp, axis=1).max() < 0.01
    sources = [s["control"]["source"] for s in rt.recorder.steps]
    assert sources[0] == SOURCE_POLICY and sources[-1] == SOURCE_OPERATOR
    ticks(rt, 3, controllers(squeeze_left=1.0))            # released: the policy resumes
    ticks(rt, 2, controllers())
    assert rt._driver == "policy"
    rt._handle("set_control_mode", {"mode": "teleop"}, 0.0)
    ticks(rt, 2, controllers())
    assert rt._driver == "operator"


def test_policy_layout_must_match_and_modes_need_a_policy(rt):
    rt._handle("set_control_mode", {"mode": "policy"}, 0.0)
    assert rt.control_mode == "teleop"
    assert load_policy("xrcollect.policy:HoldRunner", rt.env).name == "hold"

    class Wrong(HoldRunner):
        def __init__(self, info, **kw):
            super().__init__(info)
            self.layout = {"action_names": ["a", "b"]}

    import xrcollect.policy as P
    P.Wrong = Wrong
    with pytest.raises(LayoutMismatch):
        load_policy("xrcollect.policy:Wrong", rt.env)


def test_hot_reload_is_refused_while_recording_and_edits_change_the_version(cfg, task_copy):
    cfg.task = str(task_copy)
    rt = Runtime(cfg)
    events = []
    rt.add_listener(events.append)
    v0, scene0 = rt.bundle.version, rt.scene()[0]["version"]
    rt._handle("record_toggle", {}, time.monotonic())
    ticks(rt, 5, controllers())
    rt._handle("edit", {"ops": [{"op": "region_set", "name": "build_area", "center": [0.5, 0.1],
                                 "half": [0.06, 0.06]}]}, 0.0)
    assert rt.bundle.version == v0 and rt.recorder.recording          # refused: an episode has one model
    assert "0.1" not in (task_copy / "task.yaml").read_text().split("build_area")[1].split("\n")[0]
    rt._handle("discard", {}, 0.0)
    rt._handle("edit", {"ops": [
        {"op": "region_set", "name": "build_area", "center": [0.5, 0.1], "half": [0.06, 0.06]},
        {"op": "object_set", "object": "red_block", "fields": {"mass": 0.08}},
        {"op": "object_add", "name": "green_block", "asset": "blocks/cube_4cm", "pos": [0.6, -0.1],
         "fields": {"rgba": [0.1, 0.7, 0.2, 1]}},
        {"op": "camera_set", "name": "cam_side", "pos": [0.4, 1.0, 1.3], "lookat": [0.45, 0, 0.8], "fovy": 55},
        {"op": "spawn_set", "object": "green_block", "center": [0.6, -0.1], "half": [0.03, 0.03], "yaw_deg": [0, 0]},
    ]}, 0.0)
    assert rt.bundle.version != v0 and rt.scene()[0]["version"] > scene0
    assert "green_block" in rt.env.objects and "cam_side" in rt.env.camera_names
    assert rt.bundle.task["regions"]["build_area"]["center"] == [0.5, 0.1]
    assert any(e.get("type") == "scene_changed" for e in events)
    text = (task_copy / "task.yaml").read_text()
    assert "# named areas" in text and "green_block" in text          # comments survive the edit
    ticks(rt, 5, controllers())
    assert rt.status()["task_version"] == rt.bundle.version
    # An edit that doesn't build is refused and the file is put back.
    before = (task_copy / "task.yaml").read_text()
    rt._handle("edit", {"ops": [{"op": "object_add", "name": "ghost", "asset": "no/such_asset"}]}, 0.0)
    assert (task_copy / "task.yaml").read_text() == before and "ghost" not in rt.env.objects
    assert any(e.get("type") == "edit_error" for e in events)
    rt.recorder.close()


def test_edit_mode_moves_objects_and_capture_goal_defines_the_task(cfg, task_copy):
    cfg.task = str(task_copy)
    rt = Runtime(cfg)
    rt._handle("edit_mode", {"enabled": True}, 0.0)
    ticks(rt, 6, controllers())
    assert rt.status()["edit"] is not None
    t0 = rt.env.data.time
    ticks(rt, 10, controllers(squeeze_left=1.0))
    assert rt.env.data.time == t0                                      # physics paused, arms don't move
    th = rt.env.table_height
    rt._handle("edit_move", {"object": "red_block", "pos": [0.48, 0.05, th], "yaw": 0.0}, 0.0)
    rt._handle("edit_move", {"object": "blue_block", "pos": [0.48, 0.05, th + 0.04], "yaw": np.pi / 2,
                             "free_z": True}, 0.0)
    ticks(rt, 2)
    assert np.allclose(rt.env.data.xpos[rt.env.world.body["blue_block"]], [0.48, 0.05, th + 0.04], atol=1e-6)
    rt._handle("capture_goal", {"tol_pos": 0.008}, 0.0)
    raw = yaml.safe_load((task_copy / "task.yaml").read_text())
    stage = next(s for s in raw["stages"] if s["name"] == "blue_block_placed")
    rel = stage["when"]["relative_pose"]
    assert rel["base"] == "red_block" and np.allclose(rel["pos"], [0, 0, 0.04], atol=1e-3)
    assert rel["yaw_deg"] == 90.0 and rel["tol_pos"] == 0.008
    assert raw["goal_view"]["anchor"] == "red_block" and "success" not in raw
    assert "blue_block_placed" in rt.env.task.stage_names and rt.status is not None
    # Moving an object in edit mode and committing writes its pose (and moves its start region along).
    rt._handle("edit_move", {"object": "red_block", "pos": [0.40, 0.25, th], "yaw": 0.0}, 0.0)
    rt._handle("edit_commit", {}, 0.0)
    raw = yaml.safe_load((task_copy / "task.yaml").read_text())
    assert raw["objects"]["red_block"]["pos"] == [0.4, 0.25]
    assert raw["spawn"]["red_block"]["region"]["center"] == [0.4, 0.25]
    rt._handle("edit_mode", {"enabled": False}, 0.0)
    ticks(rt, 10, controllers())
    assert rt.status()["edit"] is None
    # The captured goal is a working task: arranging the objects that way is success.
    from test_task import put

    env = rt.env
    put(env, "red_block", [0.5, 0.0, th])
    put(env, "blue_block", [0.5, 0.0, th + 0.04], yaw=np.pi / 2)
    assert env.progress() == len(env.task.stage_names) and env.success()
    put(env, "blue_block", [0.5, 0.0, th + 0.04], yaw=0.0)
    assert env.progress() < len(env.task.stage_names)
    assert load_bundle(task_copy).version == rt.bundle.version
    rt.recorder.close()


def test_task_events_reach_the_operator_as_cues(cfg):
    """Stage reached -> the cue the task file names (placed) + its toast; success -> complete."""
    from test_task import put

    rt = Runtime(cfg)
    events = []
    rt.add_listener(events.append)
    ticks(rt, 6, controllers())
    th = rt.env.table_height
    put(rt.env, "red_block", [0.5, 0.0, th])
    ticks(rt, 6, controllers())
    sounds = [e["name"] for e in events if e.get("type") == "sound"]
    toasts = [e["text"] for e in events if e.get("type") == "toast"]
    assert sounds.count("placed") == 1 and "red block in the area" in toasts
    assert rt.status()["progress"] == 1 and rt.status()["next"]["object"] == "blue_block"
    guide = rt.status()["guide"]
    assert guide["object"] == "blue_block" and guide["dist_mm"] > 100
    put(rt.env, "blue_block", [0.5, 0.0, th + 0.04])
    ticks(rt, 60, controllers())
    sounds = [e["name"] for e in events if e.get("type") == "sound"]
    assert sounds.count("placed") == 2 and "complete" in sounds
    assert rt.status()["success"] and rt.status()["stages"] == [True, True]
    rt.recorder.close()


def test_single_arm_goes_to_the_right_hand_and_hands_can_be_swapped(cfg):
    cfg.task = "tasks/open_drawer"
    rt = Runtime(cfg)
    ticks(rt, 6, controllers())
    assert rt.arms == ["arm"] and rt.status()["mapping"] == {"r": "arm"}
    engage(rt, "left")
    assert not rt._engaged["arm"]                                       # the left hand drives nothing
    engage(rt, "right")
    assert rt._engaged["arm"] and rt.status()["hand_of"] == {"arm": "r"}
    rt._handle("swap_hands", {}, 0.0)                                   # left-handed operator
    ticks(rt, 6, controllers())
    assert rt.status()["mapping"] == {"l": "arm"} and not rt._engaged["arm"]
    engage(rt, "left")
    assert rt._engaged["arm"]
    rt._handle("reload_scene", {"task": "tasks/stack_two_blocks"}, 0.0)  # switching tasks is a reload
    ticks(rt, 6, controllers())
    assert rt.arms == ["l", "r"] and rt.status()["task"] == "stack_two_blocks"
    rt.recorder.close()


def test_rollout_runs_a_policy_headless_and_saves_replayable_takes(tmp_path):
    import json
    import shutil

    from typer.testing import CliRunner

    from xrcollect.cli import _policy_args, app

    assert _policy_args(["checkpoint=out/x", "n_action_steps=50", "wiggle=0.1"]) == \
        {"checkpoint": "out/x", "n_action_steps": 50, "wiggle": 0.1}
    res = CliRunner().invoke(app, ["rollout", "xrcollect.policy:HoldRunner", "--policy-arg", "wiggle=0.2",
                                   "--task", "tasks/stack_two_blocks", "--n", "2", "--seconds", "0.2",
                                   "--save", str(tmp_path / "roll"), "--out", str(tmp_path / "r.json"),
                                   *(["--video", str(tmp_path / "vid")] if shutil.which("ffmpeg") else [])])
    assert res.exit_code == 0, res.output
    out = json.loads((tmp_path / "r.json").read_text())
    assert out["success_rate"] == 0.0 and [e["success"] for e in out["episodes"]] == [False, False]
    assert out["args"] == {"wiggle": 0.2} and len({e["seed"] for e in out["episodes"]}) == 2
    files = sorted((tmp_path / "roll").glob("*.hdf5"))
    meta, data = read_episode(files[0])
    assert len(files) == 2 and meta["source"] == "policy" and meta["operator"] == "hold"
    assert (data["control/source"] == SOURCE_POLICY).all() and len(data["time"]) == 10
    if shutil.which("ffmpeg"):
        assert sorted(f.name for f in (tmp_path / "vid").iterdir()) == ["ep000_fail.mp4", "ep001_fail.mp4"]
