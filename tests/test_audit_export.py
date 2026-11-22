"""Audit, labels, selection rules (on synthetic takes) and the exporters (on scripted episodes)."""

import json
import shutil

import h5py
import numpy as np
import pytest

from xrcollect.audit import (AUDIT_VERSION, annotation_segments, audit_arrays, audit_dir, dataset_health, kept_spans,
                             load_curation, select, true_runs, update_curation, verdict, why_not)
from xrcollect.config import CurateConfig
from xrcollect.export.base import (MANIFEST, FeatureLayout, MixedVersions, constant_channels, ee9, read_manifest,
                                   run_export)

FPS = 50
META = {"stage_names": ["a", "b"], "arms": ["l", "r"], "objects": ["red", "blue"]}


def take(n=500, success=True, contacts=None, idle=(50, 450), progress_at=(100, 300), source=None, held=None,
         corr=None):
    vel = np.zeros((n, 14))
    vel[idle[0]:idle[1]] = 0.5
    c = np.zeros((n, 3), bool)
    for kind, (s, e) in (contacts or {}).items():
        c[s:e, ["link_table", "arm_arm", "tip_table"].index(kind)] = True
    progress = np.zeros(n, int)
    for k, t in enumerate(progress_at):
        progress[t:] = k + 1
    h = np.full((n, 2), -1) if held is None else held
    objects = np.tile([0.4, 0.2, 0.75, 1, 0, 0, 0], (n, 2, 1)).astype(float)
    data = {"obs/joint_vel": vel, "teleop/engaged": np.zeros((n, 2), bool), "task/progress": progress,
            "task/held": h, "contacts": c, "control/source": np.zeros(n, int) if source is None else source,
            "safety/correction": np.zeros(n) if corr is None else corr, "obs/objects": objects}
    return audit_arrays("episode_000000.hdf5", FPS, success, data, META)


def test_audit_numbers():
    held = np.full((500, 2), -1)
    held[120:200, 0] = 1
    corr = np.zeros(500)
    corr[210:215] = 0.2
    a = take(contacts={"link_table": (200, 220), "tip_table": (100, 150)}, held=held, corr=corr)
    assert a.num_steps == 500 and a.duration_s == 10 and a.active == (50, 450)
    assert a.contact_frac["link_table"] == pytest.approx(0.04) and a.contact_events["tip_table"] == 1
    assert a.idle_frac == pytest.approx(0.2) and a.stage_times == [2.0, 6.0] and a.safety_events == 1
    assert a.start_poses["red"] == [0.4, 0.2, 0.0]
    labels = [(p["label"], p["start"], p["stop"]) for p in a.phases]
    assert ("l: hold blue", 120, 200) in labels and ("a", 100, 101) in labels and ("b", 300, 301) in labels
    assert true_runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]


def test_selection_rules():
    rules = CurateConfig()
    assert verdict(take(success=False), rules, {}) == "failed take"
    assert verdict(take(success=False), CurateConfig(include_failures=True), {}) is None
    assert verdict(take(success=False), rules, {"episode_000000.hdf5": {"success": True}}) is None  # label wins
    assert "excluded by hand: wobbly" == verdict(take(), rules, {"episode_000000.hdf5": {"exclude": True,
                                                                                         "note": "wobbly"}})
    dragging = take(contacts={"tip_table": (50, 300)})
    assert "hand dragging" in verdict(dragging, rules, {})
    # A cut contact kind costs its moment (+- margin), not the take.
    a = take(contacts={"link_table": (200, 220)})
    assert verdict(a, rules, {}) is None and kept_spans(a, rules) == [(50, 190), (230, 450)]
    assert kept_spans(a, CurateConfig(trim_idle=False, cut_contacts=[])) == [(0, 500)]
    assert "arm link on the table" in verdict(a, CurateConfig(cut_contacts=[]), {})
    # Pieces shorter than a second are dropped.
    b = take(contacts={"arm_arm": (80, 85)})
    assert kept_spans(b, rules) == [(95, 450)]
    # Shared control: only the operator's steps.
    source = np.zeros(500, int)
    source[:200] = 1
    c = take(source=source)
    assert kept_spans(c, rules) == [(200, 450)] and kept_spans(c, CurateConfig(shared_frames="all")) == [(50, 450)]
    # Sub-task dataset: up to a second after stage 1; a failed take that got there still counts.
    sub = CurateConfig(until_stage=1)
    assert kept_spans(take(), sub) == [(50, 150)] and verdict(take(success=False), sub, {}) is None
    assert why_not(take(progress_at=()), sub, {}) == "nothing left after trimming"
    # A "bad" annotation is cut out; other labels are not.
    cur = {"episode_000000.hdf5": {"annotations": [{"start": 300, "stop": 340, "label": "bad"},
                                                   {"start": 100, "stop": 200, "label": "pick the block"}]}}
    assert kept_spans(take(), rules, cur) == [(50, 300), (340, 450)]


def test_curation_sidecar(tmp_path):
    f = "episode_000003.hdf5"
    e = update_curation(tmp_path, f, tags=["slow", " slow ", "odd grasp"], quality=9, note="x" * 600,
                        annotations=[{"start": 10, "stop": 5, "label": "nope"}, {"start": 5, "stop": 50, "label": "reach"}])
    assert e["tags"] == ["odd grasp", "slow"] and e["quality"] == 5 and len(e["note"]) == 500
    assert e["annotations"] == [{"start": 5, "stop": 50, "label": "reach"}]
    update_curation(tmp_path, f, success=False, exclude=True)
    assert load_curation(tmp_path)[f]["success"] is False and load_curation(tmp_path)[f]["exclude"] is True
    update_curation(tmp_path, f, success=None, exclude=False, tags=[], quality=0, note="", annotations=[])
    assert load_curation(tmp_path) == {}
    with pytest.raises(ValueError):
        update_curation(tmp_path, f, colour="red")


@pytest.fixture
def dataset(tmp_path, expert_episode):
    """Three scripted stacking episodes (different start layouts) in a dataset folder."""
    d = tmp_path / "data" / "episodes"
    d.mkdir(parents=True)
    for i in range(3):
        path, ok = expert_episode("tasks/stack_two_blocks", None, i)
        assert ok
        shutil.copy(path, d / f"episode_{i:06d}.hdf5")
    return d


def test_audit_dir_caches_and_select(dataset):
    audits = audit_dir(dataset)
    cache = json.loads((dataset / "audit.json").read_text())
    assert cache["version"] == AUDIT_VERSION and set(cache["episodes"]) == set(audits)
    again = audit_dir(dataset, progress=lambda n: pytest.fail(f"{n} audited again"))
    assert again == audits
    a = audits["episode_000000.hdf5"]
    assert a.success and len(a.stage_times) == 2 and a.stage_times[0] < a.stage_times[1]
    assert a.info["task"] == "stack_two_blocks" and set(a.start_poses) == {"red_block", "blue_block"}
    assert sum(p["kind"] == "hold" for p in a.phases) == 2 and sum(p["kind"] == "stage" for p in a.phases) == 2
    segments, rejected = select(dataset, audits, CurateConfig())
    assert len(segments) == 3 and not rejected and all(s.start > 0 for s in segments)   # the idle start is trimmed
    update_curation(dataset, "episode_000001.hdf5", exclude=True)
    update_curation(dataset, "episode_000002.hdf5", annotations=[{"start": 60, "stop": 200, "label": "pick red"}])
    segments, rejected = select(dataset, audits, CurateConfig())
    assert [s.path.name for s in segments] == ["episode_000000.hdf5", "episode_000002.hdf5"]
    assert rejected == {"episode_000001.hdf5": "excluded by hand"}
    labelled = annotation_segments(dataset, audits, CurateConfig())
    assert [(s.start, s.stop, s.label) for s in labelled] == [(60, 200, "pick red")]
    h = dataset_health(dataset, CurateConfig())
    assert h["count"] == 3 and h["used"] == 2 and h["success_rate"] == 1.0 and "expert" in h["operators"]
    assert len({tuple(e["start_poses"]["red_block"]) for e in h["episodes"]}) == 3   # coverage: 3 layouts


def test_hdf5_export_and_manifest(dataset, tmp_path):
    out = tmp_path / "exports" / "stack"
    layout = FeatureLayout(cameras=["cam_high", "cam_left_wrist"], image_size=[48, 64],
                           state=["joint_pos", "gripper_width", "ee_pose"], env_state=True)
    seen = []
    m = run_export(dataset, out, "hdf5", CurateConfig(), layout, progress=lambda d, t: seen.append((d, t)))
    assert m == read_manifest(out) and (out / MANIFEST).exists() and seen[-1] == (3, 3)
    assert m["episodes"] == 3 and m["exporter"] == "hdf5" and len(m["task_versions"]) == 1
    assert m["names"]["action"][7] == "l_gripper" and len(m["names"]["state"]) == 14 + 2 + 18
    assert m["layout"]["action_label"] == "executed" and m["segments"][0]["start"] > 0
    assert m["names"]["env_state"][:7] == [f"red_block_{k}" for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    with h5py.File(out / "episode_000000.hdf5") as f:
        n = m["segments"][0]["stop"] - m["segments"][0]["start"]
        assert f["observation/state"].shape == (n, 34) and f["action"].shape == (n, 16)
        assert f["observation/environment_state"].shape == (n, 14)
        img = f["observation/images/cam_high"]
        assert img.shape == (n, 48, 64, 3) and img.dtype == np.uint8 and img[n // 2].std() > 5
        assert "blue block" in f.attrs["instruction"]
        action, state = f["action"][()], f["observation/state"][()]
    # The executed joint target leads the measured joints; grippers are 0..1.
    assert np.abs(action[:, :7] - state[:, :7]).max() < 0.3 and set(np.unique(action[:, 7].round(3))) <= {0.0, 1.0} \
        or action[:, 7].min() >= 0
    # Images are a pure function of the recorded state: exporting again gives the same pixels (+-1).
    out2 = tmp_path / "exports" / "stack2"
    run_export(dataset, out2, "hdf5", CurateConfig(), layout)
    with h5py.File(out / "episode_000001.hdf5") as a, h5py.File(out2 / "episode_000001.hdf5") as b:
        ia, ib = a["observation/images/cam_left_wrist"][::40], b["observation/images/cam_left_wrist"][::40]
        assert np.abs(ia.astype(int) - ib.astype(int)).max() <= 1
        assert np.array_equal(a["action"][()], b["action"][()])
    with pytest.raises(FileExistsError):
        run_export(dataset, out, "hdf5")
    with pytest.raises(ValueError, match="no camera"):
        run_export(dataset, tmp_path / "x", "hdf5", layout=FeatureLayout(cameras=["cam_nope"]))


def test_export_refuses_mixed_task_versions_and_flags_constant_channels(dataset, tmp_path, expert_episode):
    single, _ = expert_episode("tasks/stack_two_blocks", "franka_single", 0)
    shutil.copy(single, dataset / "episode_000003.hdf5")
    layout = FeatureLayout(cameras=[], env_state=True)
    with pytest.raises(MixedVersions, match="2 task versions"):
        run_export(dataset, tmp_path / "mixed", "hdf5", layout=layout)
    (dataset / "episode_000003.hdf5").unlink()
    m = run_export(dataset, tmp_path / "ok", "hdf5", layout=layout)
    assert m["cameras"] == [] and not list((tmp_path / "ok").glob("**/images"))
    # A sub-task dataset that ends before the blue block is touched: its pose never varies -> flagged.
    sub = CurateConfig(until_stage=1, until_stage_tail_s=0.2)
    m = run_export(dataset, tmp_path / "sub", "hdf5", sub, FeatureLayout(cameras=[], env_state=True))
    flagged = m["constant_channels"]["env_state"]
    assert "blue_block_qx" in flagged and "red_block_x" not in flagged
    assert constant_channels({"s": np.array([[1.0, 2.0], [1.0, 3.0]])}, {"s": ["a", "b"]}) == {"s": ["a"]}


def test_annotation_export_uses_the_label_as_instruction(dataset, tmp_path):
    update_curation(dataset, "episode_000000.hdf5", annotations=[{"start": 60, "stop": 260, "label": "pick the red block"}])
    m = run_export(dataset, tmp_path / "lang", "hdf5", layout=FeatureLayout(cameras=[], segments="annotations"))
    assert m["episodes"] == 1 and m["segments"][0]["label"] == "pick the red block" and m["steps"] == 200
    with h5py.File(tmp_path / "lang" / "episode_000000.hdf5") as f:
        assert f.attrs["instruction"] == "pick the red block" and f["action"].shape[0] == 200


def test_export_at_a_lower_fps_keeps_every_nth_step(dataset, tmp_path):
    update_curation(dataset, "episode_000000.hdf5", annotations=[{"start": 60, "stop": 260, "label": "pick"}])
    full = run_export(dataset, tmp_path / "full", "hdf5", layout=FeatureLayout(cameras=[], segments="annotations"))
    half = run_export(dataset, tmp_path / "half", "hdf5",
                      layout=FeatureLayout(cameras=[], segments="annotations", fps=full["fps"] // 2))
    assert half["fps"] == full["fps"] // 2 and half["steps"] == 100 and full["steps"] == 200
    with h5py.File(tmp_path / "full" / "episode_000000.hdf5") as f, \
            h5py.File(tmp_path / "half" / "episode_000000.hdf5") as h:
        assert h.attrs["fps"] == full["fps"] // 2
        # Frames from steps 60, 62, ...; each action is the last one of its window (steps 61, 63, ...).
        np.testing.assert_array_equal(h["observation/state"][:], f["observation/state"][0::2])
        np.testing.assert_array_equal(h["action"][:], f["action"][1::2])
    with pytest.raises(ValueError, match="must divide"):
        run_export(dataset, tmp_path / "odd", "hdf5", layout=FeatureLayout(cameras=[], fps=full["fps"] - 1))


def test_ee9_is_continuous_across_the_quaternion_sign():
    q = np.array([0.02, 1.0, 0.0, 0.0])
    q /= np.linalg.norm(q)
    assert np.allclose(ee9(np.r_[1, 2, 3, q]), ee9(np.r_[1, 2, 3, -q])) and ee9(np.r_[1, 2, 3, q]).shape == (9,)
