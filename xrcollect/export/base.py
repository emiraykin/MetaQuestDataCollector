"""The dataset boundary: one exporter interface, one feature layout, one manifest.

`run_export` selects the segments with the same function the review card and the episode list use (audit.select),
builds each segment's frames (state, action, optional object state, camera images re-rendered from `sim/state` in
the scene the episode was recorded in) and hands them to an exporter plugin. It refuses to mix task versions unless
told to, and flags channels that never vary in the data (a policy input normalized by ~0 explodes at run time).

An exporter is a class with:
    def begin(self, out: Path, info: dict) -> None          # info: fps, layout, state/action names, image shapes
    def add_episode(self, frames: dict, info: dict) -> None  # frames: arrays (T, ...); info: instruction, source
    def finish(self) -> dict                                 # extra manifest fields
Register it in EXPORTERS ("name": "module:Class").
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from xrcollect.audit import Segment, annotation_segments, audit_dir, load_curation, select
from xrcollect.config import ROOT, CurateConfig
from xrcollect.embodiment.base import join_action, split_action
from xrcollect.env import env_from_episode
from xrcollect.recorder import read_episode
from xrcollect.scene.render import AppearanceRandomizer, CameraRenderer
from xrcollect.task.loader import load_object

log = logging.getLogger(__name__)
MANIFEST = "xrcollect_manifest.json"
EXPORTERS = {"hdf5": "xrcollect.export.hdf5:Hdf5Exporter", "lerobot": "xrcollect.export.lerobot:LeRobotExporter"}
STATE_CHANNELS = ("joint_pos", "gripper_width", "joint_vel", "ee_pose", "ee_wrench")
# The hand wrench spans 0.1 N (a carried part) to ~250 N (pressing one on), so it goes in log-compressed:
# sign(f) log(1 + |f| / scale). Min-max normalized as is, everything below ~10 N would look like zero.
WRENCH_SCALE = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])  # N, Nm
CONSTANT_STD = 1e-6


class FeatureLayout(BaseModel):
    """What a policy sees and outputs. Stored in every dataset's manifest; a policy runner declares the same
    fields and is refused when they don't match the scene (policy.load_policy)."""
    cameras: list[str] | None = None            # None = every camera of the scene; [] = no images
    image_size: list[int] = Field(default_factory=lambda: [224, 224])   # height, width
    # Proprioception channels, an explicit choice per dataset (everything is recorded; export selectively).
    # joint_vel is off by default: with joint-position targets the target leads the measured joints by the
    # servo lag, a policy can copy the velocity and the closed loop drifts.
    state: list[Literal["joint_pos", "gripper_width", "joint_vel", "ee_pose", "ee_wrench"]] = Field(
        default_factory=lambda: ["joint_pos", "gripper_width"])
    env_state: bool = False                     # object poses + fixture joints (privileged: simulation only)
    action: Literal["joint", "ee"] = "joint"    # joint: [J joints, gripper] per arm | ee: [xyz, 6D rot] per arm + grippers
    action_label: Literal["recorded", "executed"] = "executed"  # executed: the targets sent (with any DART drift)
    randomize_visuals: bool = False
    segments: Literal["takes", "annotations"] = "takes"   # annotations: one episode per labelled time range
    # Dataset frame rate. None = the recording's rate (scene physics.control_hz). Lower rates keep every n-th step
    # (the recording rate must be a multiple): images and state from that step, the action from the last step of
    # its window, i.e. the latest target the operator gave before the next frame.
    fps: int | None = None


def rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """(..., 4) quaternions -> (..., 6): the first two columns of the rotation matrix. Continuous and the same for
    q and -q (a gripper pointing down has w ~ 0, where quaternions flip sign from step to step)."""
    w, x, y, z = np.moveaxis(np.asarray(quat_wxyz, float), -1, 0)
    c1 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], -1)
    c2 = np.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], -1)
    return np.concatenate([c1, c2], -1)


def ee9(pose: np.ndarray) -> np.ndarray:
    """(..., 7) [xyz, qwxyz] -> (..., 9) [xyz, 6D rotation]."""
    return np.concatenate([pose[..., :3], rot6d(pose[..., 3:7])], -1)


def state_names(layout: FeatureLayout, meta: dict) -> list[str]:
    arms, names = meta["arms"], []
    J = (len(meta["action_names"]) // len(arms)) - 1
    for ch in layout.state:
        if ch == "joint_pos":
            names += [n for n in meta["action_names"] if not n.endswith("_gripper")]
        elif ch == "gripper_width":
            names += [f"{a}_gripper_width" for a in arms]
        elif ch == "joint_vel":
            names += [f"{a}_joint{i + 1}_vel" for a in arms for i in range(J)]
        elif ch == "ee_pose":
            names += [f"{a}_ee_{k}" for a in arms for k in ("x", "y", "z", "r11", "r21", "r31", "r12", "r22", "r32")]
        elif ch == "ee_wrench":
            names += [f"{a}_ee_{k}" for a in arms for k in ("fx", "fy", "fz", "tx", "ty", "tz")]
    return names


def action_names(layout: FeatureLayout, meta: dict) -> list[str]:
    if layout.action == "joint":
        return list(meta["action_names"])
    arms = meta["arms"]
    return [f"{a}_ee_{k}" for a in arms for k in ("x", "y", "z", "r11", "r21", "r31", "r12", "r22", "r32")] + \
           [f"{a}_gripper" for a in arms]


def env_state_names(meta: dict) -> list[str]:
    return [f"{o}_{k}" for o in meta["objects"] for k in ("x", "y", "z", "qw", "qx", "qy", "qz")] + \
           list(meta.get("fixture_joints", []))


def episode_arrays(env, meta: dict, data: dict, layout: FeatureLayout) -> dict[str, np.ndarray]:
    """state, action (and env_state) for every step of one episode, in the layout's channels."""
    n = len(data["time"])
    parts = []
    for ch in layout.state:
        if ch == "ee_pose":
            parts.append(ee9(data["obs/ee_pose"]).reshape(n, -1))
        elif ch == "ee_wrench":
            w = data["obs/ee_wrench"]
            parts.append((np.sign(w) * np.log1p(np.abs(w) / WRENCH_SCALE)).reshape(n, -1))
        else:
            parts.append(np.asarray(data[f"obs/{ch}"]).reshape(n, -1))
    out = {"state": np.concatenate(parts, 1) if parts else np.zeros((n, 0))}
    J = env.b.n_joints
    q, grip = split_action(data["action/joint_pos"], J)
    if layout.action_label == "executed":
        q = env.executed_targets(data["sim/state"])
    if layout.action == "joint":
        out["action"] = np.stack([join_action(q[t], grip[t], J) for t in range(n)])
    else:
        out["action"] = np.concatenate([ee9(env.tcp_poses_at(q)).reshape(n, -1), grip], 1)
    if layout.env_state:
        out["env_state"] = np.concatenate([data["obs/objects"].reshape(n, -1),
                                           data["obs/fixture_joints"].reshape(n, -1)], 1)
    return out


def constant_channels(columns: dict[str, np.ndarray], names: dict[str, list[str]]) -> dict[str, list[str]]:
    """Channels whose value never varies in the whole dataset, per feature."""
    out = {}
    for key, arr in columns.items():
        if arr.size == 0:
            continue
        flat = np.flatnonzero(arr.std(axis=0) < CONSTANT_STD)
        if len(flat):
            out[key] = [names[key][i] if i < len(names[key]) else str(i) for i in flat]
    return out


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def read_manifest(dataset_root: Path) -> dict:
    return json.loads((Path(dataset_root) / MANIFEST).read_text())


class MixedVersions(ValueError):
    pass


def run_export(episodes_dir: Path, out: Path, exporter: str = "hdf5", rules: CurateConfig | None = None,
               layout: FeatureLayout | None = None, allow_mixed: bool = False, progress=None) -> dict:
    """Export a dataset folder. Returns the manifest (also written to <out>/xrcollect_manifest.json)."""
    episodes_dir, out = Path(episodes_dir), Path(out)
    rules, layout = rules or CurateConfig(), layout or FeatureLayout()
    if exporter not in EXPORTERS:
        raise ValueError(f"unknown exporter {exporter!r}; known: {', '.join(EXPORTERS)}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already exists; pick another name or remove it first")
    audits = audit_dir(episodes_dir)
    curation = load_curation(episodes_dir)
    segments, rejected = select(episodes_dir, audits, rules, curation)
    if layout.segments == "annotations":
        segments = annotation_segments(episodes_dir, audits, rules, curation)
    if not segments:
        raise ValueError("no episodes selected: " + (json.dumps(rejected) if rejected else "the folder is empty"))
    versions = sorted({(audits[s.path.name].info or {}).get("task_version") or "?" for s in segments})
    if len(versions) > 1 and not allow_mixed:
        raise MixedVersions(f"the selected episodes come from {len(versions)} task versions ({', '.join(versions)}): "
                            "export them separately, or allow mixing explicitly")

    sink = load_object(EXPORTERS[exporter])()
    scenes: dict[str, tuple] = {}   # scene hash -> (env, renderer, randomizer)
    loaded: tuple | None = None
    written, columns, names, info0 = [], {}, {}, None
    for k, seg in enumerate(segments):
        if loaded is None or loaded[0] != seg.path:  # several segments per file come in order
            meta, data = read_episode(seg.path)
            loaded = (seg.path, meta, data, None)
        _, meta, data, arrays = loaded
        key = meta["scene_hash"]
        if key not in scenes:
            env = env_from_episode(meta)
            cams = env.camera_names if layout.cameras is None else list(layout.cameras)
            missing = [c for c in cams if c not in env.camera_names]
            if missing:
                raise ValueError(f"{seg.path.name}: the scene has no camera {missing} (it has {env.camera_names})")
            renderer = CameraRenderer(env.model, *layout.image_size, cams) if cams else None
            looks = AppearanceRandomizer(env.model, env.objects) if layout.randomize_visuals else None
            scenes[key] = (env, renderer, looks)
        env, renderer, looks = scenes[key]
        if data["sim/state"].shape[1] != env.state_size:
            raise RuntimeError(f"{seg.path.name}: recorded state size {data['sim/state'].shape[1]} != scene "
                               f"{env.state_size}")
        if arrays is None:
            arrays = episode_arrays(env, meta, data, layout)
            loaded = (seg.path, meta, data, arrays)
        rec_fps = int(meta["fps"])
        out_fps = layout.fps or rec_fps
        if out_fps > rec_fps or rec_fps % out_fps:
            raise ValueError(f"{seg.path.name}: recorded at {rec_fps} fps; an export can only keep every n-th step, "
                             f"so its fps must divide {rec_fps} (asked for {out_fps})")
        stride = rec_fps // out_fps
        steps = np.arange(seg.start, seg.stop, stride)                    # frames: images, state
        act_steps = np.minimum(steps + stride - 1, seg.stop - 1)          # latest action within each frame
        if info0 is None:
            names = {"state": state_names(layout, meta), "action": action_names(layout, meta),
                     "env_state": env_state_names(meta)}
            info0 = {"fps": out_fps, "layout": layout.model_dump(), "names": names,
                     "cameras": list(renderer.cameras) if renderer else [], "image_size": list(layout.image_size),
                     "robot_type": meta.get("task_yaml") and env.bundle.embodiment.get("name"), "task": meta["task"]}
            sink.begin(out, info0)
        elif out_fps != info0["fps"]:
            raise ValueError(f"{seg.path.name}: would be exported at {out_fps} fps, the dataset is {info0['fps']} fps")
        frames = {name: arr[act_steps if name == "action" else steps].astype(np.float32)
                  for name, arr in arrays.items()}
        if renderer is not None:
            if looks is not None:
                looks.apply(np.random.default_rng([k, int(meta.get("seed", 0)) % (2**31)]))
            images = {c: [] for c in renderer.cameras}
            for t in steps:
                env.set_state(data["sim/state"][t])
                for c, img in renderer.render(env.data).items():
                    images[c].append(img)
            frames["images"] = {c: np.stack(v) for c, v in images.items()}
        sink.add_episode(frames, {"instruction": seg.label or meta.get("instruction", ""), "file": seg.path.name,
                                  "start": seg.start, "stop": seg.stop, "success": bool(meta.get("success"))})
        for name in ("state", "action", "env_state"):
            if name in frames:
                columns.setdefault(name, []).append(frames[name])
        written.append({"file": seg.path.name, "start": seg.start, "stop": seg.stop, "label": seg.label,
                        "scene_hash": key, "task_version": meta.get("task_version"), "frames": len(steps)})
        if progress:
            progress(k + 1, len(segments))
    extra = sink.finish() or {}
    for _, renderer, looks in scenes.values():
        if looks is not None:
            looks.restore()
        if renderer is not None:
            renderer.close()
    constant = constant_channels({k: np.concatenate(v) for k, v in columns.items()}, names)
    if constant:
        log.warning("channels that never vary in this dataset (don't feed them to a policy as they are): %s", constant)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
        "exporter": exporter, "source_dir": str(episodes_dir), "task": info0["task"], "task_versions": versions,
        "fps": info0["fps"], "layout": layout.model_dump(), "names": names, "cameras": info0["cameras"],
        "rules": rules.model_dump(), "episodes": len(written), "steps": sum(s["frames"] for s in written),
        "segments": written, "rejected": rejected, "constant_channels": constant, **extra,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1))
    return manifest
