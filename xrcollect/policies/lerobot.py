"""A LeRobot checkpoint as a PolicyRunner (needs `uv sync --extra lerobot`). All LeRobot API used at run time lives
in this file, like `export/lerobot.py` holds the export side.

    uv run xrc serve --policy xrcollect.policies.lerobot:LeRobotRunner \\
        --policy-arg checkpoint=outputs/train/act_cups_v2_25fps --control shared
    uv run xrc rollout xrcollect.policies.lerobot:LeRobotRunner \\
        --policy-arg checkpoint=outputs/train/act_cups_v2_25fps/checkpoints/040000 --n 20

`checkpoint` is a training run (uses checkpoints/last), one checkpoint folder, or its `pretrained_model`. The input
is built like the export the policy was trained on (its manifest, found through the checkpoint's train_config):
the same state channels (`export.base.observation_arrays`), the same cameras rendered the same way
(`CameraRenderer`), at the dataset's frame rate. A 25 fps dataset in a 50 Hz scene asks the policy every second
control step and holds its action in between, the way the export paired frames with actions. Works for any
LeRobot policy type that `select_action` drives (ACT, Diffusion, ...), not only ACT.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from xrcollect.config import rooted
from xrcollect.embodiment.base import split_action
from xrcollect.export.base import FeatureLayout, observation_arrays, read_manifest

log = logging.getLogger(__name__)


def model_dir(checkpoint: str | Path) -> Path:
    """The `pretrained_model` folder of a run, a checkpoint, or the folder itself."""
    p = rooted(str(checkpoint))
    for cand in (p, p / "pretrained_model", p / "checkpoints" / "last" / "pretrained_model"):
        if (cand / "config.json").exists() and (cand / "model.safetensors").exists():
            return cand
    raise FileNotFoundError(f"no LeRobot checkpoint (config.json + model.safetensors) at {p}, {p}/pretrained_model "
                            f"or {p}/checkpoints/last/pretrained_model")


def dataset_root(model: Path, dataset: str | None = None) -> Path:
    if dataset is not None:
        return rooted(dataset)
    train_cfg = model / "train_config.json"
    root = json.loads(train_cfg.read_text())["dataset"].get("root") if train_cfg.exists() else None
    if not root:
        raise FileNotFoundError(f"{model} names no local dataset; pass dataset=<export folder>")
    return rooted(root)


class LeRobotRunner:
    def __init__(self, env_info: dict, checkpoint: str, dataset: str | None = None, device: str | None = None,
                 n_action_steps: int | None = None, **_):
        """checkpoint: see the module docstring. dataset: the export it was trained on (default: from its
        train_config). device: cuda when available. n_action_steps: actions executed per prediction (ACT: up
        to chunk_size; fewer re-plans more often)."""
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies import make_pre_post_processors
            from lerobot.policies.factory import get_policy_class
        except ImportError as e:
            raise RuntimeError("the LeRobot runner needs the lerobot package: uv sync --extra lerobot") from e

        self.torch = torch
        model = model_dir(checkpoint)
        manifest = read_manifest(dataset_root(model, dataset))
        self.fl = FeatureLayout(**manifest["layout"])
        if self.fl.action != "joint":
            raise ValueError(f"{model}: trained on {self.fl.action!r} actions; this runner drives joint targets only")
        if env_info["task_version"] not in manifest.get("task_versions", [env_info["task_version"]]):
            log.warning("%s was trained on task versions %s; the scene is %s (edited since?)", model,
                        manifest["task_versions"], env_info["task_version"])
        self.cameras = list(manifest["cameras"])
        self.fps = int(manifest["fps"])
        hz = int(env_info["control_hz"])
        if hz % self.fps:
            raise ValueError(f"the policy runs at {self.fps} fps, which does not divide the scene's {hz} Hz")
        self.stride = hz // self.fps
        self.layout = {"action_names": manifest["names"]["action"], "cameras": self.cameras}
        ckpt = model.parent if model.name == "pretrained_model" else model   # <run>/checkpoints/<step>
        self.name = f"{ckpt.parent.parent.name}@{ckpt.name}" if ckpt.parent.name == "checkpoints" else ckpt.name
        self.n_joints = len(manifest["names"]["action"]) // len(env_info["arms"]) - 1

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = PreTrainedConfig.from_pretrained(model)
        cfg.device = device
        if n_action_steps is not None:
            cfg.n_action_steps = int(n_action_steps)
        self.policy = get_policy_class(cfg.type).from_pretrained(model, config=cfg)
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=str(model), preprocessor_overrides={"device_processor": {"device": device}})
        self.instruction = ""
        self.renderer = self.renderer_model = None
        self.t = 0
        self.action: np.ndarray | None = None

    def reset(self, env) -> None:
        if self.renderer is None or self.renderer_model is not env.model:  # the runtime may switch scenes
            from xrcollect.scene.render import CameraRenderer
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = CameraRenderer(env.model, *self.fl.image_size, self.cameras) if self.cameras else None
            self.renderer_model = env.model
        self.policy.reset()
        self.pre.reset()
        self.post.reset()
        self.instruction = env.task.instruction
        self.t, self.action = 0, None

    def act(self, env) -> tuple[np.ndarray, np.ndarray]:
        if self.action is None or self.t % self.stride == 0:
            self.action = self.predict(env)
        self.t += 1
        q, grip = split_action(self.action, self.n_joints)
        return q, np.clip(grip, 0.0, 1.0)

    def predict(self, env) -> np.ndarray:
        torch = self.torch
        obs = {k: np.asarray(v)[None] for k, v in env.observe().items()}
        arrays = observation_arrays(obs, self.fl, 1)
        batch = {"observation.state": torch.from_numpy(arrays["state"][0].astype(np.float32)),
                 "task": self.instruction}
        if "env_state" in arrays:
            batch["observation.environment_state"] = torch.from_numpy(arrays["env_state"][0].astype(np.float32))
        if self.renderer is not None:
            for cam, img in self.renderer.render(env.data).items():   # HWC uint8 -> CHW float 0..1, as training
                batch[f"observation.images.{cam}"] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(batch)))
        return action.squeeze(0).float().cpu().numpy().astype(np.float64)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
