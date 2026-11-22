"""Real-time sim thread: input -> teleop -> IK (or policy) -> safety filter -> physics -> recording -> publish.

One thread owns all simulation state. The network layer talks to it only through thread-safe methods
(submit_input, command, latest_pose_frame, latest_ghost_frame, status, previews, scene, task) and event listeners;
every action from the headset, the dashboard and the controller buttons is a command processed at the start of a
tick.

Control tick:
  1 commands  2 teleop  3 driver (operator IK or policy)  4 safety filter  5 recorder tick
  6 snapshot before the action  7 apply, success/progress, recorded step, physics  8 append
  9 task feedback  10 warnings  11 publish (poses every tick, status ~10 Hz, previews while someone looks)
Replay and edit mode skip 2-10.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from xrcollect import editor
from xrcollect.audit import audit_arrays, kept_spans, update_curation, verdict
from xrcollect.config import Config
from xrcollect.env import CONTACT_KINDS, Env, env_from_episode
from xrcollect.frames import mat_to_quat_wxyz, quat_wxyz_to_mat
from xrcollect.ik import ArmIK
from xrcollect.policy import load_policy
from xrcollect.protocol import InputFrame, pack_pose_frame, pose_flags
from xrcollect.recorder import (SOURCE_OPERATOR, SOURCE_POLICY, Recorder, RecState, control_step, read_episode,
                                snapshot, teleop_fields)
from xrcollect.safety import SafetyFilter
from xrcollect.scene.constants import STATE_SPEC, quat_yaw, yaw_quat
from xrcollect.scene.export import body_poses, export_scene
from xrcollect.scene.render import CameraRenderer
from xrcollect.task.loader import load_bundle
from xrcollect.teleop import Teleop, command_table

log = logging.getLogger(__name__)

STATUS_EVERY = 5          # ticks (10 Hz at 50 Hz control)
PREVIEW_EVERY = 6         # ticks (~8 Hz)
GHOST_EVERY = 2
PREVIEW_SIZE = (224, 224)
DART_TAU_S = 0.5          # DART drift correlation time: slow drifts the operator sees and corrects
DART_SMOOTH_S = 0.2       # ... low-passed once more, so the arm drifts instead of trembling
GRIP_CATCH = 0.15         # a latched gripper follows its Trigger again once the Trigger is this close to it
CONTROL_MODES = ("teleop", "policy", "shared")  # shared: the policy drives until the operator takes control
REPLAY_SPEED_RANGE = (0.1, 8.0)
# Task event -> cue, unless the task's `events` table says otherwise.
DEFAULT_EVENT_CUES = {"grasp": "grasp", "release": "release", "snap_on": "click", "snap_off": "unclick",
                      "stage_reached": "placed", "stage_lost": "lost", "success": "complete"}


def _ms() -> float:
    return time.monotonic() * 1000.0


class Runtime:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._listeners: list[Callable[[dict], None]] = []
        self._input: InputFrame | None = None
        self._input_lock = threading.Lock()
        self._input_rate = 0.0
        self._input_dt = 0.0
        self._last_input_t: float | None = None
        self._rtt_ms = 0.0
        self._cmds: queue.Queue = queue.Queue()
        self._out_lock = threading.Lock()
        self._pose_frame = b""
        self._pose_seq = 0
        self._ghost_frame = b""
        self._ghost_seq = 0
        self._status: dict = {}
        self._previews: dict[str, bytes] = {}
        self._previews_wanted_until = 0.0
        self._preview_renderer: CameraRenderer | None = None
        self._preview_model = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tick_count = 0
        self._scene_version = 0
        self._noise = np.random.default_rng()
        self.control_mode = "teleop"
        self.policy = None
        self._policy_ref: str | None = None
        self._policy_lock = threading.Lock()
        self._policy_loading: str | None = None
        self._policy_ready: tuple | None = None   # (runner | None, ref, error)
        self._policy_error: str | None = None
        self._replay: dict | None = None
        self._replay_speed, self._replay_loop = 1.0, False
        self._live_env: Env | None = None          # the task's scene while another episode's scene is shown
        self._ghost: dict | None = None
        self._edit: dict | None = None
        self._placement = None
        self.recorder: Recorder | None = None
        self._build(load_bundle(cfg.task))
        self._loop_hz = float(self.env.control_hz)
        if cfg.control.policy:
            try:
                self.policy = load_policy(cfg.control.policy, self.env, **cfg.control.policy_args)
                self._policy_ref = cfg.control.policy
                self.control_mode = cfg.control.mode
            except Exception:  # noqa: BLE001
                log.exception("could not load policy %s; using teleop", cfg.control.policy)
        elif cfg.control.mode != "teleop":
            log.warning("control.mode %s needs a policy (xrc serve --policy module:Class); using teleop",
                        cfg.control.mode)

    # -- (re)building the world --------------------------------------------------------------------
    def _build(self, bundle, env: Env | None = None) -> None:
        """Everything that depends on the scene: env, IK, safety filter, teleop, recorder, exported scene."""
        self.bundle = bundle
        self.env = env or Env(bundle)
        self.dt = self.env.control_dt
        self.arms = list(self.env.arms)
        emb = bundle.embodiment
        self.ik = ArmIK(bundle, self.env.adapter, self.cfg.ik, self.dt)
        self.safety = SafetyFilter(bundle, self.env.adapter, self.cfg.ik, self.cfg.safety, self.dt)
        self.teleop = Teleop(self.cfg.teleop, self.arms, emb.get("workspace"), float(emb.get("operator_x", -0.5)))
        if self._placement is not None:  # keep where the operator put the table in the room
            self.teleop.placement = self._placement
        self._placement = self.placement = self.teleop.placement
        if self.recorder is None:
            self.recorder = Recorder(self.cfg, self.cfg.out_dir, self.env.control_hz)
        self.recorder.fps = self.env.control_hz
        self._export_scene()
        n = len(self.env.b.qadr)
        self._dart = np.zeros(n)       # current DART offset on the executed joint targets (rad)
        self._dart_ou = np.zeros(n)    # ... before smoothing
        # The smoothing shrinks the drift; scaled so its std is still record.action_noise.
        self._dart_gain = self.cfg.record.action_noise * np.sqrt(1 + DART_SMOOTH_S / DART_TAU_S)
        self._event_cues = self._cue_table(bundle.task.get("events", []))
        self._warn = {"arm_collision": False, "table_collision": False, "ik_saturated": False, "blocked": False}
        self._reach = {a: False for a in self.arms}
        self._engaged = {a: False for a in self.arms}
        self._hand_of: dict[str, str | None] = {a: None for a in self.arms}
        self._policy_grip = np.ones(len(self.arms))
        self._grip_latch: list[float | None] = [None] * len(self.arms)
        self._warn_sound_at: dict[tuple[str, str | None], float] = {}
        self._hold_tick_at = {a: 0.0 for a in self.arms}
        self._quiet_until = {a: 0.0 for a in self.arms}
        self._review: tuple[int, dict] | None = None
        self._driver = "operator"
        self._safety_rad = 0.0
        self._policy_steps = 0
        self._branch: dict | None = None
        self._success = False
        self._reset_scene()

    def _export_scene(self) -> None:
        js, blob = export_scene(self.env.model)
        self._scene_version += 1
        js["version"] = self._scene_version
        js["arms"] = self.arms
        js["objects"] = {n: int(self.env.model.body(n).id) for n in self.env.world.body}
        js["tcp_sites"] = {a: self.env.adapter.tcp_site(a) for a in self.arms}
        js["wrist_cameras"] = self.env.adapter.wrist_cameras()
        with self._out_lock:
            self._scene = (js, blob)
            self._task_spec = self.env.task.goal_spec()

    @staticmethod
    def _cue_table(events: list[dict]) -> list[dict]:
        listed = {e["on"] for e in events}
        return [*events, *({"on": k, "cue": v} for k, v in DEFAULT_EVENT_CUES.items() if k not in listed)]

    def _reload(self, task: str | None = None) -> bool:
        """Hot reload: rebuild the scene from the task files, tell the clients, reset. Refused while a take is
        being recorded (an episode has one model)."""
        if self.recorder.state != RecState.IDLE:
            self._toast("Finish or discard the take before changing the scene", "warn")
            return False
        try:
            bundle = load_bundle(task or self.bundle.path or self.cfg.task)
            env = Env(bundle)
        except Exception as e:  # noqa: BLE001 - shown to whoever edited the task
            log.exception("scene reload failed")
            self._toast(f"Scene not reloaded: {type(e).__name__}: {e}", "warn")
            self._emit({"type": "edit_error", "text": f"{type(e).__name__}: {e}"})
            return False
        self._end_replay(restore=False)
        self._ghost = None
        edit = self._edit is not None
        if task:
            self.cfg.task = task
        self._drop_policy_if_mismatch(env)
        self._build(bundle, env)
        self._edit = {"moved": {}} if edit else None
        self._emit({"type": "scene_changed", "version": self._scene_version, "task": bundle.name,
                    "task_version": bundle.version})
        return True

    def _drop_policy_if_mismatch(self, env: Env) -> None:
        if self.policy is not None and env.adapter.action_layout() != self.env.adapter.action_layout():
            self._drop_policy()
            self.control_mode = "teleop"

    # -- policy ------------------------------------------------------------------------------------
    def _load_policy_async(self, ref: str, args: dict) -> None:
        """`load_policy` command: load in the background (imports and weights take seconds); `_swap_policy` puts
        it in place on a later tick, so the control loop never waits."""
        with self._policy_lock:
            if self._policy_loading:
                self._toast("A policy is still loading", "warn")
                return
            self._policy_loading = ref
        self._policy_error = None
        env = self.env

        def work():
            try:
                result = (load_policy(ref, env, **args), ref, None)
            except Exception as e:  # noqa: BLE001 - reported to the dashboard
                log.exception("loading policy %s failed", ref)
                result = (None, ref, f"{type(e).__name__}: {e}")
            with self._policy_lock:
                self._policy_ready = result

        threading.Thread(target=work, name="policy-load", daemon=True).start()

    def _swap_policy(self) -> None:
        with self._policy_lock:
            ready, self._policy_ready = self._policy_ready, None
            if ready is not None:
                self._policy_loading = None
        if ready is None:
            return
        runner, ref, error = ready
        if runner is None:
            self._policy_error = error
            self._toast(f"Couldn't load the policy: {error}", "warn")
            return
        self._drop_policy()
        self.policy, self._policy_ref = runner, ref
        self._driver = "operator"  # its first tick as driver starts fresh
        self._toast(f"Policy {runner.name} loaded", "success")

    def _drop_policy(self) -> None:
        old, self.policy, self._policy_ref = self.policy, None, None
        if old is not None:
            try:
                old.close()
            except Exception:  # noqa: BLE001
                log.exception("closing the policy failed")

    # -- public, thread-safe ---------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="sim", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.recorder.close()

    def add_listener(self, fn: Callable[[dict], None]) -> None:
        self._listeners.append(fn)

    def submit_input(self, frame: InputFrame) -> None:
        now = time.monotonic()
        with self._input_lock:
            if self._last_input_t is not None:
                # Average the intervals, not the instantaneous rates: a burst after a stall must not read as 400 Hz.
                dt = min(now - self._last_input_t, 1.0)
                self._input_dt = 0.9 * self._input_dt + 0.1 * dt if self._input_dt else dt
                self._input_rate = 1.0 / max(self._input_dt, 1e-3)
            self._last_input_t = now
            frame.recv_time = now
            if frame.t_echo > 0:
                self._rtt_ms = 0.8 * self._rtt_ms + 0.2 * (_ms() - frame.t_echo)
            self._input = frame

    def command(self, name: str, **kwargs) -> None:
        self._cmds.put((name, kwargs))

    def latest_pose_frame(self) -> tuple[int, bytes]:
        with self._out_lock:
            return self._pose_seq, self._pose_frame

    def latest_ghost_frame(self) -> tuple[int, bytes]:
        with self._out_lock:
            return self._ghost_seq, self._ghost_frame

    def status(self) -> dict:
        with self._out_lock:
            return dict(self._status)

    def previews(self) -> dict[str, bytes]:
        self._previews_wanted_until = time.monotonic() + 2.0
        with self._out_lock:
            return dict(self._previews)

    def scene(self) -> tuple[dict, bytes]:
        with self._out_lock:
            return self._scene

    def task(self) -> dict:
        with self._out_lock:
            return self._task_spec

    def commands(self) -> list[dict]:
        return command_table()

    # -- events ------------------------------------------------------------------------------------
    def _emit(self, event: dict) -> None:
        for fn in self._listeners:
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - a broken listener must not kill the sim loop
                log.exception("listener failed")

    def _toast(self, text: str, kind: str = "info") -> None:
        self._emit({"type": "toast", "text": text, "kind": kind})

    def _cue(self, name: str | None, arm: str | None = None, now: float | None = None) -> None:
        """Play a feedback cue (config feedback.cues): its haptic pattern on the arm's controller (every
        controller if arm is None) and its sound, panned to that side."""
        cue = self.cfg.feedback.cues.get(name or "")
        if cue is None:
            return
        if cue.haptic:
            intensity, ms, repeat, gap = (*cue.haptic, 1, 0)[:4]
            for a in self.arms if arm is None else (arm,):
                self._emit({"type": "haptic", "arm": a, "hand": self._hand_of.get(a), "intensity": float(intensity),
                            "ms": int(ms), "repeat": int(repeat), "gap_ms": int(gap)})
                if now is not None:
                    self._quiet_until[a] = now + self.cfg.feedback.quiet_s
        if cue.sound:
            self._emit({"type": "sound", "name": cue.sound, "arm": arm,
                        "hand": self._hand_of.get(arm) if arm else None})

    def _warn_cue(self, name: str, arm: str | None, now: float) -> None:
        if now >= self._warn_sound_at.get((name, arm), 0.0):
            self._cue(name, arm)
            self._warn_sound_at[(name, arm)] = now + self.cfg.feedback.warn_sound_gap_s

    # -- scene state ---------------------------------------------------------------------------------
    def _ik_pose(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        p, q = self.ik.tcp_pose(arm)
        return p, quat_wxyz_to_mat(q)

    def _reset_scene(self, seed: int | None = None) -> None:
        obs = self.env.reset(seed)
        self.ik.reset(obs["joint_pos"])
        self.safety.reset(obs["joint_pos"])
        self.teleop.reset({a: self._ik_pose(a) for a in self.arms})
        self._after_state_change()
        self._policy_steps = 0
        self._branch = None
        self._grip_latch = [None] * len(self.arms)

    def _sync_to_scene(self) -> None:
        """The scene was set from outside (a replayed frame, an edit): IK, safety filter and the held operator
        targets follow it."""
        q = self.env.arm_q
        self.ik.reset(q)
        self.safety.reset(q)
        self.teleop.reset({a: self._ik_pose(a) for a in self.arms})
        self._after_state_change()

    def _after_state_change(self) -> None:
        self._held = self.env.held()
        self._last_holder: dict[str, str] = {}
        self._progress = self.env.progress()
        self._complete = self.env.success()
        self._driver = "operator"
        self._dart[:], self._dart_ou[:] = 0.0, 0.0

    # -- main loop -----------------------------------------------------------------------------------
    def _run(self) -> None:
        next_t = time.perf_counter()
        window_start, window_ticks = next_t, 0
        while not self._stop.is_set():
            now_pc = time.perf_counter()
            if now_pc < next_t:
                time.sleep(min(next_t - now_pc, 0.005))
                continue
            window_ticks += 1
            if now_pc - window_start >= 1.0:
                self._loop_hz = window_ticks / (now_pc - window_start)
                window_start, window_ticks = now_pc, 0
            try:
                self._tick(time.monotonic())
            except Exception:  # noqa: BLE001
                log.exception("sim tick failed")
            next_t += self.dt
            if time.perf_counter() - next_t > 0.25:  # fell far behind (a reload, a debugger): resync
                next_t = time.perf_counter()

    def _process_commands(self, now: float) -> None:
        while True:
            try:
                name, kw = self._cmds.get_nowait()
            except queue.Empty:
                return
            if name != "edit_move":
                log.info("command %s %s", name, {k: v for k, v in kw.items() if k != "ops"})
            try:
                self._handle(name, kw, now)
            except Exception as e:  # noqa: BLE001 - a bad command must not kill the loop
                log.exception("command %s failed", name)
                self._toast(f"{name} failed: {type(e).__name__}: {e}", "warn")

    def _handle(self, name: str, kw: dict, now: float) -> None:
        rec = self.recorder
        if name == "record_toggle":
            if self._edit is not None:
                self._toast("Leave edit mode before recording", "warn")
                return
            if self._replay is not None and rec.state == RecState.IDLE:
                self._end_replay()
            was = rec.state
            rec.toggle(now)
            if was == RecState.IDLE and rec.state == RecState.COUNTDOWN:
                self._toast("Get ready…")
        elif name == "save":
            self._save(kw)
        elif name == "discard":
            if rec.discard():
                self._toast("Take discarded", "warn")
                self._reset_scene()
        elif name == "reset":
            rec.discard()
            self._end_replay()
            self._reset_scene(kw.get("seed"))
        elif name == "set_mode":
            self.teleop.set_mode(kw.get("mode", "controllers"))
        elif name == "set_control":
            self.teleop.set_control(kw.get("control", "clutched"))
        elif name == "set_scale":
            self.teleop.set_scale(float(kw.get("scale", 1.0)))
        elif name == "swap_hands":
            self.teleop.swap_hands()
            self._toast("Hands swapped: " + ", ".join(f"{h.upper()} hand → {a}" for h, a in self.teleop.mapping.items()))
        elif name == "set_mapping":
            self.teleop.set_mapping(kw.get("mapping"))
        elif name == "recenter":
            with self._input_lock:
                frame = self._input
            if frame is not None:
                self.placement.recenter(frame.head[:3], frame.head[3:], height=kw.get("height"))
                self.teleop.release_all()
        elif name == "set_table_offset":   # operator comfort: raise/lower the scene in the room (seated use)
            self.placement.y_offset = float(np.clip(kw.get("offset", 0.0), -0.6, 0.4))
            self.placement.t[1] = self.placement.y_offset
            self.teleop.release_all()
        elif name == "set_operator":
            self.cfg.record.operator = str(kw.get("operator", "operator"))[:64]
        elif name == "set_session":
            self.cfg.record.session = str(kw.get("session", ""))[:64]
            if "target" in kw:
                self.cfg.record.target_takes = max(0, int(kw["target"] or 0))
        elif name == "set_auto_stop":
            rec.auto_stop = bool(kw.get("enabled", True))
        elif name == "replay":
            self._start_replay(kw)
        elif name == "replay_pause":
            if self._replay is not None:
                r = self._replay
                r["paused"] = bool(kw.get("paused", not r["paused"]))
                if not r["paused"] and r["i"] >= len(r["states"]) - 1:
                    r["i"] = 0  # play again from the start
        elif name == "replay_options":
            if "speed" in kw:
                self._replay_speed = float(np.clip(kw["speed"], *REPLAY_SPEED_RANGE))
            if "loop" in kw:
                self._replay_loop = bool(kw["loop"])
        elif name == "replay_seek":
            if self._replay is not None:
                self._replay["i"] = int(np.clip(kw.get("i", 0), 0, len(self._replay["states"]) - 1))
        elif name == "replay_step":
            if self._replay is not None:
                r = self._replay
                r["paused"] = True
                r["i"] = int(np.clip(r["i"] + int(kw.get("delta", 1)), 0, len(r["states"]) - 1))
        elif name == "replay_branch":
            self._start_branch(now)
        elif name == "replay_stop":
            self._end_replay()
            self._reset_scene()
        elif name == "ghost":
            self._set_ghost(kw.get("path"))
        elif name == "load_policy":
            self._load_policy_async(str(kw["ref"]), dict(kw.get("args") or {}))
        elif name == "unload_policy":
            self.control_mode = "teleop"
            self.teleop.release_all()
            self._drop_policy()
        elif name == "set_control_mode":
            mode = kw.get("mode", "teleop")
            if mode != "teleop" and self.policy is None:
                self._toast("No policy loaded", "warn")
            elif mode in CONTROL_MODES and mode != self.control_mode:
                self.control_mode = mode
                self.teleop.release_all()
                self._toast({"teleop": "You drive", "policy": "Policy drives",
                             "shared": "Policy drives: take over any time with Grip"}[mode], "info")
        elif name == "reload_scene":
            if self._reload(kw.get("task")):
                self._toast(f"Scene reloaded: {self.bundle.name}", "success")
        elif name == "edit_mode":
            self._set_edit(bool(kw.get("enabled", self._edit is None)))
        elif name == "edit_move":
            self._edit_move(kw)
        elif name == "edit_commit":
            self._edit_commit()
        elif name == "edit":
            self._apply_edits(list(kw.get("ops") or []))
        elif name == "capture_goal":
            self._capture_goal(kw)
        else:
            log.warning("unknown command %s", name)

    def _save(self, kw: dict) -> None:
        rec = self.recorder
        extra = None if self._branch is None else {"branched_from": self._branch["of"],
                                                   "branch_step": self._branch["step"]}
        path = rec.save(self.env, self.teleop.mode, kw.get("success"), source=self.control_mode, extra=extra)
        if not path:
            return
        # Labels given on the review card go to the sidecar, never into the episode file.
        labels = {k: kw[k] for k in ("tags", "quality", "note") if kw.get(k)}
        if labels:
            update_curation(Path(path).parent, Path(path).name, **labels)
        self._emit({"type": "saved", "path": path})
        self._toast(f"Saved {Path(path).name}", "success")
        self._cue("saved")
        self._reset_scene()

    # -- replay, branch, ghost -----------------------------------------------------------------------
    def _start_replay(self, kw: dict) -> None:
        if self.recorder.state != RecState.IDLE or self._edit is not None:
            return
        meta, data = read_episode(kw["path"], keys=["sim/state"])
        states = data["sim/state"]
        self._end_replay()
        if meta.get("scene_hash") != self.env.scene_hash:
            # Recorded in another scene (another task or an older version of this one): show it in its own.
            self._live_env = self.env
            self.env = env_from_episode(meta)
            self._export_scene()
            self._emit({"type": "scene_changed", "version": self._scene_version, "task": meta.get("task"),
                        "task_version": meta.get("task_version")})
        self._branch = None
        self._replay = {"path": kw["path"], "states": states, "paused": bool(kw.get("paused", False)), "meta": meta,
                        "i": int(np.clip(kw.get("i", 0), 0, len(states) - 1)), "acc": 0.0}

    def _end_replay(self, restore: bool = True) -> None:
        self._replay = None
        if self._live_env is not None:
            env, self._live_env = self._live_env, None
            if restore:
                self.env = env
                self._export_scene()
                self._emit({"type": "scene_changed", "version": self._scene_version, "task": self.bundle.name,
                            "task_version": self.bundle.version})
                self._reset_scene()

    def _start_branch(self, now: float) -> None:
        """Take over from the replayed frame: the scene stays at that state through the countdown, and the take
        records the operator's correction from exactly there (recovery data without a policy)."""
        r, rec = self._replay, self.recorder
        if r is None or rec.state != RecState.IDLE:
            return
        if self._live_env is not None:
            self._toast("This episode was recorded in a different scene: it can be replayed, not continued", "warn")
            return
        state = np.array(r["states"][r["i"]])
        self._replay = None
        self.env.set_state(state)
        self._sync_to_scene()
        # Each gripper keeps its command at that frame (a held object stays held) until its Trigger matches it.
        self._grip_latch = [float(g) for g in self.env.adapter.gripper_command(self.env.data, self.env.b)]
        self._branch = {"of": r["path"], "step": int(r["i"]), "state": state, "frozen": True}
        rec.start(now)
        self._toast(f"Taking over at {r['i'] / self.env.control_hz:.1f} s: get ready…")

    def _hold_branch(self, now: float) -> bool:
        """Countdown of a take-over: hold the scene at the branch state (True = this tick is done)."""
        b = self._branch
        if b is None or not b["frozen"]:
            return False
        if self.recorder.state == RecState.COUNTDOWN and now < self.recorder.countdown_end:
            self.env.set_state(b["state"])
            with self._input_lock:
                frame = self._input
            out = self.teleop.update(frame, now, self.dt)  # only for the buttons (cancel); the arms stay put
            for cmd in out.commands:
                self._handle(cmd, {}, now)
            if self._branch is not None and self._branch["frozen"]:
                self.teleop.reset({a: self._ik_pose(a) for a in self.arms})
            self._publish(now, None)
            return True
        b["frozen"] = False
        if self.recorder.state == RecState.IDLE:  # countdown cancelled: a live scene at that state, no take
            self._branch = None
        self.env.set_state(b["state"])
        self._sync_to_scene()
        return False

    def _set_ghost(self, path: str | None) -> None:
        """A translucent replay of a reference take, drawn for the operator only (never in a camera)."""
        if not path:
            self._ghost = None
            self._emit({"type": "ghost", "on": False})
            return
        meta, data = read_episode(path, keys=["sim/state"])
        if meta.get("scene_hash") != self.env.scene_hash:
            self._toast("The reference take was recorded in a different scene", "warn")
            return
        self._ghost = {"path": path, "states": data["sim/state"], "i": 0, "data": mujoco.MjData(self.env.model)}
        self._emit({"type": "ghost", "on": True, "path": path})

    def _publish_ghost(self) -> None:
        g = self._ghost
        if g is None or self._tick_count % GHOST_EVERY or self._replay is not None:
            return
        n = len(g["states"])
        if self.recorder.recording:
            g["i"] = min(len(self.recorder.steps), n - 1)   # in step with the take
        else:
            g["i"] = (g["i"] + GHOST_EVERY) % n             # looping preview
        d = g["data"]
        mujoco.mj_setState(self.env.model, d, np.asarray(g["states"][g["i"]][:self.env.mj_state_size], float),
                           STATE_SPEC)
        mujoco.mj_kinematics(self.env.model, d)
        frame = pack_pose_frame(self._ghost_seq + 1, _ms(), body_poses(d), 0, ghost=True)
        with self._out_lock:
            self._ghost_seq += 1
            self._ghost_frame = frame

    # -- edit mode -----------------------------------------------------------------------------------
    def _set_edit(self, on: bool) -> None:
        if on and self.recorder.state != RecState.IDLE:
            self._toast("Finish or discard the take before editing the scene", "warn")
            return
        if on and self.bundle.path is None:
            self._toast("This task has no file to edit", "warn")
            return
        self._end_replay()
        if on and self._edit is None:
            self._edit = {"moved": {}}
            self._reset_scene(0)  # physics paused at a fixed layout
            self._place_at_defaults()
            self._toast("Edit mode: physics paused", "info")
        elif not on and self._edit is not None:
            self._edit = None
            self._reset_scene()
            self._toast("Edit mode off", "info")

    def _place_at_defaults(self) -> None:
        """Edit mode shows each object at the centre of its start region (what an edit moves)."""
        env, t = self.env, self.bundle.task
        for name, sp in t.get("spawn", {}).items():
            if name in env.world.qadr:
                r = t["regions"][sp["region"]] if isinstance(sp["region"], str) else sp["region"]
                a = env.world.qadr[name]
                env.data.qpos[a:a + 3] = [*r["center"], env.table_height if sp.get("z") is None else sp["z"]]
                env.data.qpos[a + 3:a + 7] = yaw_quat(0.0)
        mujoco.mj_forward(env.model, env.data)

    def _edit_move(self, kw: dict) -> None:
        """Grab-and-place: put a free object at a pose (physics is paused). Remembered until edit_commit."""
        if self._edit is None:
            return
        env, name = self.env, kw["object"]
        if name not in env.world.qadr:
            return
        pos = [float(v) for v in kw["pos"]]
        z = pos[2] if len(pos) > 2 and kw.get("free_z") else env.table_height
        yaw = float(kw.get("yaw", 0.0))
        a = env.world.qadr[name]
        env.data.qpos[a:a + 3] = [pos[0], pos[1], z]
        env.data.qpos[a + 3:a + 7] = yaw_quat(yaw)
        mujoco.mj_forward(env.model, env.data)
        self._edit["moved"][name] = {"pos": [round(pos[0], 4), round(pos[1], 4)], "yaw_deg": round(np.rad2deg(yaw), 1),
                                     "z": z}

    def _edit_commit(self) -> None:
        """Write the moved objects back to the task file (default pose; a start region follows) and reload."""
        if self._edit is None or not self._edit["moved"]:
            return
        ops = [{"op": "object_pose", "object": n, "pos": m["pos"], "yaw_deg": m["yaw_deg"]}
               for n, m in self._edit["moved"].items()]
        self._apply_edits(ops)

    def _apply_edits(self, ops: list[dict]) -> None:
        if not ops:
            return
        if self.recorder.state != RecState.IDLE:
            self._toast("Finish or discard the take before changing the scene", "warn")
            return
        path = self.bundle.path
        if path is None:
            self._toast("This task has no file to edit", "warn")
            return
        try:
            before = editor.apply_edits(path, ops)
        except Exception as e:  # noqa: BLE001 - an invalid edit: nothing was written
            self._toast(f"Edit refused: {e}", "warn")
            self._emit({"type": "edit_error", "text": str(e)})
            return
        if not self._reload():
            Path(path).write_text(before)  # the scene didn't build: put the file back
            return
        if self._edit is not None:
            self._edit["moved"] = {}
            self._place_at_defaults()
        self._toast(f"Task updated (version {self.bundle.version[:8]})", "success")

    def _capture_goal(self, kw: dict) -> None:
        """Store the objects' current arrangement as the goal: relative poses to the anchor become stages with
        editable tolerances, plus the goal view."""
        env = self.env
        free = env.objects
        if len(free) < 2:
            self._toast("Capture goal needs at least two movable objects", "warn")
            return
        # Default anchor: the lowest object (ties: the first), i.e. what the others rest on.
        z = {n: float(env.data.xpos[env.world.body[n]][2]) for n in free}
        anchor = kw.get("anchor") or min(free, key=lambda n: (round(z[n], 3), free.index(n)))
        ba = env.world.body[anchor]
        Ra, pa, yaw_a = env.data.xmat[ba].reshape(3, 3), env.data.xpos[ba], quat_yaw(env.data.xquat[ba])
        poses = {}
        for n in sorted((n for n in free if n != anchor), key=lambda n: z[n]):
            b = env.world.body[n]
            rel = Ra.T @ (env.data.xpos[b] - pa)
            if np.linalg.norm(rel) > float(kw.get("max_dist", 0.25)):
                continue  # far from the anchor: not part of the arrangement
            dyaw = np.rad2deg((quat_yaw(env.data.xquat[b]) - yaw_a + np.pi) % (2 * np.pi) - np.pi)
            poses[n] = {"pos": rel.tolist(), "yaw_deg": float(dyaw)}
        if not poses:
            self._toast("No object is near the anchor: arrange the goal first", "warn")
            return
        self._apply_edits([{"op": "capture_goal", "anchor": anchor, "poses": poses,
                            "tol_pos": kw.get("tol_pos", 0.01), "tol_rot_deg": kw.get("tol_rot_deg", 10.0)}])

    # -- one control tick ------------------------------------------------------------------------------
    def _tick(self, now: float) -> None:
        self._process_commands(now)
        self._tick_count += 1
        env = self.env
        self._swap_policy()
        if self._hold_branch(now):
            return

        if self._replay is not None:
            r = self._replay
            env.set_state(r["states"][r["i"]])
            if not r["paused"]:
                r["acc"] += self._replay_speed  # recorded steps per tick: speed 1 = real time
                steps, r["acc"] = int(r["acc"]), r["acc"] % 1.0
                last = len(r["states"]) - 1
                if r["i"] + steps <= last:
                    r["i"] += steps
                elif self._replay_loop:
                    r["i"] = 0
                else:
                    r["i"], r["paused"] = last, True  # hold the last frame until replay_stop
            self._replay_buttons(now)
            self._publish(now, None)
            return

        if self._edit is not None:
            self._replay_buttons(now)
            self._publish(now, None)
            return

        with self._input_lock:
            frame = self._input
            input_hz = self._input_rate
        out = self.teleop.update(frame, now, self.dt)
        self._hand_of = dict(out.hand_of)
        for cmd in out.commands:
            self._handle(cmd, {}, now)
        if self.env is not env or self._replay is not None or self._edit is not None:
            return  # a button changed the scene or the mode: start over next tick
        for arm, cue in out.cues:
            self._cue(cue, arm)
        self._engaged = dict(out.engaged)

        q_measured = env.arm_q
        driver = self._who_drives(out)
        n_arms = len(self.arms)
        if driver == "policy":
            if self._driver != "policy":  # (re)starting: fresh from the current state
                self.policy.reset(env)
            q_wanted, grippers = self.policy.act(env)
            q_wanted, grippers = np.asarray(q_wanted, float), np.array(grippers, float)
            self._policy_grip = grippers.copy()
            self._grip_latch = [None] * n_arms
        else:
            for a in self.arms:
                p, R = out.targets[a]
                self.ik.set_target(a, p, mat_to_quat_wxyz(R))
            self.ik.follow(q_measured)
            q_wanted, grippers = self.ik.solve(), np.array(out.grippers, float)
            if self._driver == "policy":  # takeover: each gripper keeps the policy's command (a held object stays held)
                self._grip_latch = [float(g) for g in self._policy_grip]
            for i in range(n_arms):       # ... until that hand's trigger is brought to the same opening
                if self._grip_latch[i] is not None:
                    if abs(grippers[i] - self._grip_latch[i]) < GRIP_CATCH:
                        self._grip_latch[i] = None
                    else:
                        grippers[i] = self._grip_latch[i]

        prev_state = self.recorder.state
        self.recorder.tick(now)
        if prev_state == RecState.COUNTDOWN and self.recorder.recording:
            self._cue("rec_start")
            self._toast("Recording", "rec")

        recording = self.recorder.recording
        before = snapshot(env) if recording else None
        # DART: execute a slightly perturbed target but record the clean one, so the data shows how the operator
        # recovers from small errors (Laskey et al., 2017). The offset is a slow random drift: an Ornstein-Uhlenbeck
        # process (correlation DART_TAU_S) smoothed once more (DART_SMOOTH_S), because per-tick white noise is
        # filtered away by the servos and a raw OU path makes the arm tremble. Only on arms the operator is moving,
        # while recording; otherwise it decays to zero, never jumps.
        if self._dart_gain:
            a, b = self.dt / DART_TAU_S, self.dt / DART_SMOOTH_S
            on = np.repeat([recording and driver == "operator" and out.engaged[arm] for arm in self.arms],
                           env.b.n_joints)
            self._dart_ou += -a * self._dart_ou + on * self._dart_gain * np.sqrt(2 * a) * \
                self._noise.standard_normal(len(self._dart))
            self._dart += b * (self._dart_ou - self._dart)
        # The drift goes in before the safety filter, so it can't push a hand onto the table or the other arm;
        # the recorded action is what was executed minus the drift.
        q_exec, self._safety_rad = self.safety.filter(q_wanted + self._dart, q_measured,
                                                      env.data.qpos[env.b.finger_qadr])
        q_target = q_exec - self._dart
        ee = self.safety.tcp_poses(q_target)
        if driver == "policy":
            # Keep the operator side where the arms are, so taking control never makes them jump.
            self.ik.reset(q_target)
            self.teleop.sync_hold({a: self._ik_pose(a) for a in self.arms})
        self._driver = driver
        self._policy_steps += driver == "policy"
        teleop = teleop_fields(frame, [out.engaged[a] for a in self.arms], [out.tracked[a] for a in self.arms],
                               self._rtt_ms, input_hz) if recording else None
        source = SOURCE_POLICY if driver == "policy" else SOURCE_OPERATOR
        step, success, _ = control_step(env, before, q_exec, grippers, ee, teleop, source, q_recorded=q_target,
                                        correction=self._safety_rad)
        if step is not None and self.recorder.append(now, step, success):
            self._toast("Success! Review, then save or discard", "success")
        self._success = success
        self._task_feedback(now, success)
        self._check_warnings(now)
        self._publish(now, out, success)

    def _replay_buttons(self, now: float) -> None:
        """While nothing is driven (replay, edit mode) the controller buttons still send their commands."""
        with self._input_lock:
            frame = self._input
        out = self.teleop.update(frame, now, self.dt)
        self._hand_of = dict(out.hand_of)
        self.teleop.release_all()
        for cmd in out.commands:
            self._handle(cmd, {}, now)

    def _who_drives(self, out) -> str:
        if self.policy is None or self.control_mode == "teleop":
            return "operator"
        if self.control_mode == "shared" and any(out.engaged.values()):
            return "operator"  # the operator took over; the policy resumes when every arm is released
        return "policy"

    # -- feedback ------------------------------------------------------------------------------------
    def _task_event(self, name: str, now: float, arm: str | None = None, obj: str | None = None,
                    stage: str | None = None) -> None:
        """A task event -> its cue (the task's `events` table, then the defaults) and its toast."""
        if arm is None and obj is not None:
            arm = self._last_holder.get(obj)
        for e in self._event_cues:
            if e["on"] == name and (e.get("stage") in (None, stage)):
                self._cue(e["cue"], arm, now)
                if e.get("toast"):
                    label = stage
                    if stage is not None:
                        label = next((s["label"] for s in self._task_spec["stages"] if s["name"] == stage), stage)
                    self._toast(e["toast"].format(stage=label or "", object=obj or ""), "success")
                return

    def _task_feedback(self, now: float, success: bool) -> None:
        """Grasp / hold / release, the task's own events (snaps), each stage reached or lost, and success."""
        env = self.env
        held = env.held()
        for a in self.arms:
            b, was = held[a], self._held.get(a)
            if b is not None:
                self._last_holder[b] = a
            if b is not None and was is None:
                self._task_event("grasp", now, a, b)
                self._hold_tick_at[a] = now + self.cfg.feedback.hold_tick_s
            elif b is None and was is not None:
                self._task_event("release", now, a, was)
            elif b is not None and now >= self._hold_tick_at[a]:
                if now >= self._quiet_until[a]:
                    cue = self.cfg.feedback.cues.get("hold")
                    if cue and cue.haptic:
                        self._emit({"type": "haptic", "arm": a, "hand": self._hand_of.get(a),
                                    "intensity": cue.haptic[0], "ms": int(cue.haptic[1]), "repeat": 1, "gap_ms": 0})
                self._hold_tick_at[a] = now + self.cfg.feedback.hold_tick_s
        self._held = held

        for ev in env.events:
            self._task_event(ev.name, now, ev.arm, ev.object, ev.stage)
        progress = env.progress()
        names = env.task.stage_names
        stage_objects = [s["object"] for s in self._task_spec["stages"]]
        for k in range(self._progress, progress):       # a stage that was lost can fire again later
            self._task_event("stage_reached", now, None, stage_objects[k] if k < len(stage_objects) else None,
                             names[k] if k < len(names) else None)
        for k in range(progress, self._progress):
            self._task_event("stage_lost", now, None, None, names[k] if k < len(names) else None)
        self._progress = progress
        if success and not self._complete:
            self._task_event("success", now)
        self._complete = success

    def _check_warnings(self, now: float) -> None:
        link_table, arm_hit, tip_table = (bool(f) for f in self.env.contact_flags())
        table_hit = link_table or tip_table
        reach = {a: e[0] > self.cfg.ik.warn_m for a, e in self.ik.errors().items()}
        blocked = self._safety_rad > self.cfg.safety.warn_rad  # the operator pushes into the table or the other arm
        new = {"arm_collision": arm_hit, "table_collision": table_hit, "ik_saturated": any(reach.values()),
               "blocked": blocked}
        for k, v in new.items():
            if v and not self._warn[k]:
                self._cue("warn")
        if arm_hit and not self._warn["arm_collision"]:
            self._warn_cue("collision", None, now)
        if table_hit and not self._warn["table_collision"]:
            self._warn_cue("table", None, now)
        if blocked and not self._warn["blocked"]:
            self._warn_cue("collision", None, now)
        for a in self.arms:
            if reach[a] and not self._reach.get(a):
                self._warn_cue("out_of_reach", a, now)
        self._warn, self._reach = new, reach

    # -- publish -------------------------------------------------------------------------------------
    def _publish(self, now: float, out, success: bool = False) -> None:
        env = self.env
        engaged = [bool(out.engaged[a]) for a in self.arms] if out is not None and self._live_env is None else []
        frame = pack_pose_frame(self._pose_seq + 1, _ms(), body_poses(env.data),
                                pose_flags(engaged, self.recorder.recording))
        with self._out_lock:
            self._pose_seq += 1
            self._pose_frame = frame
        self._publish_ghost()
        if self._tick_count % STATUS_EVERY == 0:
            self._update_status(now, out, success)
        if self._tick_count % PREVIEW_EVERY == 0 and now < self._previews_wanted_until:
            self._render_previews()

    def _render_previews(self) -> None:
        env = self.env
        if self._preview_model is not env.model:  # first use, or the scene changed: a renderer for this model
            if self._preview_renderer is not None:
                self._preview_renderer.close()
            self._preview_renderer, self._preview_model = None, env.model
            try:
                cams = [c for c in (self.cfg.record.cameras or env.camera_names) if c in env.camera_names]
                self._preview_renderer = CameraRenderer(env.model, *PREVIEW_SIZE, cams)
            except Exception:  # noqa: BLE001 - previews are optional (no GL available)
                log.exception("camera previews disabled: offscreen rendering failed")
        if self._preview_renderer is not None:
            jpegs = self._preview_renderer.render_jpeg(env.data)
            with self._out_lock:
                self._previews = jpegs

    def _review_summary(self) -> dict | None:
        """Quality of the take under review, judged by the same rules the audit and the exporters use, so the
        operator can discard a take that a dataset would not use anyway (computed once per take)."""
        rec = self.recorder
        if rec.state != RecState.REVIEW or not rec.steps:
            self._review = None
            return None
        if self._review is None or self._review[0] != len(rec.steps):
            steps = rec.steps
            data = {"obs/joint_vel": np.array([s["obs"]["joint_vel"] for s in steps]),
                    "teleop/engaged": np.array([s["teleop"]["engaged"] for s in steps]),
                    "task/progress": np.array([s["task"]["progress"] for s in steps]),
                    "task/held": np.array([s["task"]["held"] for s in steps]),
                    "contacts": np.array([s["contacts"] for s in steps]),
                    "control/source": np.array([s["control"]["source"] for s in steps]),
                    "safety/correction": np.array([s["safety"]["correction"] for s in steps])}
            meta = {"stage_names": self.env.task.stage_names, "arms": self.arms, "objects": self.env.objects}
            a = audit_arrays("take", self.env.control_hz, rec.success_seen, data, meta)
            why = verdict(a, self.cfg.curate, {})
            spans = [] if why else kept_spans(a, self.cfg.curate)
            if not why and not spans:
                why = "nothing left after trimming"
            self._review = (len(steps), {
                "duration_s": round(a.duration_s, 1), "success": a.success, "idle_pct": round(100 * a.idle_frac),
                "events": {k: a.contact_events[k] for k in CONTACT_KINDS}, "safety_events": a.safety_events,
                "progress": int(data["task/progress"].max()), "stages": len(self.env.task.stage_names),
                "rejected": why, "used_s": round(sum(e - s for s, e in spans) / a.fps, 1),
                "tags": self._task_spec.get("tags", [])})
        return self._review[1]

    def _guide(self) -> dict | None:
        """Where the next object belongs and how far it is from there (operator-only ghost + distance readout)."""
        if not self.cfg.feedback.guidance or self._live_env is not None:
            return None
        env = self.env
        goal = getattr(env.task, "next_goal", lambda *_: None)(env.model, env.data)
        if goal is None:
            return None
        p = env.data.xpos[env.world.body[goal["object"]]]
        d = np.asarray(goal["pos"]) - p
        return {**goal, "dist_mm": round(float(np.linalg.norm(d)) * 1000, 1),
                "xy_mm": round(float(np.linalg.norm(d[:2])) * 1000, 1), "z_mm": round(float(-d[2]) * 1000, 1)}

    def _next(self) -> dict | None:
        env = self.env
        if self._live_env is not None:
            return None
        obj = env.task.next_object(env.model, env.data)
        if obj is None:
            return None
        arm = None
        if obj in env.world.body:
            xy = env.data.xpos[env.world.body[obj]][:2]
            mounts = self.bundle.embodiment["arms"]
            arm = min(self.arms, key=lambda a: float(np.linalg.norm(np.asarray(mounts[a]["pos"][:2]) - xy)))
        return {"object": obj, "arm": arm}

    def _update_status(self, now: float, out, success: bool) -> None:
        rec, env = self.recorder, self.env
        with self._input_lock:
            frame = self._input
            input_age = (now - frame.recv_time) if frame is not None else None
            rate = self._input_rate if input_age is not None and input_age < 1.0 else 0.0
        live = self._live_env is None
        errs = self.ik.errors() if live else {}
        arms = list(env.arms)
        status = {
            "type": "status",
            "task": env.task.name,
            "task_version": env.bundle.version,
            "scene_version": self._scene_version,
            "instruction": env.task.instruction,
            "arms": arms,
            "mode": self.teleop.mode,
            "teleop_control": self.teleop.control,
            "mapping": dict(self.teleop.mapping),
            "hand_of": dict(self._hand_of) if live else {},
            "scale": self.teleop.scale,
            "rec_state": rec.state.value,
            "countdown": round(rec.countdown_left(now), 2),
            "elapsed": round(rec.elapsed(now), 2),
            "steps": len(rec.steps),
            "auto_stop": rec.auto_stop,
            "success": bool(success),
            "success_seen": rec.success_seen,
            "progress": env.progress(),
            "stages": env.stage_flags().tolist(),
            "held": env.held() if live else {a: None for a in arms},
            "next": self._next(),
            "guide": self._guide(),
            "operator": self.cfg.record.operator,
            "session": {"name": self.cfg.record.session, "saved": rec.saved_count, "discarded": rec.discarded_count,
                        "target": self.cfg.record.target_takes,
                        "elapsed_s": round(time.monotonic() - rec.session_start)},
            "engaged": {a: bool(out.engaged[a]) if out and live else False for a in arms},
            "tracked": {a: bool(out.tracked[a]) if out and live else False for a in arms},
            "mode_mismatch": bool(out.mode_mismatch) if out else False,
            "sources": {h: (frame.hand(h).source if frame is not None and input_age < 1.0 else 0) for h in ("l", "r")},
            "grippers": {a: round(float(out.grippers[i]), 3) if out and live else 1.0 for i, a in enumerate(arms)},
            "ik_error_mm": {a: round(e[0] * 1000, 1) for a, e in errs.items()},
            "out_of_reach": {a: bool(errs[a][0] > self.cfg.ik.warn_m) if a in errs else False for a in arms},
            "ee_pos": {a: env.data.site_xpos[sid].round(4).tolist() for a, sid in zip(arms, env.b.tcp_sites)},
            "warnings": dict(self._warn),
            "input_hz": round(rate, 1),
            "rtt_ms": round(self._rtt_ms, 1) if rate > 0 else None,
            "loop_hz": round(self._loop_hz, 1),
            "connected": rate > 0,
            "placement": self.placement.column_major(),
            "table_offset": getattr(self.placement, "y_offset", 0.0),
            "control": {"mode": self.control_mode, "driver": self._driver,
                        "policy": None if self.policy is None else self.policy.name, "ref": self._policy_ref,
                        "loading": self._policy_loading, "error": self._policy_error,
                        "policy_s": round(self._policy_steps * self.dt, 1), "safety_rad": round(self._safety_rad, 3)},
            "review": self._review_summary(),
            "edit": None if self._edit is None else {"moved": list(self._edit["moved"]),
                                                     "file": str(self.bundle.path)},
            "ghost": None if self._ghost is None else {"path": self._ghost["path"], "i": self._ghost["i"],
                                                       "n": len(self._ghost["states"])},
            "branch": None if self._branch is None else {"of": self._branch["of"], "step": self._branch["step"],
                                                          "frozen": self._branch["frozen"]},
            "replay": None if self._replay is None else {
                "path": self._replay["path"], "i": self._replay["i"], "n": len(self._replay["states"]),
                "fps": int(self._replay["meta"].get("fps", env.control_hz)),
                "paused": self._replay["paused"], "speed": self._replay_speed, "loop": self._replay_loop,
                "foreign": not live},
        }
        with self._out_lock:
            self._status = status
