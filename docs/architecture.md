# Architecture

For developers changing xrcollect. User-facing behaviour is in the [README](../README.md); the short version for
coding agents is in [CLAUDE.md](../CLAUDE.md). This file explains how the parts connect: threads, the control
tick, the wire protocol, the HTTP endpoints, and which files have to change together.

## Processes and threads

`xrc serve` is one Python process:

| Thread | Owns | Talks to the others through |
|---|---|---|
| **sim** (`Runtime._run`, 50 Hz) | `Env` (MjModel/MjData), IK, teleop, safety filter, recorder state machine, policy `act` | a command queue in; lock-protected snapshots (pose frame, ghost frame, status, previews) and event listeners out |
| **asyncio** (uvicorn, `server.py`) | HTTP endpoints, websockets | `Runtime.command`, `submit_input`, `status()`, `latest_pose_frame()`; events arrive with `call_soon_threadsafe` |
| **hdf5-writer** (`Recorder._writer`, 1 worker) | writing a finished take to disk | a `Future` per save; the sim thread never blocks on disk |
| **policy-load** (short-lived) | importing and constructing a `PolicyRunner` | the result is handed to the sim thread at a tick boundary |
| `asyncio.to_thread` / executor workers | episode lists, audit, dataset health, export jobs | read episode files only; never touch the live `Env` |

The rule: **only the sim thread touches simulator state.** `server.py` never reads `env.data`. Every action, from
the headset, the dashboard, a controller button or a test, becomes a command:

- `POST /api/command {name, args}`, or
- `{"type": "command", name, args}` on either websocket.

Commands are queued and run at the start of the next tick. Exports and the audit rebuild their own `Env` from each
episode (`env_from_episode`), on worker threads.

`xrc rollout` runs a `PolicyRunner` without the server: its own `Env` per task, the same safety filter and
`recorder.control_step` as a live tick, start layouts from `stream_seed("eval", i)`. The bundled
`policies/lerobot.py:LeRobotRunner` builds its inputs with `export.base.observation_arrays` and `CameraRenderer`
(the exporter's own code) and reads the layout and fps from the export manifest. See [training.md](training.md).

## The control tick

One tick = 1/50 s of sim time, in this order (`runtime.py` docstring):

1. **Commands**: drain the queue (record toggle, save, reset, edit ops, replay controls, load policy, ...).
2. **Teleop**: `Teleop.update(latest InputFrame)` turns XR poses into per-arm TCP targets. The pipeline is
   placement into the MuJoCo world, jump rejection, One-Euro filter, clutch, workspace clamp. A frame older than
   `INPUT_TIMEOUT_S` (0.25 s) counts as lost tracking, and the arm freezes.
3. **Driver**: the operator's targets go through IK (`ik.py`). In policy mode, or shared mode before a takeover,
   `PolicyRunner.act(env)` drives instead.
4. **Safety filter** (`safety.py`): every joint target, whoever made it, is clamped away from collisions. The
   result is the recorded action.
5. **Recorder tick**: countdown, auto-stop when success has held for `auto_stop_tail_s`.
6. **Snapshot** of `Env.get_state()` before the action (this becomes `sim/state[t]`).
7. **Apply and step**: DART noise on the executed target (if on), physics substeps, success, progress and stages,
   then the recorded step (`recorder.make_step`).
8. **Append** the step to the take's buffers.
9. **Task feedback**: `Task.after_step` events (grasp, snap, stage reached, ...) mapped to cues.
10. **Warnings**: IK error, safety interventions, mode mismatch, out of reach.
11. **Publish**: a pose frame every tick, status every 5 ticks (10 Hz), camera previews about every 6 ticks while
    someone is watching.

Replay and edit mode skip steps 2–10 and only publish.

- **Replay** sets recorded `sim/state` rows and runs `mj_forward`.
- **Edit mode** pauses physics, shows objects at their default poses and moves them with `edit_move`.
  `edit_commit` writes the moves into the task file.

## Wire protocol

**Binary frames** are described once, in `xrcollect/wire/schema.json`. `xrc codegen` writes
`xrcollect/protocol_gen.py` and `web/src/protocol.gen.ts`; both are checked in, and `tests/test_protocol.py`
fails if they drift from the schema. Little endian.

| Frame | Direction | Content |
|---|---|---|
| input (`MSG_INPUT`) | headset → server, every XR frame | header `type, src_left, src_right, version, seq, t_echo_ms`; head pose; per hand: pose, 8 buttons, 25 hand joints × 7, 2 pinch distances |
| pose (`MSG_POSE`) | server → clients, every tick | header `type, flags (engaged bits, recording), nbody, seq, t_server_ms`; body poses for the three.js scene |
| ghost (`MSG_GHOST`) | server → clients | the same layout, for the ghost of a replayed take |

Every input frame carries the protocol `version`. A mismatched client gets a readable
`{"type": "error", text}` and is closed with code 4000. `web/src/net.ts` shows it as `Link.refused`. `t_echo_ms`
echoes the last `t_server_ms`, which gives the server its RTT estimate.

**JSON messages** on the websockets:

- `Status`: about 10 Hz, the full UI state.
- `ServerEvent`: toasts, cues, haptics, `scene_changed`, `edit_error`, `error`.

They are plain dicts built in `runtime.py` (`status()`, `_emit`) and hand-typed in `web/src/protocol.ts`. There is
no generator for them: when a dict changes, update `protocol.ts` in the same change, and `npm run build` (tsc)
catches the clients that break.

| Socket | Receives input | Pose frames | Purpose |
|---|---|---|---|
| `/ws/xr` | yes | every new tick | headset page |
| `/ws/dash` | no | ≤ 30 Hz | dashboard viewer |

## HTTP endpoints (`server.py`)

| Endpoint | |
|---|---|
| `GET /api/status` | the same `Status` dict as the websockets |
| `POST /api/command` | `{name, args}` → the sim thread's command queue |
| `GET /api/scene`, `/api/scene.bin` | visual geometry of the current scene (JSON + binary meshes); `MujocoScene.ts` rebuilds it |
| `GET /api/task` | `TaskSpec`: instruction, objects, stages, goal view; drives the goal panel and stage lists |
| `GET /api/commands` | the controller button table (`teleop.BUTTONS` → `command_table`) |
| `GET /api/protocol` | the protocol version |
| `GET /api/schema` | JSON Schema of task files (`task/schema.py`), used to build the Scene tab forms |
| `GET /api/tasks`, `/api/library` | task folders; object assets and the predicate catalogue |
| `GET /api/taskfile` | the task file being edited: text, parsed data, and the **runtime's** version (it lags the text until the reload finishes) |
| `POST /api/scene/reload` | hot reload after editing files by hand |
| `GET /api/datasets` | episode folders next to `record.out_dir` (with optional `dataset.json`) |
| `GET /api/episodes`, `PATCH /api/episodes/{file}` | episode rows with labels; label changes go through `audit.update_curation` |
| `GET /api/audit`, `/api/health` | per-episode verdicts and stats; dataset health for the Data tab |
| `GET /api/exporters`, `POST /api/export/preview`, `POST /api/export` | exporters, layout, curate rules, jobs; preview of the selection; start a background export to `<out_dir>/../exports/<name>` |
| `GET /api/preview/{cam}.jpg` | live camera preview |
| `GET /api/hint` | connection help for the headset landing page |
| `/xr/`, `/dashboard/` | the built web app (`web/dist`) |

## Data model in one paragraph

A **bundle** (`task/loader.py`) is the resolved task: embodiment, scene, objects and task logic, plus plugin
source. Its content hash is `task_version`. `Env(bundle)` composes the scene with MjSpec (`scene/compose.py`) and
always runs the model compiled from the saved XML, so a live scene and one rebuilt from an episode are identical.
An **episode** stores the bundle, the config and the compiled XML, so `env_from_episode` rebuilds its world
without the library files. Everything derived (success, stages, audit verdicts, export frames, camera images) is
recomputed from `sim/state`. That is why predicates must be pure functions of the state, and why task state
outside MuJoCo goes through `extra_state`.

## Gymnasium (`gym_env.py`)

`XRCollectEnv` wraps one `Env` and registers `xrcollect/<task>-v0` for every task folder. `step` = (optional)
`SafetyFilter` → `Env.apply_action` → `Env.step`, with reward from `Env.success` / `Env.progress`; the action layout
is a recording's `action/joint_pos`, so demonstrations replay as gym trajectories (`tests/test_gym_env.py`).
`reset` draws the layout seed from gymnasium's `np_random`; `options={"state": ...}` then applies a saved
`sim/state` through `Env.set_state`. It adds no task logic of its own: new tasks are gym environments for free.

## Files that change together

| When you change | Also change |
|---|---|
| a binary frame field or constant | `wire/schema.json`, then `xrc codegen`; bump `version` if old clients would misread it |
| a `Status` / event / `TaskSpec` / `EpisodeRow` dict | `web/src/protocol.ts`, and the consumers in `xr/` and `dashboard/` |
| a recorded key or file attribute | `recorder.py` (`make_step`, `episode_meta`, docstring), `SCHEMA_VERSION` + changelog, the README data table, readers in `audit.py` / `export/base.py` |
| a task file field | `task/loader.py` / `declarative.py`, `task/schema.py` (forms), `editor.py` if it is editable, the README task guide |
| a predicate | `task/predicates.py` and its `CATALOGUE` entry (doc + template for the editor) |
| a controller button | `teleop.BUTTONS` only: labels and hints are served from it |
| a sound or haptic cue name | `config._default_cues`, `runtime.DEFAULT_EVENT_CUES`, `web/src/xr/sounds.ts` |
| a contact kind | `env.CONTACT_KINDS`, the curate config (`max_contact_frac`, `cut_contacts`), the Data tab |
| curation rules | `audit.why_not` / `select` / `kept_spans` only: the review card, Episodes tab and exporters all call them |
| what goes into an exported state / image | `export.base.observation_arrays` / `CameraRenderer` only: `LeRobotRunner` builds a policy's live input with them |
| an exporter | a class with `begin` / `add_episode` / `finish`, registered in `export/base.EXPORTERS` |
| a headset panel | create it in `xr/main.ts`, register world panels with `layout.add`, expose it on `window.__xrc` for e2e |
| a setup step (extra, fetched asset, build step, Node or Python requirement) | `scripts/install.sh`, the README Quick start and its manual-install block, the CLAUDE.md command list, `xrc doctor` if it can be checked |
| anything a user sees or does | README (recapture `docs/media/` with `web/e2e/screenshots.mjs` if it shows); anything about threads, tick order, endpoints or protocol: this file |

## Testing

- `uv run --group dev pytest`. Covers task loading and predicates, snap, bit-exact recording and `resim`, the
  runtime (commands, replay, branching, edit ops, hot reload), the safety filter, audit and export, server
  endpoints, protocol drift, and the Gymnasium wrapper (`check_env` per task, rewind, demos replayed as gym
  trajectories).
- `web/e2e/emulated-quest.mjs` (`npm test`). Meta's IWER emulates a Quest 3 in headless Chromium and drives the
  real headset page. It covers:
  - controllers and hands, clutch, gripper, recording buttons;
  - review card, timeline replay, edit mode and capture goal (pressed through each panel's widgets);
  - the mode-mismatch warning.
- `web/e2e/dashboard.mjs` (`npm run test:dashboard`). The Scene tab (click select, gizmo drag, Apply, inspector,
  regions, capture goal) and the Data tab (tiles, tooltips, export preview, an export job).
- Both e2e scripts need a scratch setup:
  - `record.out_dir` outside `data/`;
  - `--task` pointing at a copy of a task outside the repo.

  The edit checks skip themselves otherwise. Debug handles are `window.__xrc` (headset page),
  `window.__xrcViewer` (dashboard 3D view) and `window.__xr` (IWER).
- `web/e2e/screenshots.mjs` (`npm run screenshots`) is not a test: it captures the README screenshots into
  `docs/media/` (instructions at its top).
- The emulated headset must keep its input stream inside the 0.25 s tracking timeout. On a heavily loaded machine
  the controller-follow checks can fail for that reason alone; the test prints the server-measured input rate next
  to each check.
