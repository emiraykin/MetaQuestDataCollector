# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

xrcollect: a task-agnostic platform for collecting robot manipulation demonstrations in MuJoCo with a Meta Quest 3
(WebXR). The operator teleoperates simulated arms (bimanual or single Franka), takes are recorded losslessly to
HDF5 (full `sim/state` per step, no images), reviewed and labelled, then exported (raw HDF5, LeRobot) with camera
images re-rendered from the states. A task is data (YAML, optionally a Python plugin), not code in the platform.
Python 3.12 backend (`xrcollect/`), TypeScript/three.js/React frontend (`web/`: `xr/` headset page, `dashboard/`).

Distilled from QuestCollect, an earlier single-task (DUPLO tower) collector that is not part of this repo. If a
local checkout of it is around, treat it as a **read-only reference**: copy from it, never write to it. Old
QuestCollect episode files are not readable here (schema restarted at v1).

Human docs to keep in sync with behaviour: `README.md` (the public GitHub front page: pitch, screenshots, install,
controls, task authoring, data format) and `docs/architecture.md` (developers: tick order, threads, endpoints,
protocol, files that change together). README media live in `docs/media/`: the `*.png` screenshots come from
`web/e2e/screenshots.mjs` (`npm run screenshots`; recapture after visible UI changes), the task GIFs and
`cameras.gif` from the scripted expert rendered offscreen through `Expert.on_step` (free camera for the tasks,
`scene.render.CameraRenderer` for the camera grid), encoded with ffmpeg palettegen at 480 px. Keep each GIF under
~4 MB, and never show local paths or IPs in a screenshot (serve a task from inside the repo so the Scene tab shows
a relative path).

## Commands

```bash
./scripts/install.sh [--lerobot] [--mesh] [--e2e] [--all] [--yes]   # all setup below in one go, no sudo, re-runnable
./scripts/fetch_assets.sh               # Franka model from mujoco_menagerie -> assets/third_party/ (gitignored)
uv sync                                 # extras: --extra lerobot (LeRobot export, brings torch), --extra mesh (import-mesh)
(cd web && npm ci && npm run build)     # the server serves web/dist: rebuild after every web change
uv run xrc doctor                       # environment checklist (assets, EGL, web build, protocol drift, adb)

uv run xrc serve [--task tasks/duplo_tower] [--mode hands] [--config my.yaml] [--no-https] [--out data/x]
uv run xrc serve --policy my_pkg.mod:Runner [--control shared|policy]
uv run xrc setup-quest | certs          # USB port forward (adb reverse) | self-signed TLS for Wi-Fi WebXR
uv run xrc tasks | view [task] | replay <ep.hdf5> | inspect <ep.hdf5> | resim <ep.hdf5>
uv run xrc validate-task tasks/<name> [--n 5] [--embodiment franka_single]   # scripted expert must reach success
uv run xrc audit [--exclude ep.hdf5 --note why]
uv run xrc export <name> [--exporter hdf5|lerobot] [--cameras a,b|none] [--fps 25] [--annotations] [--allow-mixed]
uv run xrc rollout xrcollect.policies.lerobot:LeRobotRunner --policy-arg checkpoint=outputs/train/<run> --n 20   # success rate, headless
uv run xrc import-mesh mug.obj kitchen/mug [--scale 0.001]   # -> objects/kitchen/mug.yaml (+ convex pieces)
uv run xrc codegen [--check]            # regenerate protocol files from xrcollect/wire/schema.json
uv run python -c "import gymnasium as gym, xrcollect.gym_env; gym.make('xrcollect/stack_cups-v0')"   # any task as a gym env

uv run --group dev pytest                                   # all tests (~1 min); MUJOCO_GL=egl is set by conftest/cli
uv run --group dev pytest tests/test_task.py::test_name     # single test

cd web && npm run typecheck             # tsc --noEmit (also part of npm run build)
cd web && npm run dev                   # vite on :5173, proxies /api and /ws to :8000
```

`scripts/install.sh` installs uv per user if missing, uses the system Node only if it is 20.19+ / 22.12+ (Vite 8)
and otherwise downloads Node 22 LTS into `.tools/node` (gitignored, checksum-verified), then runs the steps above
and `xrc doctor`. It must stay free of absolute paths and sudo: system libraries (EGL, Playwright's Chromium deps)
are only detected and the command printed. When setup steps change (a new extra, asset, build step or Node
requirement), update the script, the README Quick start (and its manual-install block) and this list together. If
`node` is not on PATH, use `PATH="$PWD/.tools/node/bin:$PATH"`.

Headset-free end-to-end tests in `web/e2e/` (setup at the top of each script): `emulated-quest.mjs` (`npm test`,
Meta's IWER emulated Quest 3 + Playwright on the real headset page) and `dashboard.mjs` (`npm run test:dashboard`,
Scene + Data tabs). Once: `./scripts/install.sh --e2e` (or `cd web/e2e && npm ci && npx playwright install chromium && npm run build`). Serve a
scratch setup: `cp -r tasks/stack_two_blocks <scratch>/task` and `uv run xrc serve --no-https --config <yaml with a
scratch record.out_dir> --task <scratch>/task`, then `XRC_URL=http://localhost:<port> npm test`. They record and
export into the server's out_dir and rewrite the served task file, so never point them at `data/episodes` or a
task in `tasks/` (the edit checks skip themselves when the task file is inside the repo). Debug handles:
`window.__xrc` (headset page objects), `window.__xrcViewer` (dashboard 3D view: renderer, camera, mj, gizmo),
`window.__xr` (IWER). `screenshots.mjs` (`npm run screenshots`) is not a test: it writes `docs/media/*.png`,
records and discards one take, and enters edit mode without applying, so it may serve a repo task (it never edits
the file) but still needs a scratch out_dir; seed that with `xrc validate-task <task> --n 8 --save <dir>` so the
Episodes and Data tabs have content. `/api/taskfile` returns the file text from disk but the runtime's `version`, which changes
only after the reload: tests that wait for an edit must wait for the version too. Under heavy CPU load the
emulated input stream can exceed the 0.25 s tracking timeout and the controller-follow checks fail.

## Architecture

Data flow: Quest Browser → `/ws/xr` binary input frames → `Runtime` sim thread (teleop → IK or policy → safety
filter → physics → recorder) → binary pose frames back to the headset + JSON status/events to both clients
(`/ws/dash` for the dashboard).

- **Four task layers** (`task/loader.py`): embodiment (`embodiments/*.yaml`: adapter, arms + mounts, home pose,
  limits, safety margins), scene (`scenes/*.yaml`: table, props, lights, cameras), objects (`objects/**.yaml`
  library, referenced as `asset:`), task logic (`tasks/<name>/task.yaml`, optional `task.py` plugin, optional
  `expert.py`). `load_bundle` resolves them into one plain-data `Bundle`; its content hash is the `task_version`.
  Every episode stores the bundle (`task_yaml`), the platform config and the compiled `model_xml`, so it rebuilds
  its exact scene without the library files.
- **Task protocol** (`task/base.py`): the runtime, recorder, audit and UI only know `Task`. A YAML task compiles
  into `DeclarativeTask` (`task/declarative.py`); a plugin subclasses it. Stages and success are predicates
  (`task/predicates.py`, `CATALOGUE` feeds the editor) and must be **pure functions of the simulator state**, so
  replay, audit and export recompute them from `sim/state`. Anything a task changes outside MuJoCo's state goes
  through `extra_state` / `set_extra_state`.
- **State**: always `Env.get_state` / `Env.set_state` (`mj_getState` + task extra state, e.g. snap weld poses in
  `eq_data`). Never raw `mj_getState`.
- **Snap** (`task/snap.py`): press fits are inactive welds that switch on (touch + alignment test + press force)
  and off (tension + shear above `pull_n`, compression excluded). The weld must stay softer than the contacts.
- **Robots** are behind `RobotAdapter` (`embodiment/base.py`; Franka in `embodiment/franka.py`). The rest of the
  platform uses the adapter and the `Bound` index tables only. Arm names (`l`, `r`, ...) come from the embodiment
  and key every per-arm field (`status.grippers[arm]`, `ee_pos[arm]`); hands map to arms via `teleop.mapping`.
- **Teleop** (`teleop.py`): placement → jump rejection → One-Euro filter → clutch → workspace clamp. `ArmClutch`
  has three position modes (`teleop.control` / `set_control`, `CONTROL_MODES`): `clutched` (default, relative,
  scaled), `absolute` (gripper at the hand), `rate` (hand offset = velocity, `rate_gain`, own workspace clamp).
  Orientation is always relative to the engage pose, so re-engaging never jumps. Grip toggles the clutch; in hands
  mode `HandMapper` toggles it when middle, ring and little finger curl (`finger_curl`: tip–wrist / knuckle–wrist,
  computed server-side from the 25 joints; `curl_on` / `curl_off` hysteresis) and holds the gripper while they
  are curled. Thumb–index distance (`pinch[0]`) is the gripper; `pinch[1]` (thumb–middle) is only recorded.
- **Scene composition** (`scene/compose.py`): `build_spec` with MjSpec, then the model that runs is always compiled
  from the saved XML (`compile_scene`), so a live scene and one rebuilt from an episode are identical. World: z up,
  +x away from the operator, +y left. Geom group 5 = operator-only visuals (cameras render groups 0–2); regions are
  tape geoms in group 1 (visible to cameras). `scene/export.py` serves visual geometry at `/api/scene(.bin)`;
  `web/src/MujocoScene.ts` rebuilds it in three.js for both pages.
- **`runtime.py` is the hub.** One real-time thread (50 Hz) owns env, IK, teleop, safety, recorder. `server.py`
  never touches sim state: every action from headset, dashboard or buttons is a command (`POST /api/command` or a
  websocket `{"type": "command"}`) processed at the start of a tick. Tick order is in the module docstring.
  Replay and edit mode skip teleop through warnings.
- **Wire protocol**: one description in `xrcollect/wire/schema.json`; `xrc codegen` writes
  `xrcollect/protocol_gen.py` and `web/src/protocol.gen.ts` (checked in; `tests/test_protocol.py` fails on drift).
  Hand-written `protocol.py` / `web/src/protocol.ts` use only the generated names. A version byte in every input
  frame; the server refuses mismatched clients with a readable `error` event (`Link.refused` in `web/src/net.ts`).
  JSON message types (`Status`, `ServerEvent`, `TaskSpec`, `EpisodeRow`, ...) are hand-typed in `protocol.ts`:
  when the server's dicts change, update them and the dashboard together (`npm run build` catches mismatches).
- **UI is data-driven**: no task names in client code. Goal picture, stage list, next-object hint and guidance
  ghost come from `GET /api/task` (`TaskSpec`) and `status.progress/stages/next/guide`; `web/src/goal.ts` draws the
  goal for the headset `GoalPanel` and the dashboard Goal card. Dashboard tabs (`web/src/dashboard/App.tsx`):
  Record, Episodes, Policy (`Policy.tsx`), Scene (`SceneEditor.tsx` forms + `Viewer3D.tsx` click-select and
  TransformControls gizmo -> `edit_move`), Data (`DataTab.tsx`: `/api/health` charts and the export form with a live
  `/api/export/preview`). Controller labels come from `GET /api/commands`
  (`teleop.BUTTONS` / `command_table`, the single button table).
- **Headset page** (`web/src/xr/main.ts` wires it; `immersive-vr`, or `immersive-ar` with passthrough for "Enter
  Mixed Reality"): `Panel.ts` (canvas panels), `Layout.ts` (`PanelLayout`, ray-grab and persisted panel poses;
  follow modes world/body/wrist; register world panels with `layout.add`), `WristMenu.ts` (three pages: main =
  mode, record/save/discard, recenter, scale; layout = previews and panels; tools = replay/ghost, edit, guide, swap
  hands, control mode, table height), `Hud.ts`, `GoalPanel.ts`,
  `CamPreviewPanel.ts`, `ReviewPanel.ts` (review card: verdict, success toggle, tags, quality), `TimelinePanel.ts`
  (VR replay: play/pause/seek/speed/step, take over), `EditMode.ts` (move objects, capture goal),
  `visuals.ts` (controller models vendored in `web/public/xr-profiles`, hints, pointer), `sounds.ts` (WebAudio cues;
  names must match the server's cue names). Anything the operator needs mid-session should be reachable from the
  headset (a button sends the same command the dashboard does); the dashboard adds desktop-only work (task
  editing forms, browsing old episodes, policy loading, export).
- **Feedback**: task events map to cues by name (`events:` in the task file + `runtime.DEFAULT_EVENT_CUES`); cue
  patterns are `config.feedback.cues` (defaults in `config._default_cues`). Haptic events carry `repeat`/`gap_ms`,
  scheduled client-side.
- **Safety filter** (`safety.py`) runs on every joint target whoever made it (operator IK, policy, script); it only
  prevents collisions (arms apart, links/fingers off the table), never helps with the task. The filtered targets
  are the recorded action; `safety/correction` is recorded per step. No grasp assist.
- **Recording** (`recorder.py`, file layout in its docstring, `SCHEMA_VERSION`): IDLE → COUNTDOWN → RECORDING →
  REVIEW → save/discard. Images are never recorded. DART noise (`record.action_noise`) perturbs executed, not
  recorded, targets; `Env.executed_targets` recovers what was sent from `sim/state`. Recorded actions replay
  bit-exactly from `sim/state[0]` (`xrc resim`, tests). "Take over here" (`replay_branch`, `_start_branch`) writes
  `branched_from` / `branch_step`.
- **Replaying an episode**: `env_from_episode(meta)` (its own bundle + `scene_hash` check), never an `Env` from the
  current config. The runtime switches to the episode's scene if it differs and emits `scene_changed`.
- **Hot reload + editing** (`editor.py`): edits are small ops applied to the task YAML with ruamel.yaml (comments
  preserved), then `Runtime._reload` rebuilds env, IK and safety filter and emits `scene_changed`; clients refetch
  `/api/scene` and `/api/task`. Refused while recording. An edit changes `task_version`; old episodes stay
  replayable. `capture_goal` writes `relative_pose` stages + `goal_view` from current poses. Dashboard forms come
  from `GET /api/schema` (`task/schema.py`).
- **One definition per concept**: contacts = `Env.contact_flags` (kinds `env.CONTACT_KINDS`); a recorded step =
  `recorder.make_step`; one control step = `recorder.control_step` (runtime, expert, policy rollouts); file attrs =
  `recorder.episode_meta`; episode selection = `audit.why_not` / `select` / `kept_spans` (`curate` config +
  `curation.json`; the VR review card, the Episodes tab and every exporter use it); success label =
  `audit.labelled_success`; feature layout = `export.base.FeatureLayout` (stored in each export manifest, checked
  against a policy's `layout`). Manual labels go to `curation.json` only through `audit.update_curation`; episode
  files are never rewritten.
- **Export** (`export/base.py`): `run_export` takes segments from `audit.select`, re-renders cameras
  (`scene/render.py`, matte, optional visual randomization) in each episode's own scene, refuses mixed task
  versions unless `allow_mixed`, flags constant channels. Exporters are plugins registered in `EXPORTERS`; all
  LeRobot API stays in `export/lerobot.py` (export) and `policies/lerobot.py` (running a checkpoint).
- **Policy hook** (`policy.py`): `PolicyRunner` loaded lazily from `module:Class` / `file.py:Class`; the platform
  imports no ML library. Control modes teleop | policy | shared; while the policy drives the operator side is held
  at the current pose so a Grip takeover never jumps. Loading happens off the sim thread.
- **Gymnasium** (`gym_env.py`): `XRCollectEnv` wraps `Env` for every task (`xrcollect/<task>-v0`, registered on
  import). Action = per arm `[joint targets, gripper]` (= recorded `action/joint_pos`), optional safety filter,
  obs = `Env.observe()` + optional `images.<cam>`, reward sparse or progress, `reset(options={"state": s})`
  rewinds to a saved `sim/state`. Keep it task-agnostic: no task names in it.
- **stack_cups** (`tasks/stack_cups/`): mujoco_toy-project's task migrated as a plugin (`task.py`: the toy's
  start-state randomization and `check_cups_stacked`; `expert.py`: grasp below the rim; cup geometry from
  `objects/cups/cup.py`, cameras in `scenes/cups_table.yaml`).
- **Scripted expert** (`expert.py`): keyframed TCP moves on the real stack; generic pick-and-place plan from the
  stages, or a task's own `expert.py` (`def script(expert)`). `xrc validate-task` is the end-to-end check for a task.
- **Config**: `config.yaml` → pydantic models in `config.py`: platform settings only (server, teleop, ik, record,
  curate, safety toggles, control, feedback). Nothing that defines the world goes there.

## Design constraints

- Simulation-first; keep HDF5 lossless (sim state, raw XR poses and hand joints) so data can be re-retargeted
  and re-rendered.
- Non-goals unless asked: policy training/eval, real-robot backend, deformables, multi-finger retargeting, several
  operators per server, voice, RLDS exporter, reading QuestCollect episodes.
- Adding a task must not need platform code changes: if it does, extend a predicate, the snap rule or the plugin
  hooks instead of special-casing a task name.
- `cli.py` defaults `MUJOCO_GL=egl`; a harmless EGLError can print at interpreter exit.
- `pyproject.toml` overrides away `evdev` (lerobot → pynput → evdev needs a C compiler); don't remove it.
