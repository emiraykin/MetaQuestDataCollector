# Training and evaluating a policy

From recorded takes to a policy that drives the arms: export a LeRobot dataset, train with LeRobot, measure the
success rate in sim, then drive with it in VR. xrcollect does no training itself. It hands LeRobot a dataset and
loads the checkpoint that comes back. The examples use the cup task and ACT; any task works the same way.

```text
data/episodes/*.hdf5 --xrc export--> data/exports/<set> --lerobot-train--> outputs/train/<run>/checkpoints/<step>
                                                                                  |
                                              xrc rollout  <----------------------+---->  xrc serve --policy
                                          (success rate, headless)                       (VR, shared control)
```

## 1. Install

```bash
uv sync --extra lerobot --extra train    # LeRobot + torch; name every extra you want on each uv sync
```

`lerobot` is enough to export and to run a checkpoint (`xrc rollout`, `xrc serve --policy`). `train` adds what
`lerobot-train` needs.

## 2. Export

```bash
uv run xrc export cups_v2_25fps --exporter lerobot --cameras left,right,cam_wrist --fps 25
```

- **Episodes:** only the takes the curate rules keep (successful, not rejected in review), the same selection as
  the Episodes tab.
- **Cameras:** images are rendered now from the recorded states, at 224×224 by default.
- **Frame rate:** `--fps` keeps every n-th step of the 50 Hz recording. It must divide 50: 50, 25, 10, ...
- **Manifest:** `xrcollect_manifest.json` in the export holds the feature layout (state channels, cameras, image
  size, fps). The runner in step 5 reads it back, so the policy sees the same inputs at run time.

## 3. Train

```bash
uv run lerobot-train \
  --dataset.repo_id=xrcollect/cups_v2_25fps --dataset.root=data/exports/cups_v2_25fps \
  --dataset.eval_split=0.1 --eval_steps=2000 \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=8 --steps=100000 --save_freq=20000 \
  --output_dir=outputs/train/act_cups_v2_25fps --job_name=act_cups_v2_25fps
```

| Flag | What it does |
| --- | --- |
| `--dataset.eval_split=0.1` | holds out 10% of the episodes for validation: whole episodes, the last ones by index (not random) |
| `--eval_steps=2000` | logs `eval_loss` on the held-out episodes every 2000 steps |
| `--policy.push_to_hub=false` | without it, LeRobot uploads the model to the Hugging Face Hub |
| `--save_freq=20000` | a checkpoint every 20k steps, in `outputs/train/<run>/checkpoints/<step>/` |
| `--output_dir` | must be new for every run (LeRobot refuses an existing one unless resuming) |

- **Tracking:** add `--trackio.enable=true`, then `uv run trackio show --project lerobot`. Runs stay local.
- **Speed:** on an 8 GB laptop RTX 4070, ACT with three 224×224 cameras at batch size 8 runs at about 12 steps/s
  with 1.4 GB of GPU memory, so 100k steps take about 2 h 20 min.
- **Running it in the background:** append `> outputs/logs/<run>.log 2>&1`, then check the loss with
  `grep eval_loss outputs/logs/<run>.log`.

### Reading the losses (ACT)

| Name | Meaning |
| --- | --- |
| `l1_loss` | mean absolute error between the predicted and the recorded actions, over the whole chunk (`chunk_size` = 100 steps, 4 s at 25 fps) and every action dimension, in **normalized** units (each dimension scaled to std 1). 0.1 = off by a tenth of that joint's spread. |
| `kld_loss` | KL term of ACT's VAE: keeps the "style" latent `z` close to a standard normal. Drops to ~0 early. |
| `loss` | what is optimized: `l1_loss + 10 × kld_loss` |
| `eval_loss` | L1 on the held-out episodes with `z = 0`, as at run time (no KL term). **Compare it with `l1_loss`, not with `loss`.** |
| `grdn` | gradient norm before clipping (clipped at 10) |

`eval_loss` is **open-loop**: every frame gets the recorded observation, so it never shows the errors a policy
makes by driving itself into states it hasn't seen. Use it to spot overfitting: training L1 keeps falling while
`eval_loss` stays flat or rises. Don't use it to decide whether a policy works. With ACT on ~100 episodes,
`eval_loss` usually flattens after 10–20k steps while closed-loop success can keep changing, so test each
checkpoint in sim (next step).

## 4. Evaluate in sim: `xrc rollout`

```bash
uv run xrc rollout xrcollect.policies.lerobot:LeRobotRunner \
  --policy-arg checkpoint=outputs/train/act_cups_v2_25fps/checkpoints/040000 \
  --n 20 --out outputs/eval/act_cups_v2_25fps/040000.json
```

- **What it does:** runs a policy runner headless, with no server or headset, on `--n` start layouts and prints
  success or failure per episode and the success rate.
- **Start layouts:** they come from the `eval` seed stream, so they never match a recorded take and are the same
  for every checkpoint. That makes runs comparable.
- **Same loop as live:** the policy's targets go through the same safety filter and control step as in
  `xrc serve`.
- **End of an episode:** at the task's success check, or after `--seconds` (default 40).

| Option | |
| --- | --- |
| `--policy-arg key=value` | constructor arguments of the runner (repeatable) |
| `--task` | task folder (default: the config's `task`) |
| `--n`, `--seconds` | episodes, time limit per episode |
| `--save <dir>` | save every rollout as an episode file: watch with `xrc replay <file>` or `xrc serve --out <dir>` (Episodes tab) |
| `--video <dir>` | one MP4 per episode, `ep000_success.mp4` / `ep001_fail.mp4`: the scene cameras side by side at 25 fps (needs `ffmpeg`) |
| `--video-cameras a,b` | which cameras go in the videos (default: all of the scene's) |
| `--out <file.json>` | per-episode results (seed, success, time, stages reached) |
| `--no-safety` | skip the safety filter |

To compare checkpoints:

```bash
for c in 020000 040000 060000 080000 100000; do
  uv run xrc rollout xrcollect.policies.lerobot:LeRobotRunner \
    --policy-arg checkpoint=outputs/train/act_cups_v2_25fps/checkpoints/$c \
    --n 20 --out outputs/eval/act_cups_v2_25fps/$c.json
done
```

### Example: `act_cups_v2_25fps` (108 training episodes, ACT, 100k steps)

| Checkpoint | 20k | 40k | 60k | 80k | 100k |
| --- | --- | --- | --- | --- | --- |
| `eval_loss` | 0.404 | 0.392 | 0.396 | 0.393 | 0.393 |
| success (20 episodes) | 10% | 25% | 20% | 30% | 15% |
| success (50 episodes) | 12% | 22% | **26%** | 18% | 20% |
| success (50, `n_action_steps=50`) | 12% | 22% | 18% | 36% | **38%** |
| success (50, `n_action_steps=20`) | 8% | 12% | 12% | 18% | 28% |

How much of each chunk to execute matters: over all five checkpoints, `n_action_steps=50` (a new chunk every 2 s)
reached 63/250, the full chunk (100) 49/250 and `n_action_steps=20` 39/250; at 80k `n_action_steps=10` scored 0%.
The gain from 50 is in the trained-out checkpoints: 80k + 100k went from 19/100 to 37/100 (p ≈ 0.005). Half a chunk
re-plans often enough to recover from a missed grasp, while still executing long, smooth stretches. `eval_loss` hardly moves while the success rate ranges from 10% to 30%, which is why checkpoints are chosen by
rollout. With 50 episodes, a ~20% rate is only known to about ±11%. So 40k–100k are indistinguishable here: the
"best" checkpoint at 20 episodes (80k, 30%) was 18% at 50. The usual failures are a grasp that closes beside the cup, and a cup placed on the other one that doesn't
quite trigger the stack check. More demonstrations would help most.

### Example: 25 fps against 50 fps

The same takes exported at 50 fps (`cups_v2_50fps`) and trained with identical settings. Success over 50 episodes;
"every" is the time between two predictions (`n_action_steps` / fps):

| `n_action_steps` | 25 fps: every | 25 fps: success | 50 fps: every | 50 fps: success |
| --- | --- | --- | --- | --- |
| 100 | 4 s | 49/250 (19.6%) | **2 s** | **66/250 (26.4%)** |
| 50 | **2 s** | **63/250 (25.2%)** | 1 s | 39/250 (15.6%) |
| 20 | 0.8 s | 39/250 (15.6%) | 0.4 s | 8/250 (3.2%) |

What counts is the time between predictions, not the frame rate: re-planning every 2 s is best at both rates and
gives the same result; more often or less often is worse. At equal timing the 50 fps policy is no better, and its
`eval_loss` (0.29 against 0.39) is lower only because a 100-step chunk spans 2 s instead of 4 s. Best single
checkpoints: 38% (25 fps 100k at 50 steps, 50 fps 80k at 100 steps).

### Example: training longer

`act_cups_v2_25fps` resumed from 100k to 200k (`--config_path=<ckpt>/pretrained_model/train_config.json
--resume=true --steps=200000`), each checkpoint tested over 50 episodes at `n_action_steps=50`:

| Checkpoint | 100k | 120k | 140k | 160k | 180k | 200k |
| --- | --- | --- | --- | --- | --- | --- |
| `eval_loss` | 0.393 | 0.394 | 0.392 | 0.391 | 0.390 | 0.390 |
| success | 38% | 36% | 30% | 20% | 32% | 26% |

The second 100k steps bought nothing: 120k–200k reached 72/250 (29%), against 37/100 for 80k + 100k. Training L1
keeps falling, `eval_loss` stays flat, and success drifts within noise. With ~100 demonstrations the policy has
learned what the data holds; more (or more varied) takes, or DAgger corrections in shared mode, are the next lever,
not more steps.

### Example: a wrong success check

The numbers above were measured with a `stack_cups` check that took a cup turned clockwise about its own axis by more
than ~36° for a tipped-over one (a sign error in its quaternion test), so cleanly nested cups often counted as
failures; it also let a cup leaning 20–36° on the rim pass. The check now measures the tilt of each cup's axis
(`params.stack.max_tilt_deg: 15`). The recorded takes were unaffected (no nested take was labelled failed): only
the rollout rates were too low. Re-measured over 50 episodes:

| Policy | old check | fixed check |
| --- | --- | --- |
| 25 fps 100k, `n_action_steps=50` | 38%, 44% | **50%** |
| 50 fps 80k, `n_action_steps=100` | 38%, 32% | 44% |
| 25 fps 100k, `n_action_steps=100` | 20% | 34% |
| 25 fps 120k, `n_action_steps=50` | 36% | 36% |

The ranking holds: 25 fps 100k with a new chunk every 2 s is the best policy. When a task's success check changes,
re-run the rollouts: old rollout files replay with the check they were recorded with (each episode carries its
own task), so re-scoring them needs the current task (`Env(load_bundle(task))` + `set_state`).

Twenty episodes is a coarse measure: 9/20 and 12/20 are within noise of each other. Use 50 or more to decide
between close checkpoints. GPU inference isn't bit-exact, so the same seed can end differently from run to run.

### Runner options (`LeRobotRunner`)

| Arg | |
| --- | --- |
| `checkpoint` | a run folder (uses `checkpoints/last`), one checkpoint folder, or its `pretrained_model` |
| `dataset` | the export it was trained on; default: read from the checkpoint's `train_config.json` |
| `device` | `cuda` when available, else `cpu` |
| `n_action_steps` | ACT executes this many actions of each predicted chunk before predicting again (default: the trained value, 100). Lower re-plans more often and reacts sooner to a missed grasp. |

The runner works with any LeRobot policy type that `select_action` drives (ACT, Diffusion, ...). It needs
joint-space actions (`--action joint`, the export default). If the task was edited after the export, it warns
that the task version differs from the one it was trained on.

## 5. Drive with it in VR

```bash
uv run xrc serve --policy xrcollect.policies.lerobot:LeRobotRunner \
  --policy-arg checkpoint=outputs/train/act_cups_v2_25fps/checkpoints/040000 --control shared
```

Or load it from the dashboard's Policy tab: ref `xrcollect.policies.lerobot:LeRobotRunner`, args
`{"checkpoint": "outputs/train/act_cups_v2_25fps/checkpoints/040000"}`. In shared mode the policy drives until you
press Grip. The steps you drive are recorded as operator steps, and the next export keeps only those, so your
corrections become new training data (DAgger).

## How the runner matches training

- **State:** built from `Env.observe()` with the same function the exporter uses
  (`export.base.observation_arrays`).
- **Images:** rendered by the same `CameraRenderer` (matte, no shadows, groups 0–2) at the export's size, as
  float RGB 0–1, channels first.
- **Normalization:** LeRobot's own pre/post-processors saved with the checkpoint normalize the inputs and
  un-normalize the actions.
- **Frame rate:** at 25 fps in a 50 Hz scene, the policy is asked every second control step and its action is held
  for both, the way the export paired each frame with the action that followed it.

To check this pipeline on your own checkpoint, set the sim to recorded training states and compare the runner's
first action with the exported one. The error should be close to the training `l1_loss`, or lower. For the
checkpoint above it was about 0.03 std per joint.

## Writing your own runner

Any class with `reset(env)` and `act(env) -> (joint targets, grippers 0..1)` works with both `xrc rollout` and
`xrc serve --policy`. See `xrcollect/policy.py` for the protocol and `xrcollect/policies/lerobot.py` as an example.
