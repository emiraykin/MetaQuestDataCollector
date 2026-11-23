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
- **Training on, or resuming a stopped run:** start from a checkpoint's saved config; `--steps` is the new total.
  The run continues in the same `--output_dir`, with the optimizer state and step count of that checkpoint:

  ```bash
  uv run lerobot-train --resume=true --steps=200000 \
    --config_path=outputs/train/act_cups_v2_25fps/checkpoints/100000/pretrained_model/train_config.json
  ```

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
  --policy-arg checkpoint=outputs/train/act_cups_v2_25fps/checkpoints/100000 \
  --policy-arg n_action_steps=50 \
  --n 50 --out outputs/eval/act_cups_v2_25fps/100000.json
```

It prints one line per episode and the rate at the end:

```text
act_cups_v2_25fps@100000 on stack_cups (version 806152417e3f62d3): 50 episodes, 40 s each
    0  seed  384246851  fail      40.0 s  stages 0/1
    4  seed  602975432  success   14.1 s  stages 1/1
  ...
success 25/50 (50%), mean time 18.9 s
```

`stages` is the furthest stage reached (a fail at 1/1 was stacked but never let go or came to rest); `mean time` is
over the successful episodes. A warning that the policy "was trained on task versions ..." means the task changed
since the export: the rollout still runs, but check that the change doesn't alter what the policy sees.

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
    --policy-arg checkpoint=outputs/train/act_cups_v2_25fps/checkpoints/$c --policy-arg n_action_steps=50 \
    --n 50 --out outputs/eval/act_cups_v2_25fps/$c.json
done
```

To watch what a policy does, keep the episodes and videos of a run:

```bash
uv run xrc rollout xrcollect.policies.lerobot:LeRobotRunner \
  --policy-arg checkpoint=outputs/train/act_cups_v2_25fps/checkpoints/100000 --policy-arg n_action_steps=50 \
  --n 50 --save outputs/eval/best/episodes --video outputs/eval/best/videos
uv run xrc replay outputs/eval/best/episodes/episode_000003.hdf5    # one rollout in the viewer
uv run xrc serve --out outputs/eval/best/episodes                    # all of them in the Episodes tab
```

The videos are named by result (`ep003_fail.mp4`), so `ls outputs/eval/best/videos/*_fail.mp4` lists the failures.

### Example: `act_cups_v2_25fps` (108 training episodes, ACT)

Success over 50 episodes; "every" is the time between two predictions (`n_action_steps` / fps):

| Policy | every | success |
| --- | --- | --- |
| 25 fps, 100k, `n_action_steps=50` | 2 s | **50%** (47% over 100 episodes) |
| 25 fps, 100k, `n_action_steps=100` | 4 s | 34% |
| 25 fps, 120k, `n_action_steps=50` (resumed to 200k) | 2 s | 36% |
| 50 fps (`cups_v2_50fps`, same settings), 80k, `n_action_steps=100` | 2 s | 44% |

- **How much of each chunk to run matters:** a new chunk every 2 s re-plans often enough to recover from a missed
  grasp while still executing long, smooth stretches. The full 4 s chunk does worse.
- **The frame rate doesn't:** at the same 2 s between predictions, 25 fps and 50 fps are within noise. The 50 fps
  `eval_loss` is lower (0.29 against 0.39) only because a 100-step chunk spans 2 s instead of 4 s.
- **Training longer doesn't help:** from 100k to 200k, training L1 kept falling (0.061 → 0.045) while `eval_loss`
  stayed at 0.39 and the success rate did not rise. With ~100 demonstrations, more (or more varied) takes, or DAgger
  corrections in shared mode, are the next lever, not more steps.
- **`eval_loss` doesn't pick checkpoints:** it hardly moves between checkpoints whose success rates differ, so
  choose by rollout.
- **Failures:** usually a grasp that closes beside the cup, or a cup knocked over.

**When a task's success check changes, re-run the rollouts.** An earlier `stack_cups` check counted a cup turned
clockwise about its own axis as tipped over, which made every rate above look 5–15 points lower. Old rollout files
replay with the check they were recorded with (each episode carries its own task), so re-scoring them needs the
current task (`Env(load_bundle(task))` + `set_state`).

Twenty episodes is a coarse measure: 9/20 and 12/20 are within noise of each other, and with 50 a rate near 50% is
only known to about ±14%. Use 50 or more to decide between close checkpoints. GPU inference isn't bit-exact, so
the same seed can end differently from run to run.

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
  --policy-arg checkpoint=outputs/train/act_cups_v2_25fps/checkpoints/100000 --policy-arg n_action_steps=50 \
  --control shared
```

Or load it from the dashboard's Policy tab: ref `xrcollect.policies.lerobot:LeRobotRunner`, args
`{"checkpoint": "outputs/train/act_cups_v2_25fps/checkpoints/100000", "n_action_steps": 50}`. In shared mode the
policy drives until you press Grip. The steps you drive are recorded as operator steps, and the next export keeps
only those, so your corrections become new training data (DAgger).

## 6. Share on the Hugging Face Hub

Log in once with a token that has write access (huggingface.co/settings/tokens), or through the browser:

```bash
uv run hf auth login
```

A dataset goes up with LeRobot's own upload, which also writes a dataset card and tags the LeRobot format version.
The repo name doesn't have to match the export's local `repo_id`:

```bash
uv run python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; \
  D('<user>/cups_v2_25fps', root='data/exports/cups_v2_25fps').push_to_hub(private=True)"
```

It uploads `data/`, `meta/`, `videos/` and `xrcollect_manifest.json` (the runner needs the manifest). Others then
load it with `LeRobotDataset('<user>/cups_v2_25fps')`. A trained model is the checkpoint's `pretrained_model`
folder:

```bash
uv run hf upload <user>/act_cups_v2_25fps \
  outputs/train/act_cups_v2_25fps/checkpoints/100000/pretrained_model --repo-type model --private
```

The runner takes local folders only: download a model with `uv run hf download <user>/act_cups_v2_25fps --local-dir
models/act_cups`, then pass `checkpoint=models/act_cups` and the export it was trained on, `dataset=<local export
folder>` (the path saved in its `train_config.json` is the trainer's).
`private` keeps either repo visible to you only; change it on the Hub when you want to publish.

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
