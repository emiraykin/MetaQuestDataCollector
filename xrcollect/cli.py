"""`xrc` command line interface."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

import typer

from xrcollect.config import ROOT, load_config, rooted

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="VR data collection for simulated robot manipulation (Meta Quest 3 + MuJoCo)")


def _headless_gl() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packets are sent; selects the outbound interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


@app.command()
def serve(
    config: Optional[Path] = typer.Option(None, help="YAML config (default: config.yaml)"),
    task: Optional[str] = typer.Option(None, help="Task folder or file, e.g. tasks/duplo_tower"),
    mode: Optional[str] = typer.Option(None, help="controllers | hands"),
    https: bool = typer.Option(True, help="Also serve HTTPS for Wi-Fi (needs `xrc certs`)"),
    policy: Optional[str] = typer.Option(None, help="A PolicyRunner to drive the arms: module:Class or file.py:Class"),
    control: Optional[str] = typer.Option(None, help="teleop | policy | shared (default with --policy: shared)"),
    out: Optional[str] = typer.Option(None, help="Recording folder (default: record.out_dir)"),
):
    """Run the sim, teleop server, and web clients."""
    _headless_gl()
    import uvicorn

    from xrcollect.runtime import Runtime
    from xrcollect.server import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(config)
    if task:
        cfg.task = task
    if out:
        cfg.record.out_dir = out
    if policy:
        cfg.control.policy = policy
        cfg.control.mode = control or ("shared" if cfg.control.mode == "teleop" else cfg.control.mode)
    elif control:
        cfg.control.mode = control
    if mode:
        cfg.teleop.mode = mode
    runtime = Runtime(cfg)
    runtime.start()
    web = create_app(runtime)

    servers = [uvicorn.Server(uvicorn.Config(web, host=cfg.server.host, port=cfg.server.port, log_level="warning",
                                             ws_max_size=1 << 20))]
    cert_dir = ROOT / cfg.server.cert_dir
    cert, key = cert_dir / "cert.pem", cert_dir / "key.pem"
    if https and cert.exists() and key.exists():
        servers.append(uvicorn.Server(uvicorn.Config(web, host=cfg.server.host, port=cfg.server.https_port,
                                                     ssl_certfile=str(cert), ssl_keyfile=str(key),
                                                     log_level="warning", ws_max_size=1 << 20)))
    ip = lan_ip()
    typer.secho(f"\n  xrcollect is running: {runtime.bundle.name} ({', '.join(runtime.arms)})", fg="cyan", bold=True)
    typer.echo(f"  Dashboard      http://localhost:{cfg.server.port}/dashboard/")
    typer.echo(f"  Quest (USB)    http://localhost:{cfg.server.port}/xr/     (after `xrc setup-quest`)")
    if len(servers) > 1:
        typer.echo(f"  Quest (Wi-Fi)  https://{ip}:{cfg.server.https_port}/xr/")
    elif https:
        typer.echo("  Wi-Fi HTTPS disabled: run `xrc certs` first")
    typer.echo("")

    async def main():
        await asyncio.gather(*(s.serve() for s in servers))

    try:
        asyncio.run(main())
    finally:
        runtime.stop()


@app.command()
def certs(ip: Optional[str] = typer.Option(None, help="LAN IP to put in the certificate")):
    """Create a self-signed TLS certificate for Wi-Fi WebXR (the Quest needs HTTPS off-localhost)."""
    if not shutil.which("openssl"):
        raise typer.Exit("openssl not found (sudo apt install openssl)")
    cfg = load_config()
    ip = ip or lan_ip()
    d = ROOT / cfg.server.cert_dir
    d.mkdir(exist_ok=True)
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825",
        "-keyout", str(d / "key.pem"), "-out", str(d / "cert.pem"), "-subj", "/CN=xrcollect",
        "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost",
    ], check=True, capture_output=True)
    typer.secho(f"certificate for {ip} written to {d}", fg="green")
    typer.echo(f"Open https://{ip}:{cfg.server.https_port}/xr/ on the Quest and accept the warning once.")


@app.command("setup-quest")
def setup_quest():
    """Forward the server port to a USB-connected Quest (adb reverse). Run again after replugging."""
    adb = shutil.which("adb")
    if not adb:
        typer.secho("adb not found. Install it: sudo apt install android-tools-adb", fg="red")
        raise typer.Exit(1)
    cfg = load_config()
    out = subprocess.run([adb, "devices"], capture_output=True, text=True).stdout
    devices = [line for line in out.splitlines()[1:] if line.strip()]
    if not devices:
        typer.secho("No device. Plug in the Quest, enable developer mode, accept 'Allow USB debugging'.", fg="red")
        raise typer.Exit(1)
    if any("unauthorized" in d for d in devices):
        typer.secho("Device unauthorized: put the headset on and accept the USB debugging prompt.", fg="yellow")
        raise typer.Exit(1)
    port = cfg.server.port
    subprocess.run([adb, "reverse", f"tcp:{port}", f"tcp:{port}"], check=True)
    typer.secho(f"OK: Quest localhost:{port} -> this PC:{port}", fg="green")
    typer.echo(f"In Quest Browser open: http://localhost:{port}/xr/")


@app.command()
def codegen(check: bool = typer.Option(False, help="Only compare; exit 1 if the generated files are stale")):
    """Regenerate both sides of the wire protocol from xrcollect/wire/schema.json."""
    from xrcollect.wire.codegen import PY_OUT, TS_OUT, generate

    ok = generate(check=check)
    if check and not ok:
        typer.secho("generated protocol files are out of date: run `xrc codegen`", fg="red")
        raise typer.Exit(1)
    typer.secho("up to date" if ok else f"written: {PY_OUT.name}, {TS_OUT.name}", fg="green")


@app.command()
def tasks():
    """List the tasks in tasks/."""
    from xrcollect.task.loader import list_tasks

    for t in list_tasks():
        typer.echo(f"{t['path']:<28} {t['instruction']}")


@app.command()
def view(task: Optional[str] = typer.Argument(None, help="Task folder (default: the config's task)")):
    """Open a task's scene in MuJoCo's native viewer (sanity check, no headset)."""
    import mujoco.viewer

    from xrcollect.env import Env
    from xrcollect.task.loader import load_bundle

    env = Env(load_bundle(task or load_config().task))
    env.reset()
    mujoco.viewer.launch(env.model, env.data)


@app.command()
def replay(path: Path, speed: float = typer.Option(1.0)):
    """Play back an episode's recorded states in MuJoCo's native viewer, in the scene stored in the episode."""
    import mujoco.viewer

    from xrcollect.env import env_from_episode
    from xrcollect.recorder import read_episode

    meta, data = read_episode(path, keys=["sim/state"])
    env = env_from_episode(meta)
    if not env.scene_matches:
        typer.secho("warning: the scene fingerprint differs from the one saved with the episode", fg="yellow")
    dt = 1.0 / meta["fps"] / speed
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        for s in data["sim/state"]:
            if not v.is_running():
                break
            env.set_state(s)
            v.sync()
            time.sleep(dt)


@app.command()
def inspect(path: Path):
    """Print an episode's metadata and dataset shapes."""
    from xrcollect.recorder import read_episode

    meta, data = read_episode(path)
    for k, v in meta.items():
        if k in ("model_xml", "config_yaml", "task_yaml"):
            v = f"<{len(v)} chars>"
        typer.echo(f"{k:>16}: {v}")
    typer.echo("")
    for k in sorted(data):
        typer.echo(f"{k:<32} {str(data[k].shape):<18} {data[k].dtype}")


@app.command()
def resim(path: Path):
    """Re-simulate an episode from its first state with the recorded actions and compare every step bit for bit
    (detects nondeterminism, hidden state and physics changes)."""
    _headless_gl()
    import numpy as np

    from xrcollect.embodiment.base import split_action
    from xrcollect.env import env_from_episode
    from xrcollect.recorder import read_episode

    meta, data = read_episode(path)
    env = env_from_episode(meta)
    states = data["sim/state"]
    q, grip = split_action(data["action/joint_pos"], env.b.n_joints)
    if meta.get("source") != "scripted":
        q = env.executed_targets(states)  # what was sent, including any DART drift
    env.set_state(states[0])
    for i in range(len(states)):
        if not np.array_equal(env.get_state(), states[i]):
            typer.secho(f"diverges at step {i} of {len(states)} "
                        f"(max |diff| {np.abs(env.get_state() - states[i]).max():.3g})", fg="red")
            raise typer.Exit(1)
        env.apply_action(q[i], grip[i])
        env.step()
    typer.secho(f"{len(states)} steps replay bit-exactly", fg="green")


@app.command()
def audit(
    raw: Optional[Path] = typer.Option(None, help="Episode folder (default: record.out_dir from the config)"),
    config: Optional[Path] = typer.Option(None, help="YAML config (default: config.yaml)"),
    exclude: Optional[str] = typer.Option(None, help="Exclude this episode file from datasets"),
    include: Optional[str] = typer.Option(None, help="Undo a manual exclude"),
    note: str = typer.Option("", help="Why (with --exclude)"),
):
    """Quality report of every episode (contacts, idle time, stage timing) and which ones a dataset would use."""
    from xrcollect.audit import audit_dir, select, update_curation
    from xrcollect.env import CONTACT_KINDS

    cfg = load_config(config)
    folder = rooted(raw) if raw else cfg.out_dir
    if exclude or include:
        update_curation(folder, exclude or include, exclude=exclude is not None, note=note)
    audits = audit_dir(folder, progress=lambda name: typer.echo(f"  auditing {name} …"))
    segments, rejected = select(folder, audits, cfg.curate)
    kept: dict[str, int] = {}
    for s in segments:
        kept[s.path.name] = kept.get(s.path.name, 0) + s.num_steps
    head = "".join(f"{k:>12}" for k in CONTACT_KINDS)
    typer.echo(f"\n{'episode':<22}{'length':>8}{head}{'idle':>7}{'safety':>8}   stages at (s)         verdict")
    for name, a in audits.items():
        stages = " ".join("—" if t is None else f"{t:5.1f}" for t in a.stage_times)
        contacts = "".join(f"{a.contact_frac[k]:12.1%}" for k in CONTACT_KINDS)
        verdict = "✓ " + f"{kept[name] / a.fps:.0f} s used" if name in kept else "✗ " + rejected[name]
        typer.secho(f"{name:<22}{a.duration_s:7.1f}s{contacts}{a.idle_frac:7.0%}{a.safety_events:8d}   "
                    f"{stages:<22} {verdict}", fg=None if name in kept else "yellow")
    fps = next(iter(audits.values())).fps if audits else 50
    total = sum(a.num_steps for a in audits.values()) / fps / 60
    typer.secho(f"\n{len(kept)} of {len(audits)} episodes used: {sum(kept.values()) / fps / 60:.1f} of {total:.1f} min "
                f"({len(segments)} segments). Rules: `curate` in the config; manual excludes: --exclude.", fg="green")


@app.command()
def export(
    name: str = typer.Argument(..., help="Dataset name: written to <data>/exports/<name>"),
    raw: Optional[Path] = typer.Option(None, help="Episode folder (default: record.out_dir)"),
    exporter: str = typer.Option("hdf5", help="hdf5 | lerobot"),
    config: Optional[Path] = typer.Option(None, help="YAML config (default: config.yaml)"),
    cameras: Optional[str] = typer.Option(None, help="Comma-separated camera names; 'none' = no images"),
    size: str = typer.Option("224x224", help="Image height x width"),
    state: str = typer.Option("joint_pos,gripper_width", help="Proprioception channels: joint_pos, gripper_width, "
                                                              "joint_vel, ee_pose, ee_wrench"),
    env_state: bool = typer.Option(False, help="Also export object poses (privileged: simulation only)"),
    action: str = typer.Option("joint", help="joint | ee"),
    action_label: str = typer.Option("executed", help="recorded | executed"),
    randomize_visuals: bool = typer.Option(False, help="Vary lights, colours and camera mounts per episode"),
    annotations: bool = typer.Option(False, help="One exported episode per labelled time range"),
    allow_mixed: bool = typer.Option(False, help="Allow episodes from different task versions in one dataset"),
    fps: Optional[int] = typer.Option(None, help="Dataset frame rate (default: the recording's). Must divide it: "
                                                 "50 Hz takes export at 50, 25, 10, ..."),
):
    """Export a dataset folder through an exporter plugin (camera images are rendered now, from the states)."""
    _headless_gl()
    from xrcollect.export.base import FeatureLayout, run_export

    cfg = load_config(config)
    folder = rooted(raw) if raw else cfg.out_dir
    h, w = (int(v) for v in size.lower().split("x"))
    layout = FeatureLayout(
        cameras=None if cameras is None else ([] if cameras == "none" else cameras.split(",")),
        image_size=[h, w], state=[s for s in state.split(",") if s], env_state=env_state, action=action,
        action_label=action_label, randomize_visuals=randomize_visuals,
        segments="annotations" if annotations else "takes", fps=fps)
    out = cfg.out_dir.parent / "exports" / name

    def progress(done: int, total: int) -> None:
        typer.echo(f"\r  episode {done}/{total}", nl=done == total)

    try:
        m = run_export(folder, out, exporter, cfg.curate, layout, allow_mixed=allow_mixed, progress=progress)
    except (ValueError, FileExistsError, RuntimeError) as e:
        typer.secho(str(e), fg="red")
        raise typer.Exit(1) from e
    typer.secho(f"{m['episodes']} episodes, {m['steps']} steps -> {out}", fg="green")
    for name_, why in m["rejected"].items():
        typer.secho(f"  left out {name_}: {why}", fg="yellow")
    for feature, chans in m["constant_channels"].items():
        typer.secho(f"  never varies in {feature}: {', '.join(chans)}", fg="yellow")


@app.command("validate-task")
def validate_task(
    task: str = typer.Argument(..., help="Task folder or file"),
    n: int = typer.Option(1, help="Start layouts to try (seeds of the `validate` stream)"),
    save: Optional[Path] = typer.Option(None, help="Folder to save the scripted episodes into"),
    embodiment: Optional[str] = typer.Option(None, help="Run the task with another embodiment"),
):
    """Run the scripted expert on a task: the goal must be reachable with the real control stack."""
    _headless_gl()
    from xrcollect.env import Env, stream_seed
    from xrcollect.expert import Expert
    from xrcollect.recorder import next_episode_path
    from xrcollect.task.loader import load_bundle

    bundle = load_bundle(task, {"embodiment": embodiment} if embodiment else None)
    env = Env(bundle)
    typer.echo(f"{bundle.name}  version {bundle.version}  arms {env.arms}  objects {env.objects}  "
               f"stages {env.task.stage_names}")
    ok_all = True
    for i in range(n):
        ex = Expert(env, load_config(), record=save is not None)
        ex.reset(stream_seed("validate", i))
        try:
            ok = ex.run()
        except RuntimeError as e:
            typer.secho(str(e), fg="red")
            raise typer.Exit(1) from e
        typer.secho(f"  layout {i}: progress {env.progress()}/{len(env.task.stage_names)}  success: {ok}",
                    fg="green" if ok else "red")
        ok_all &= ok
        if save is not None:
            ex.save(next_episode_path(rooted(save)), success=ok)
    if not ok_all:
        raise typer.Exit(1)


@app.command("import-mesh")
def import_mesh(
    mesh: Path = typer.Argument(..., help="A mesh file (obj, stl, ply, glb ...)"),
    name: str = typer.Argument(..., help="Asset name, e.g. kitchen/mug -> objects/kitchen/mug.yaml"),
    scale: float = typer.Option(1.0, help="Scale to metres"),
    mass: float = typer.Option(0.1, help="kg"),
    threshold: float = typer.Option(0.05, help="CoACD concavity threshold: lower = more, tighter pieces"),
    max_hulls: int = typer.Option(16, help="Most convex pieces"),
):
    """Import an arbitrary mesh as an object asset: convex decomposition for collisions (CoACD), the decimated
    original for looks. Needs `uv sync --extra mesh`."""
    from xrcollect.meshes import import_mesh as run

    path = run(mesh, name, scale=scale, mass=mass, threshold=threshold, max_hulls=max_hulls)
    typer.secho(f"written {path.relative_to(ROOT)}; use it in a task as `asset: {name}`", fg="green")


@app.command()
def doctor():
    """Check the environment: assets, rendering, web build, protocol files, adb, network."""
    _headless_gl()
    import mujoco

    from xrcollect.env import Env
    from xrcollect.task.loader import list_tasks, load_bundle
    from xrcollect.wire.codegen import generate

    def row(ok: bool, label: str, hint: str = ""):
        typer.secho(f"  {'✓' if ok else '✗'} {label}" + (f"  — {hint}" if hint and not ok else ""),
                    fg="green" if ok else "red")

    def render_ok(model) -> bool:
        try:
            mujoco.Renderer(model, 64, 64).close()
            return True
        except Exception:  # noqa: BLE001
            return False

    cfg = load_config()
    panda = ROOT / "assets/third_party/mujoco_menagerie/franka_emika_panda/panda.xml"
    row(panda.exists(), "menagerie Panda model", "run scripts/fetch_assets.sh")
    row(generate(check=True), "generated protocol files match wire/schema.json", "xrc codegen")
    env = None
    for t in list_tasks():
        try:
            e = Env(load_bundle(t["path"]))
            row(True, f"task {t['task']} compiles ({e.model.nbody} bodies, arms {e.arms})")
            env = env or e
        except Exception as ex:  # noqa: BLE001
            row(False, f"task {t['task']}", f"{type(ex).__name__}: {ex}")
    if env is not None:
        row(render_ok(env.model), f"offscreen rendering (MUJOCO_GL={os.environ['MUJOCO_GL']})",
            "try MUJOCO_GL=osmesa or check the NVIDIA EGL drivers")
    row((ROOT / "web" / "dist" / "xr" / "index.html").exists(), "web client built",
        "./scripts/install.sh (or: cd web && npm ci && npm run build)")
    adb = shutil.which("adb")
    row(adb is not None, "adb installed", "sudo apt install android-tools-adb")
    if adb:
        out = subprocess.run([adb, "devices"], capture_output=True, text=True).stdout
        row(any("\tdevice" in line for line in out.splitlines()), "Quest connected over USB",
            "plug in + allow USB debugging, then `xrc setup-quest`")
    row((ROOT / cfg.server.cert_dir / "cert.pem").exists(), "TLS cert for Wi-Fi", "xrc certs")
    typer.echo(f"  LAN IP: {lan_ip()}")


if __name__ == "__main__":
    app()
