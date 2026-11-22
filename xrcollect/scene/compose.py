"""Builds the MuJoCo model of a bundle with MjSpec: scene (table, props, lights, cameras) + robots (adapter) + task
objects + regions + whatever the task adds (welds).

World frame: z up, +x away from the operator (towards the far table edge), +y the operator's left.

The model that runs is always compiled from the saved XML (`compile_scene`): the live scene and a scene rebuilt from
an episode's `model_xml` then go through the same path and are the same model, bit for bit.
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

from xrcollect.config import ROOT, rooted
from xrcollect.scene.constants import (CORE_CONAFFINITY, GEOM_TYPES, HIDDEN_GROUP, OPERATOR_GROUP, RIGID_SOLIMP, RIGID_SOLREF,
                                       TAPE_RGBA, TAPE_W, lookat_xyaxes, yaw_quat)
from xrcollect.task.loader import Bundle

BOX = mujoco.mjtGeom.mjGEOM_BOX


def _volume(g: dict, meshes: dict) -> float | None:
    t, s = g.get("type", "box"), g.get("size", [])
    if t == "box":
        return float(8 * np.prod(s))
    if t == "cylinder":
        return float(np.pi * s[0] ** 2 * 2 * s[1])
    if t == "sphere":
        return float(4 / 3 * np.pi * s[0] ** 3)
    if t == "capsule":
        return float(np.pi * s[0] ** 2 * 2 * s[1] + 4 / 3 * np.pi * s[0] ** 3)
    if t == "ellipsoid":
        return float(4 / 3 * np.pi * np.prod(s))
    if t == "mesh":
        return meshes.get(g["mesh"], {}).get("volume")
    return None


def add_environment(spec: mujoco.MjSpec, scene: dict) -> None:
    phys = scene.get("physics", {})
    spec.option.timestep = float(phys.get("dt", 0.002))
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Elliptic friction cones are exact (the default pyramid approximates them and lets grasps creep);
    # impratio > 1 makes friction stiffer than normal contact, which MuJoCo recommends against slip.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = float(phys.get("impratio", 10.0))
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.visual.headlight.ambient = [0.45, 0.45, 0.48]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]

    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.11, 0.12, 0.15], rgb2=[0.15, 0.16, 0.2], width=512, height=512)
    spec.add_material(name="floor_mat", textures=["", "grid"], texrepeat=[8, 8], reflectance=0.15)
    t = scene["table"]
    spec.add_material(name="table_mat", rgba=list(t.get("rgba", [0.32, 0.27, 0.23, 1])), specular=0.2)
    spec.add_material(name="leg_mat", rgba=[0.2, 0.2, 0.22, 1])

    wb = spec.worldbody
    lights = scene.get("lights") or [
        {"name": "key", "pos": [0.4, 0.0, 3.0], "dir": [0, 0, -1], "diffuse": [0.45, 0.45, 0.45], "shadow": True},
        {"name": "fill", "pos": [-1.0, 1.0, 2.5], "dir": [0.5, -0.3, -1], "diffuse": [0.3, 0.3, 0.35]},
    ]
    for i, li in enumerate(lights):
        wb.add_light(name=li.get("name", f"light{i}"), pos=li["pos"], dir=li.get("dir", [0, 0, -1]),
                     diffuse=li.get("diffuse", [0.4, 0.4, 0.4]), castshadow=bool(li.get("shadow", False)))
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.05], material="floor_mat")

    (cx, cy), (hx, hy), th = t["center"], t["half"], float(t["height"])
    table = wb.add_body(name="table", pos=[cx, cy, 0])
    table.add_geom(name="table_top", type=BOX, size=[hx, hy, 0.02], pos=[0, 0, th - 0.02], material="table_mat",
                   friction=[float(t.get("friction", 1.0)), 0.005, 0.0001], solref=RIGID_SOLREF, solimp=RIGID_SOLIMP)
    for sx in (-1, 1):
        for sy in (-1, 1):
            table.add_geom(type=BOX, size=[0.025, 0.025, (th - 0.04) / 2],
                           pos=[sx * (hx - 0.05), sy * (hy - 0.05), (th - 0.04) / 2], material="leg_mat",
                           contype=0, conaffinity=0)
    for i, p in enumerate(scene.get("props") or []):  # static geometry: walls, bins, fixtures made of primitives
        collide = bool(p.get("collide", True))
        yaw = np.deg2rad(p.get("yaw_deg", 0.0))
        wb.add_geom(name=p.get("name", f"prop{i}"), type=GEOM_TYPES[p.get("type", "box")], size=p["size"],
                    pos=p["pos"], quat=yaw_quat(yaw), rgba=list(p.get("rgba", [0.5, 0.5, 0.5, 1])),
                    contype=int(collide), conaffinity=int(collide), solref=RIGID_SOLREF, solimp=RIGID_SOLIMP)
    for name, c in (scene.get("cameras") or {}).items():
        wb.add_camera(name=name, pos=c["pos"], xyaxes=lookat_xyaxes(c["pos"], c["lookat"]),
                      fovy=float(c.get("fovy", 60)))
        # Operator-only wireframe pyramid (group 5): visualization of the camera's view frustum
        pos = np.asarray(c["pos"], float)
        lookat = np.asarray(c["lookat"], float)
        fovy = float(c.get("fovy", 60))
        xyaxes = lookat_xyaxes(list(pos), list(lookat))
        r, u = np.asarray(xyaxes[:3]), np.asarray(xyaxes[3:])
        f = lookat - pos
        f /= np.linalg.norm(f)
        depth = 0.15
        half = np.tan(np.deg2rad(fovy) / 2) * depth
        base_c = pos + f * depth
        corners = [base_c + sx * half * r + sy * half * u for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        edges = [(pos, c) for c in corners] + [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        for i, (p, q) in enumerate(edges):
            wb.add_geom(name=f"{name}_view_{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*p, *q],
                        size=[0.002, 0, 0], rgba=[0.1, 0.3, 0.9, 1], group=OPERATOR_GROUP, contype=0,
                        conaffinity=0, mass=0)


def add_regions(spec: mujoco.MjSpec, regions: dict, th: float) -> None:
    """Four tape strips (visual only, group 1: the cameras see them) outlining each region from the inside."""
    t = TAPE_W / 2
    for name, r in regions.items():
        if not r.get("tape", True):
            continue
        (cx, cy), (hx, hy) = r["center"], r["half"]
        strips = [([cx, cy + s * (hy - t)], [hx, t]) for s in (-1, 1)] + \
                 [([cx + s * (hx - t), cy], [t, max(hy - 2 * t, 1e-4)]) for s in (-1, 1)]
        for i, (xy, half) in enumerate(strips):
            spec.worldbody.add_geom(name=f"region_{name}_{i}", type=BOX, size=[*half, 0.0002],
                                    pos=[*xy, th + 0.0002], rgba=list(TAPE_RGBA), group=1, contype=0, conaffinity=0)


def add_object(spec: mujoco.MjSpec, name: str, o: dict, index: int, th: float) -> None:
    """One task object. Body frame at the bottom centre; free unless `fixed`. Uniform density over the geoms gives
    the stated mass and a realistic inertia."""
    pos = list(o.get("pos", [0.3 + 0.1 * index, 0.0]))
    pos = [pos[0], pos[1], pos[2] if len(pos) > 2 else th]
    quat = yaw_quat(np.deg2rad(o.get("yaw_deg", 0.0)))
    if o.get("mjcf"):  # an MJCF fragment (drawer, door, articulated fixture), attached as it is
        child = mujoco.MjSpec.from_file(str(rooted(o["mjcf"])))
        frame = spec.worldbody.add_frame(pos=pos, quat=quat)
        spec.attach(child, prefix=f"{name}_", frame=frame)
        return
    meshes = o.get("meshes", {})
    for mname, m in meshes.items():
        full = f"{name}__{mname}"
        if m.get("vertices") is not None:
            spec.add_mesh(name=full, uservert=list(np.ravel(m["vertices"])),
                          inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX)
        else:
            scale = m.get("scale", [1, 1, 1])
            spec.add_mesh(name=full, file=str(rooted(m["file"])), scale=[scale] * 3 if np.isscalar(scale) else scale)
    rgba = list(o.get("rgba", [0.7, 0.7, 0.7, 1]))
    spec.add_material(name=f"{name}_mat", rgba=rgba, specular=float(o.get("specular", 0.4)),
                      shininess=float(o.get("shininess", 0.6)))
    body = spec.worldbody.add_body(name=name, pos=pos, quat=quat)
    if not o.get("fixed"):
        body.add_freejoint(name=f"{name}_free")
    size = o.get("size", [0.04, 0.04, 0.04])
    geoms = o.get("geoms") or [{"type": "box", "size": [size[0] / 2, size[1] / 2, size[2] / 2],
                                "pos": [0, 0, size[2] / 2]}]
    solid = [g for g in geoms if not g.get("visual")]
    vols = [_volume(g, meshes) for g in solid]
    mass = float(o.get("mass", 0.05))
    density = mass / sum(vols) if solid and all(v is not None for v in vols) else None
    friction = [float(o.get("friction", 0.8)), 0.005, 0.0001]
    for g in geoms:
        kw: dict = {"type": GEOM_TYPES[g.get("type", "box")], "pos": g.get("pos", [0, 0, 0])}
        if g.get("type") == "mesh":
            kw["meshname"] = f"{name}__{g['mesh']}"
        else:
            kw["size"] = g["size"]
        if g.get("quat"):
            kw["quat"] = g["quat"]
        if g.get("rgba"):
            kw["rgba"] = g["rgba"]
        else:
            kw["material"] = f"{name}_mat"
        if g.get("visual"):       # looks only (e.g. a detailed mesh over simple collision shapes)
            body.add_geom(contype=0, conaffinity=0, group=1, mass=0, **kw)
            continue
        if g.get("hidden"):       # collision only (e.g. convex pieces under a visual mesh)
            kw.pop("material", None)
            kw.update(group=HIDDEN_GROUP, rgba=[0, 0, 0, 0])
        if density is not None:
            kw["density"] = density
        else:
            kw["mass"] = mass / len(solid)
        # priority=1: object contact params win against the table; finger pads (priority 2) win against objects.
        body.add_geom(condim=4, friction=friction, solref=RIGID_SOLREF, solimp=RIGID_SOLIMP, priority=1, **kw)
    core = o.get("core")
    if core:  # an invisible solid filling a hollow object that only the robot hits
        body.add_geom(name=f"{name}_core", type=BOX, size=core["size"], pos=core["pos"], density=0,
                      group=HIDDEN_GROUP, rgba=[0, 0, 0, 0], contype=0, conaffinity=CORE_CONAFFINITY, condim=4,
                      friction=friction, solref=RIGID_SOLREF, solimp=RIGID_SOLIMP, priority=1)


def build_spec(bundle: Bundle, adapter, task=None, with_objects: bool = True) -> mujoco.MjSpec:
    """The scene spec: with_objects=False gives the robots-only model the IK and the safety filter use."""
    spec = mujoco.MjSpec()
    spec.modelname = bundle.embodiment.get("name", "robot")
    add_environment(spec, bundle.scene)
    adapter.attach(spec)
    for base in adapter.base_bodies():
        spec.add_exclude(bodyname1="table", bodyname2=base)
    if with_objects:
        th = bundle.table_height
        add_regions(spec, bundle.task.get("regions", {}), th)
        for i, (name, o) in enumerate(bundle.objects.items()):
            add_object(spec, name, o, i, th)
        if task is not None:
            task.build(spec, bundle)
    return spec


_FILE_ATTR = re.compile(r'file="([^"]+)"')


def load_model(xml: str, asset_dirs: list[Path] | None = None) -> mujoco.MjModel:
    """Compile a saved MJCF (an episode's `model_xml`). Files it names (meshes, textures) are looked up by name in
    the asset folders: the robot's, the object library and any folder given."""
    dirs = [Path(d) for d in (asset_dirs or [])] + [ROOT / "objects", ROOT / "assets"]
    assets: dict[str, bytes] = {}
    meshdir = re.search(r'meshdir="([^"]*)"', xml)
    if meshdir:
        dirs.insert(0, Path(meshdir.group(1)))
    index: dict[str, Path] | None = None
    for ref in set(_FILE_ATTR.findall(xml)):
        p = Path(ref)
        found = p if p.is_absolute() and p.exists() else next((d / ref for d in dirs if (d / ref).exists()), None)
        if found is None:
            if index is None:  # by bare file name, anywhere under the asset folders
                index = {}
                for d in dirs:
                    if d.is_dir():
                        for f in d.rglob("*"):
                            if f.is_file():
                                index.setdefault(f.name, f)
            found = index.get(p.name)
        if found is None:
            raise FileNotFoundError(f"the scene needs {ref}, which is in none of {[str(d) for d in dirs]} "
                                    "(run scripts/fetch_assets.sh)")
        assets[ref] = found.read_bytes()
    # Absolute and relative directories in the compiler tag would be searched before `assets`: strip them.
    xml = re.sub(r'\s(meshdir|texturedir|assetdir)="[^"]*"', "", xml)
    return mujoco.MjModel.from_xml_string(xml, assets)


def compile_scene(bundle: Bundle, adapter, task=None) -> tuple[mujoco.MjModel, str]:
    """(model, the XML it was compiled from). The XML is what episodes store."""
    spec = build_spec(bundle, adapter, task)
    spec.compile()
    xml = spec.to_xml()
    return load_model(xml, getattr(adapter, "asset_dirs", [])), xml
