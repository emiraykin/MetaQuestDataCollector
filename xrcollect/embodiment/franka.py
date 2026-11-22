"""Franka Emika Panda (MuJoCo Menagerie) with the Franka Hand: any number of arms, each at its own mount pose.

Control like the real robot: position-controlled joints with gravity compensation; the gripper is a stiff servo on
the finger tendon with a capped force and the real hand's speed as a rate limit on the target.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from xrcollect.config import rooted
from xrcollect.embodiment.base import Bound
from xrcollect.scene.constants import OPERATOR_GROUP, RIGID_SOLIMP, RIGID_SOLREF, ROBOT_CONTYPE, lookat_xyaxes

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ("finger_joint1", "finger_joint2")
FINGERS = ("left_finger", "right_finger")
TCP_OFFSET = 0.1034        # flange-to-fingertip-centre along the hand z axis
GRIPPER_CTRL_MAX = 255.0   # Menagerie's ctrl range: 0 = closed, 255 = fully open
# Gripper like the real Franka Hand (70 N continuous, 140 N peak, 50 mm/s per finger, 80 mm stroke): a stiff
# position servo on the finger tendon whose force is capped at the configured grasp force.
FINGER_TRAVEL = 0.04       # per finger
GRIP_KP = 28000.0          # N/m on the tendon (= mean finger travel); saturates within ~5 mm of contact
GRIP_KD = 200.0            # N s/m, about critical damping for the fingers
WRIST_CAM_POS = (0.1, 0.0, 0.0)  # wrist camera in the hand frame: 10 cm out to the side of the hand
LINK_BODIES = ("link4", "link5", "link6", "link7", "hand")
DEFAULT_HOME = (0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785)


class FrankaAdapter:
    n_joints = 7

    def __init__(self, emb: dict):
        self.emb = emb
        self.name = emb.get("name", "franka")
        self.arms = list(emb["arms"])
        self.model_path = rooted(emb.get("model", "assets/third_party/mujoco_menagerie/franka_emika_panda/panda.xml"))
        self.home_q = np.tile(np.asarray(emb.get("home_q", DEFAULT_HOME), float), len(self.arms))
        self.max_joint_vel = float(emb.get("max_joint_vel", 2.5))
        grip = emb.get("gripper", {})
        self.grip_force = float(grip.get("force", 40.0))
        self.finger_speed = float(grip.get("finger_speed", 0.05))

    @property
    def asset_dirs(self) -> list[Path]:
        return [self.model_path.parent / "assets"]

    # -- model -------------------------------------------------------------------------------------
    def attach(self, spec: mujoco.MjSpec) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"{self.model_path} is missing: run scripts/fetch_assets.sh")
        for arm, a in self.emb["arms"].items():
            self._add_arm(spec, arm, a)

    def _add_arm(self, spec: mujoco.MjSpec, arm: str, a: dict) -> None:
        panda = mujoco.MjSpec.from_file(str(self.model_path))
        for body in panda.bodies:
            body.gravcomp = 1.0  # the real Franka controller compensates gravity internally
        # Gripper: force = kp * (target - travel) - kd * speed on the "split" tendon (mean finger travel). Each
        # finger gets half the tendon force, so the cap is 2x the grasp force (squeeze between the fingers).
        grip = panda.actuator("actuator8")
        grip.gainprm = [GRIP_KP * FINGER_TRAVEL / GRIPPER_CTRL_MAX, *grip.gainprm[1:]]
        grip.biasprm = [0.0, -GRIP_KP, -GRIP_KD, *grip.biasprm[3:]]
        grip.forcerange = [-2 * self.grip_force, 2 * self.grip_force]
        for g in panda.geoms:
            if g.contype:
                g.solref, g.solimp = RIGID_SOLREF, RIGID_SOLIMP
                g.contype = ROBOT_CONTYPE
        # Grippy finger pads; priority 2 so their friction/stiffness win over the objects.
        # Menagerie's fingertip is a big pad plus four small boxes standing 1.5 mm proud of it with 4-5 mm slots in
        # between; an edge pressed into such a slot gets clamped by opposing contacts and the object stays on the
        # open finger. One flat box over the same area (same inner face, so grasp width is unchanged).
        for f in FINGERS:
            for g in panda.body(f).geoms:
                if g.classname.name == "fingertip_pad_collision_1":
                    g.size, g.pos = [0.0085, 0.00475, 0.0085], [0, 0.00475, 0.0445]
                elif g.classname.name.startswith("fingertip_pad_collision"):
                    panda.delete(g)
                    continue
                if g.contype:
                    g.friction = [1.5, 0.01, 0.001]
                    g.condim, g.priority, g.solref = 4, 2, [0.004, 1]
        hand = panda.body("hand")
        hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.006, 0.006, 0.006], rgba=[0, 1, 1, 0.0], group=4)
        # Grasp-point marker for the operator (headset + dashboard only, not the recorded cameras): the point
        # midway between the finger pads that the arm is steered by. The web client colours it per arm.
        hand.add_geom(name="tcp_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0, 0],
                      pos=[0, 0, TCP_OFFSET], rgba=[1, 1, 1, 1], group=OPERATOR_GROUP, contype=0, conaffinity=0,
                      mass=0)
        cam_pos = list(WRIST_CAM_POS)
        target = [0, 0, 0.16]
        xyaxes = lookat_xyaxes(cam_pos, target, up=[1, 0, 0])
        hand.add_camera(name="wrist_cam", pos=cam_pos, xyaxes=xyaxes, fovy=80)
        # Operator-only wireframe pyramid (group 5, no collisions): the wrist camera's view frustum, 12 cm deep, so
        # you can see where it points. The recorded cameras render groups 0-2 and don't see it.
        r, u = np.asarray(xyaxes[:3]), np.asarray(xyaxes[3:])
        f = np.asarray(target, float) - np.asarray(cam_pos, float)
        f /= np.linalg.norm(f)
        depth = 0.12
        half = np.tan(np.deg2rad(80) / 2) * depth
        apex = np.asarray(cam_pos, float)
        base_c = apex + f * depth
        corners = [base_c + sx * half * r + sy * half * u for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        edges = [(apex, c) for c in corners] + [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        for i, (p, q) in enumerate(edges):
            hand.add_geom(name=f"wrist_cam_view_{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*p, *q],
                          size=[0.002, 0, 0], rgba=[0.1, 0.3, 0.9, 1], group=OPERATOR_GROUP, contype=0,
                          conaffinity=0, mass=0)
        yaw = np.deg2rad(a.get("yaw_deg", 0.0))
        frame = spec.worldbody.add_frame(pos=list(a["pos"]), quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        spec.attach(panda, prefix=f"{arm}_", frame=frame)
        spec.camera(f"{arm}_wrist_cam").name = self.wrist_cameras()[arm]

    def wrist_cameras(self) -> dict[str, str]:
        return {arm: a.get("wrist_camera", f"cam_wrist_{arm}") for arm, a in self.emb["arms"].items()}

    def base_bodies(self) -> list[str]:
        """Bodies that sit on the mount surface (contact with the table is excluded)."""
        return [f"{a}_link0" for a in self.arms]

    def tcp_site(self, arm: str) -> str:
        return f"{arm}_tcp"

    def joint_names(self, arm: str) -> list[str]:
        return [f"{arm}_{j}" for j in ARM_JOINTS]

    def collision_bodies(self, arm: str, kind: str) -> list[str]:
        names = {"link": LINK_BODIES, "finger": FINGERS, "arm": (*LINK_BODIES, *FINGERS)}[kind]
        return [f"{arm}_{n}" for n in names]

    def action_layout(self) -> list[str]:
        return [n for a in self.arms for n in (*self.joint_names(a), f"{a}_gripper")]

    # -- a compiled model --------------------------------------------------------------------------
    def bind(self, model: mujoco.MjModel) -> Bound:
        m, arms = model, self.arms
        names = [m.body(b).name for b in range(m.nbody)]
        body_arm = np.zeros(m.nbody, int)
        for i, a in enumerate(arms):
            body_arm[m.body_rootid == m.body_rootid[m.body(f"{a}_link0").id]] = i + 1  # the arm's kinematic tree
        tips = ("_hand", *(f"_{f}" for f in FINGERS))
        fingers = [m.joint(f"{a}_{f}") for a in arms for f in FINGER_JOINTS]
        return Bound(
            arms=list(arms), n_joints=self.n_joints,
            qadr=np.array([m.joint(j).qposadr[0] for a in arms for j in self.joint_names(a)]),
            vadr=np.array([m.joint(j).dofadr[0] for a in arms for j in self.joint_names(a)]),
            finger_qadr=np.array([j.qposadr[0] for j in fingers]).reshape(len(arms), 2),
            finger_dof=np.array([int(j.dofadr[0]) for j in fingers]),
            finger_jid=np.array([j.id for j in fingers]),
            arm_act=np.array([m.actuator(f"{a}_actuator{i}").id for a in arms for i in range(1, 8)]),
            grip_act=np.array([m.actuator(f"{a}_actuator8").id for a in arms]),
            tcp_sites=[m.site(self.tcp_site(a)).id for a in arms],
            body_arm=body_arm,
            body_tip=np.array([body_arm[i] > 0 and n.endswith(tips) for i, n in enumerate(names)]),
            hand_bodies=[np.array([m.body(f"{a}_{b}").id for b in ("hand", *FINGERS)]) for a in arms],
            finger_bodies=[[m.body(f"{a}_{f}").id for f in FINGERS] for a in arms],
        )

    def reset(self, data: mujoco.MjData, b: Bound) -> None:
        data.qpos[b.qadr] = self.home_q
        data.qpos[b.finger_qadr] = FINGER_TRAVEL
        data.ctrl[b.arm_act] = self.home_q
        data.ctrl[b.grip_act] = GRIPPER_CTRL_MAX  # fingers start open

    def apply(self, data: mujoco.MjData, b: Bound, joint_targets, grippers, control_dt: float) -> None:
        """joint_targets: (n_arms * 7,) rad; grippers: (n_arms,) in [0 closed, 1 open]. The finger target moves
        towards the command at the real hand's speed."""
        data.ctrl[b.arm_act] = joint_targets
        step = GRIPPER_CTRL_MAX * self.finger_speed * control_dt / FINGER_TRAVEL
        cur = data.ctrl[b.grip_act]
        target = np.clip(grippers, 0.0, 1.0) * GRIPPER_CTRL_MAX
        data.ctrl[b.grip_act] = cur + np.clip(target - cur, -step, step)

    def gripper_width(self, data: mujoco.MjData, b: Bound) -> np.ndarray:
        return data.qpos[b.finger_qadr].sum(axis=1)

    def gripper_command(self, data: mujoco.MjData, b: Bound) -> np.ndarray:
        return data.ctrl[b.grip_act] / GRIPPER_CTRL_MAX
