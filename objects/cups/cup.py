"""Builder for a tapered, nestable cup as plain object data (objects/cups/cup.yaml calls it).

Migrated from mujoco_toy-project (models/generate_cup.py), same geometry: MuJoCo only collides convex shapes, so
a cup mesh would act like a solid cylinder. Instead the cup is a thin cylinder base plus a ring of thin boxes,
tilted so the top leans outwards. Body frame = centre of the cup bottom.
"""

import numpy as np


def _quat_mul(a, b) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def cup(bottom_radius: float = 0.028, top_radius: float = 0.036, height: float = 0.07, thickness: float = 0.003,
        n_wall: int = 16) -> dict:
    """bottom_radius / top_radius: outer radius at the bottom / rim (the Panda opens to 8 cm, so the cup has to be
    thinner than that where it is grasped); n_wall: boxes forming the wall."""
    tilt = np.arctan2(top_radius - bottom_radius, height)
    slant = float(np.hypot(top_radius - bottom_radius, height))
    mid_radius = (bottom_radius + top_radius) / 2 - thickness / 2
    half_width = float(top_radius * np.tan(np.pi / n_wall) * 1.05)  # a bit wider than the chord so neighbours overlap
    geoms = [{"type": "cylinder", "size": [bottom_radius, thickness / 2], "pos": [0, 0, thickness / 2]}]
    for i in range(n_wall):
        a = 2 * np.pi * i / n_wall
        # rotate around z to the wall position, then tilt around local y
        q = _quat_mul([np.cos(a / 2), 0, 0, np.sin(a / 2)], [np.cos(tilt / 2), 0, np.sin(tilt / 2), 0])
        geoms.append({"type": "box", "size": [thickness / 2, half_width, slant / 2],
                      "pos": [float(mid_radius * np.cos(a)), float(mid_radius * np.sin(a)), height / 2],
                      "quat": [float(v) for v in q]})
    return {"shape": "cup", "size": [2 * top_radius, 2 * top_radius, height], "geoms": geoms}
