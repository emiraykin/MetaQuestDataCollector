"""Scripted run for `xrc validate-task tasks/stack_cups`: put each cup into the one below it.

A cup is grasped from above just below its rim: lower down, the fingers would hit the rim of the cup below when
nesting. It is let go about a centimetre above its resting place and drops in.
"""

import numpy as np

from xrcollect.expert import grasp_quat

LIFT = 0.12


def script(ex) -> None:
    env, task = ex.env, ex.env.task
    arm = env.arms[0]
    q = grasp_quat(0.0)
    for lower, upper in zip(task.cups, task.cups[1:]):
        grasp_z = float(env.world.size[upper][2]) - 0.012
        p, _ = ex.object_pose(upper)
        ex.move(arm, p + [0, 0, grasp_z + LIFT], q, seconds=1.2)
        ex.move(arm, p + [0, 0, grasp_z], seconds=0.8)
        ex.move(arm, grip=0.0, seconds=0.6)
        ex.move(arm, p + [0, 0, grasp_z + LIFT], seconds=0.8)
        goal = ex.object_pose(lower)[0] + np.array([0, 0, task.params["nest_dz"]])
        ex.move(arm, goal + [0, 0, grasp_z + LIFT], seconds=1.2)
        ex.move(arm, goal + [0, 0, grasp_z + 0.01], seconds=0.9)
        ex.move(arm, grip=1.0, seconds=0.6)
        ex.move(arm, goal + [0, 0, grasp_z + LIFT], seconds=0.8)
    ex.go_home(arm)
