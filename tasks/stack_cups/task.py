"""mujoco_toy-project's cup stacking as a plugin task: what the predicate library doesn't express.

- Start state: the toy's `reset_arm` (home pose plus uniform joint noise) and `reset_object_positions` (cups at
  random spots in a rectangle in front of the arm base, each at least min_dist from the one placed before it).
- Stages: the toy's `check_cups_stacked`, for every cup on the one below it (cups are listed bottom to top), with
  upright measured as tilt from vertical: the toy's quaternion test failed cups turned clockwise about their axis.
"""

import numpy as np

from xrcollect.task.declarative import DeclarativeTask


class StackCupsTask(DeclarativeTask):
    def __init__(self, bundle):
        self.params = bundle.task["params"]
        self.cups = list(bundle.objects)                       # bottom to top
        arms = bundle.embodiment["arms"]
        self.base = np.asarray(arms[next(iter(arms))]["pos"], float)  # the first arm's mount: the toy's origin
        super().__init__(bundle)

    # -- start state ---------------------------------------------------------------------------------
    def reset(self, model, data, rng: np.random.Generator) -> None:
        super().reset(model, data, rng)
        P, w = self.params, self.world
        b = w.bound
        home = np.tile(np.asarray(P["home_q"], float), b.n_arms)
        q = home + rng.uniform(-P["arm_noise"] / 2, P["arm_noise"] / 2, home.shape)
        data.qpos[b.qadr] = q
        data.qvel[b.vadr] = 0.0
        data.ctrl[b.arm_act] = q   # hold the start pose

        s = P["spawn"]
        lo = np.array([s["x"][0], s["y"][0]])
        span = np.array([s["x"][1] - s["x"][0], s["y"][1] - s["y"][0]])
        xy = [lo + rng.random(2) * span]
        for _ in self.cups[1:]:  # as in the toy: only the previous cup is kept at a distance
            while True:
                p = lo + rng.random(2) * span
                if np.linalg.norm(p - xy[-1]) >= s["min_dist"]:
                    xy.append(p)
                    break
        for name, p in zip(self.cups, xy):
            a, v = w.qadr[name], w.vadr[name]
            data.qpos[a:a + 3] = [*(self.base[:2] + p), w.table_height]
            data.qpos[a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[v:v + 6] = 0.0

    # -- stages --------------------------------------------------------------------------------------
    def cups_stacked(self, data, lower: str, upper: str) -> bool:
        """Both cups upright, the upper one centred on the lower one and resting inside it. Upright is the tilt of
        the cup's axis from vertical, so a cup turned about its own axis (either way) still counts."""
        S, w = self.params["stack"], self.world
        lo, up = w.body[lower], w.body[upper]
        cos_max = np.cos(np.radians(S["max_tilt_deg"]))
        if any(data.xmat[b][8] < cos_max for b in (lo, up)):   # xmat[8]: z of the body's z axis
            return False
        d = data.xpos[up] - data.xpos[lo]
        return bool(np.all(np.abs(d[:2]) <= S["tol_xy"]) and S["dz"][0] <= d[2] <= S["dz"][1])

    def stage_flags(self, model, data) -> np.ndarray:
        """Stage k: cup k+1 sits in cup k, and every cup below is stacked too."""
        flags = np.zeros(len(self.stage_names), bool)
        for k in range(len(flags)):
            if not self.cups_stacked(data, self.cups[k], self.cups[k + 1]):
                break
            flags[k] = True
        return flags
