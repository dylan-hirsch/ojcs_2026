"""
Figure for Example 4 using only the efficient approach: the downstream task
(tour of the fruit stands) is solved on [T/2, T] with Algorithm 2, and a single
upstream GRA solve on [0, T/2] attaches it via Theorem 1. Built from the saved
outputs of compute_value.py:

  * data/values.npz       (value functions)
  * data/closed_loop.npz  (closed-loop trajectories)

Panels (2 x 2):
  (top left)     V(x, y, theta, omega; t = 0), with the number of GRA solves
  (top right)    closed loop that reaches the grocery store (target 0) by t = 5
  (bottom left)  closed loop that is at the truck site at exactly t = 5
  (bottom right) closed loop that tours fruit stands 1, 2, 3 during [5, 10]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import compute_value as cv
from plot_results import plot_closed_loops, plot_value


def kind(events):
    """Which way the closed loop completed the task."""
    names = [n for _, n in events]
    if "target 0" in names:
        return "grocery"
    if "truck" in names:
        return "truck"
    return "tour"


if __name__ == "__main__":

    model, grid, q, rs = cv.build_problem()
    V_up = np.load(cv.HERE / "data" / "values.npz")["V_up"]
    closed_loop = np.load(cv.HERE / "data" / "closed_loop.npz")

    sims = {}
    for i in range(len(cv.INITIAL_STATES)):
        events = list(zip(closed_loop[f"event_times_{i}"],
                          closed_loop[f"event_names_{i}"]))
        sims[kind(events)] = (closed_loop[f"x_{i}"], closed_loop[f"t_{i}"], events)
    missing = {"grocery", "truck", "tour"} - set(sims)
    if missing:
        raise RuntimeError(f"no closed loop completes the task via {missing}; "
                           "rerun compute_value.py or change INITIAL_STATES")

    # Algorithm 2 needs one solve per nonempty subset of the fruit stands,
    # then Theorem 1 needs one upstream GRA solve
    n_stands = len(cv.TARGETS) - 1
    n_solves = (2 ** n_stands - 1) + 1

    # manual layout so all four panels have the same size: the value colorbar
    # sits in the gap right of the value panel, the time colorbar (shared by
    # the three closed loops) at the right edge
    fig, ((ax_v, ax_g), (ax_t, ax_s)) = plt.subplots(2, 2, figsize=(10.5, 9.6))
    fig.subplots_adjust(left=0.07, right=0.87, bottom=0.06, top=0.95,
                        wspace=0.55, hspace=0.25)

    mesh = plot_value(ax_v, grid, V_up[-1], f"Value function ({n_solves} GRA solves)")
    cax_v = ax_v.inset_axes([1.04, 0.0, 0.045, 1.0])
    fig.colorbar(mesh, cax=cax_v, extend="both",
                 label=r"$V(x, y, \theta, \omega; 0)$")

    norm = Normalize(0.0, cv.T_FINAL)
    plot_closed_loops(ax_g, [sims["grocery"]], "Closed loop: grocery store", norm)
    plot_closed_loops(ax_t, [sims["truck"]], "Closed loop: truck", norm)
    lc = plot_closed_loops(ax_s, [sims["tour"]], "Closed loop: fruit stands", norm)
    top, bottom = ax_g.get_position(), ax_s.get_position()
    cax_t = fig.add_axes([top.x1 + 0.025, bottom.y0, 0.018, top.y1 - bottom.y0])
    fig.colorbar(lc, cax=cax_t, label="time")

    fig.savefig(cv.HERE / "results_efficient.png", dpi=200)
    fig.savefig(cv.HERE / "results_efficient.pdf")
    plt.show()
