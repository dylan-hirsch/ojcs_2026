"""
Maximal error over time t in [0, 60] of the left- and right-hand ramp
approximations of the Example 2 value function, for eps = 10, 1, 0.1.

Exact (Theorem 1, as in compute_value.py):
    V(., t) = V_RA[r_A, q; 60](., t)                               for t in (40, 60],
    V(., t) = V_GRA[V_RA[r_A, q; 60](., 40), max{r_A, r_B}, q; 40](., t)  for t in [0, 40].
Approximations: V_RA[r_eps, q; 60] (right-hand ramp on [40, 40 + eps]) and
V_RA[r_check_eps, q; 60] (left-hand ramp on [40 - eps, 40]), solved on [0, 60].

At every snapshot time we record max_x |V_approx(x, t) - V(x, t)| over the
grid. Snapshots: every second (as in Table S1) plus 11 points inside each ramp.
Error curves are saved to approximation_over_time.npz and plotted in
approximation_over_time.pdf. Run from this folder.
"""

import time as timer
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from approximation import (build_problem, right_target, left_target,
                           solve_reach_avoid, B, S, RAMP_RES)

HERE = Path(__file__).resolve().parent
OUT_FILE = HERE / "approximation_over_time.npz"
FIG_FILE = HERE / "approximation_over_time.pdf"

EPSILONS = [10.0, 1.0, 0.1]
BASE_TIMES = np.arange(0.0, B + 1.0, 1.0)     # 1 s snapshots on [0, 60]
FLOOR = 1e-7                                    # for plotting exact zeros on a log axis


def ramp_times(eps, side):
    lo, hi = (S, S + eps) if side == "right" else (S - eps, S)
    return np.linspace(lo, hi, RAMP_RES)


def descending(*arrays):
    # round so the same time from different arrays is recognised as equal
    return np.unique(np.round(np.concatenate(arrays), 10))[::-1]


if __name__ == "__main__":
    model, grid, r_A, r_B, q = build_problem()
    r_AB = jnp.maximum(r_A, r_B)

    # exact value on a snapshot set containing every approximation's snapshots
    t_down = descending(BASE_TIMES[BASE_TIMES >= S],
                        *[ramp_times(e, "right") for e in EPSILONS])
    t_up = descending(BASE_TIMES[BASE_TIMES <= S],
                      *[ramp_times(e, "left") for e in EPSILONS])
    tic = timer.time()
    V_down = solve_reach_avoid(model, grid, t_down, lambda t: r_A, q, r_A)
    V_up = solve_reach_avoid(model, grid, t_up, lambda t: r_AB, q, V_down[-1])
    print(f"exact solve: {timer.time() - tic:.1f}s")

    exact = {float(t): V_down[k] for k, t in enumerate(t_down) if t > S}
    exact.update({float(t): V_up[k] for k, t in enumerate(t_up)})   # t = 40: post-jump value

    results = {}
    for eps in EPSILONS:
        for side, make_r in [("right", right_target), ("left", left_target)]:
            times = descending(BASE_TIMES, ramp_times(eps, side))
            r = make_r(r_A, r_B, eps)
            tic = timer.time()
            V = solve_reach_avoid(model, grid, times, r, q, r(B))
            err = np.array([np.max(np.abs(V[k] - exact[float(t)]))
                            for k, t in enumerate(times)])
            del V
            print(f"{side} eps={eps:g}: {timer.time() - tic:.1f}s, "
                  f"max err {err.max():.3e} at t = {times[np.argmax(err)]:g}, "
                  f"err(0) = {err[-1]:.3e}")
            results[f"t_{side}_eps{eps:g}"] = times[::-1]
            results[f"err_{side}_eps{eps:g}"] = err[::-1]

    np.savez_compressed(OUT_FILE, epsilons=np.array(EPSILONS), **results)
    print(f"saved error curves to {OUT_FILE}")

    # plot: one panel per eps, both approximations, ramp intervals shaded
    colors = {"right": "tab:blue", "left": "tab:red"}
    fig, axes = plt.subplots(1, len(EPSILONS), figsize=(15, 4.5), sharey=True,
                             layout="constrained")
    for ax, eps in zip(axes, EPSILONS):
        for side in ["left", "right"]:
            t = results[f"t_{side}_eps{eps:g}"]
            e = np.maximum(results[f"err_{side}_eps{eps:g}"], FLOOR)
            ax.semilogy(t, e, ".-", color=colors[side], lw=1.5, ms=3,
                        label=f"{side}-hand ramp")
            lo, hi = ramp_times(eps, side)[[0, -1]]
            ax.axvspan(lo, hi, color=colors[side], alpha=0.12)
        ax.axvline(S, color="gray", lw=0.8, ls="--")
        ax.set_title(f"$\\varepsilon = {eps:g}$")
        ax.set_xlabel("$t$")
        ax.set_xlim(0, B)
    axes[0].set_ylabel(r"$\max_x |V_\varepsilon(x, t) - V(x, t)|$")
    axes[0].legend(loc="upper left")
    fig.savefig(FIG_FILE)
    print(f"saved figure to {FIG_FILE}")
    plt.show()
