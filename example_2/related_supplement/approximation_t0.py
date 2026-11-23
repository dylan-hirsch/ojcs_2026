"""
Exact vs. right-hand ramp approximation of the Example 2 value function at
t = 0 (Section V.B).

Exact (Theorem 1, as in compute_value.py):
    V0 = V_RA[r_A, q; 60]           on [40, 60],
    V  = V_GRA[V0(., 40), max{r_A, r_B}, q; 40]   on [0, 40].
Approximation (right-hand ramp r_eps, ramp on [40, 40 + eps]):
    V_eps = V_RA[r_eps, q; 60]      on [0, 60].

Since r_eps >= the discontinuous target and r_eps decreases to it as eps -> 0,
Proposition S5 predicts V_eps(., 0) decreases to V(., 0).

The t = 0 value functions are saved to approximation_t0_values.npz.
Run from this folder (imports helpers from approximation.py).
"""

import time as timer
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from approximation import (build_problem, right_target, solve_reach_avoid,
                           error_stats, B, S, TIME_RES_1, RAMP_RES, SPACE_RES)

TIME_RES_2 = 41          # snapshots on [0, S], matching compute_value.py
EPSILONS = [10.0, 1.0, 0.1]
X0 = (5.0, 1.5, 5.0, 3.5)   # initial state used in compute_value.py

OUT_FILE = Path(__file__).resolve().parent / "approximation_t0_values.npz"


if __name__ == "__main__":
    model, grid, r_A, r_B, q = build_problem()
    r_AB = jnp.maximum(r_A, r_B)
    times_upstream = np.linspace(S, 0.0, TIME_RES_2)

    # exact value via Theorem 1
    tic = timer.time()
    V0 = solve_reach_avoid(model, grid, np.linspace(B, S, TIME_RES_1),
                           lambda t: r_A, q, r_A)
    V_true = solve_reach_avoid(model, grid, times_upstream,
                               lambda t: r_AB, q, V0[-1])[-1]
    print(f"exact solve: {timer.time() - tic:.1f}s")

    axis = np.asarray(grid.coordinate_vectors[0])
    iX0 = tuple(int(np.argmin(np.abs(axis - x))) for x in X0)

    results = {"V_true": V_true}
    rows = []
    for eps in EPSILONS:
        times = np.unique(np.concatenate([
            np.linspace(B, S + eps, TIME_RES_1),
            np.linspace(S + eps, S, RAMP_RES),
            times_upstream,
        ]))[::-1]
        r = right_target(r_A, r_B, eps)
        tic = timer.time()
        V = solve_reach_avoid(model, grid, times, r, q, r(B))[-1]
        print(f"right eps={eps:g}: {timer.time() - tic:.1f}s")
        results[f"V_right_eps{eps:g}"] = V
        st = error_stats(V, V_true, q)
        st["min_signed"] = float(np.min(V - V_true))   # >= 0 if approx from above
        st["V_at_X0"] = float(V[iX0])
        rows.append((eps, st))

    keys = ["max_abs", "mean_abs", "rms", "max_abs_feasible", "sign_mismatch",
            "min_signed", "V_at_X0"]
    print(f"\nError of V_RA[r_eps, q; 60](., 0) against the exact V(., 0)"
          f"  (exact V at X0 = {V_true[iX0]:.4f})")
    print(f"{'eps':>6}" + "".join(f"{k:>18}" for k in keys))
    for eps, st in rows:
        print(f"{eps:>6g}" + "".join(f"{st[k]:>18.4e}" for k in keys))

    np.savez_compressed(
        OUT_FILE,
        **results,
        epsilons=np.array(EPSILONS),
        r_A=np.asarray(r_A), r_B=np.asarray(r_B), q=np.asarray(q),
        grid_lo=np.array([0., 0., 0., 0.]), grid_hi=np.array([10., 10., 10., 10.]),
        space_res=SPACE_RES, t=0.0, X0=np.array(X0),
        error_table=np.array([[st[k] for k in keys] for _, st in rows]),
        error_rows=np.array([f"right_eps{e:g}" for e, _ in rows]),
        error_cols=np.array(keys),
    )
    print(f"\nsaved value functions to {OUT_FILE}")
