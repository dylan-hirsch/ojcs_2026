"""
Left- vs. right-hand continuous approximations of the discontinuous target-set
function in Example 2 (Section V.B and Section S.XI of the supplement).

The exact value function at the switching time S = 40 is

    V_true(., 40) = min{ q, max{ r_A, r_B, V_RA[r_A, q; 60](., 40) } }.

We compare it against V_RA[r_eps, q; 60](., 40) for the right-hand
approximation r_eps (ramp from max{r_A, r_B} to r_A on [40, 40 + eps]) and the
left-hand approximation r_check_eps (ramp on [40 - eps, 40]), for several eps.
Only the t = 40 slices are needed, so each approximation is solved backward
from B = 60 to S = 40.

The value functions at t = 40 are saved to `approximation_values.npz`.
"""

import time as timer
from pathlib import Path

import hj_reachability as hj
import jax.numpy as jnp
import numpy as np

from util.robot import RobotDynamics
from util.warehouse_builder import Warehouse, Wall, Shelf

# problem parameters (identical to compute_value.py)
uA_max = 10 / 60
uB_max = 5 / 60
d_max = 2 / 60
B = 60.0
S = 40.0
SPACE_RES = 41
TIME_RES_1 = 21          # snapshots on [S, B], matching compute_value.py
RAMP_RES = 11            # extra snapshots resolving the ramp [S, S + eps]
SAFE_DIST = 0.5
EPSILONS = [1.0, 0.1, 0.01]

OUT_FILE = Path(__file__).resolve().parent / "approximation_values.npz"


def build_problem():
    warehouse = Warehouse()
    warehouse.add_wall(Wall(0., 0., 0.5, 10.))
    warehouse.add_wall(Wall(0., 0., 10., 0.5))
    warehouse.add_wall(Wall(9.5, 0., 10., 10.))
    warehouse.add_wall(Wall(0., 9.5, 10., 10.))
    warehouse.add_wall(Wall(2.0, 2.0, 4.0, 3.0))
    warehouse.add_wall(Wall(6.0, 2.0, 8.0, 3.0))
    warehouse.add_wall(Wall(2.0, 4.0, 4.0, 5.0))
    warehouse.add_wall(Wall(6.0, 4.0, 8.0, 5.0))
    warehouse.add_wall(Wall(2.0, 6.0, 4.0, 7.0))
    warehouse.add_wall(Wall(6.0, 6.0, 8.0, 7.0))
    warehouse.add_shelf(Shelf(3.0, 8.0, 7.0, 9.0))

    model = RobotDynamics(uA_max=uA_max, uB_max=uB_max, d_max=d_max)
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box([0, 0, 0, 0], [10, 10, 10, 10]),
        [SPACE_RES] * 4,
        periodic_dims=None,
    )

    wall_sdf = warehouse.wall_sdf(grid.states[:, :, 0, 0, :])
    shelf_sdf = warehouse.shelf_sdf(grid.states[:, :, 0, 0, :])
    r_A = np.broadcast_to(shelf_sdf[:, :, None, None], [SPACE_RES] * 4)
    r_B = np.broadcast_to(shelf_sdf[None, None, :, :], [SPACE_RES] * 4)
    q_A = np.broadcast_to(wall_sdf[:, :, None, None], [SPACE_RES] * 4)
    q_B = np.broadcast_to(wall_sdf[None, None, :, :], [SPACE_RES] * 4)
    positions = np.asarray(grid.states[:, :, 0, 0, :])
    q_AB = np.linalg.norm(
        positions[:, :, None, None, :] - positions[None, None, :, :, :], axis=-1
    )
    q = np.minimum.reduce([q_A, q_B, q_AB]) - SAFE_DIST
    return model, grid, jnp.asarray(r_A), jnp.asarray(r_B), jnp.asarray(q)


def right_target(r_A, r_B, eps):
    """r_eps: max{r_A, r_B} for t <= S, ramps to r_A on [S, S + eps]."""
    def r(t):
        s = jnp.clip((t - S) / eps, 0.0, 1.0)
        return jnp.maximum(r_A, (1.0 - s) * r_B + s * r_A)
    return r


def left_target(r_A, r_B, eps):
    """r_check_eps: max{r_A, r_B} for t < S - eps, ramps to r_A on [S - eps, S]."""
    def r(t):
        s = jnp.clip((S - t) / eps, 0.0, 1.0)
        return jnp.maximum(r_A, (1.0 - s) * r_A + s * r_B)
    return r


def solve_reach_avoid(model, grid, times, r, q, l):
    """
    V_RA with a (possibly time-varying) target-set function r(t), solved
    backward over `times` from terminal cost min{max{l, r(times[0])}, q}.
    """
    def value_postprocessor(t, V):
        return jnp.minimum(jnp.maximum(V, r(t)), q)

    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", value_postprocessor=value_postprocessor
    )
    V = hj.solve(
        solver_settings, model, grid, times,
        jnp.minimum(jnp.maximum(l, r(times[0])), q),
    )
    return np.asarray(V)


def error_stats(V, V_true, q):
    err = V - V_true
    feasible = np.asarray(q) >= 0.0   # states that satisfy the constraints
    return {
        "max_abs": float(np.max(np.abs(err))),
        "mean_abs": float(np.mean(np.abs(err))),
        "rms": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_feasible": float(np.max(np.abs(err[feasible]))),
        # fraction of grid points where the sign of V (task success) is wrong
        "sign_mismatch": float(np.mean((V >= 0) != (V_true >= 0))),
    }


if __name__ == "__main__":
    model, grid, r_A, r_B, q = build_problem()
    r_AB = jnp.maximum(r_A, r_B)

    # ground truth: V_RA[r_A, q; 60](., 40), then the GRA jump at t = 40
    times_downstream = np.linspace(B, S, TIME_RES_1)
    tic = timer.time()
    V0 = solve_reach_avoid(model, grid, times_downstream, lambda t: r_A, q, r_A)
    print(f"downstream solve: {timer.time() - tic:.1f}s")
    V_RA_40 = V0[-1]
    V_true = np.asarray(jnp.minimum(q, jnp.maximum(r_AB, V_RA_40)))

    results = {"V_true": V_true, "V_RA_A_40": V_RA_40}
    rows = []
    for eps in EPSILONS:
        # snapshots on [S + eps, B] plus a fine sweep through the ramp so the
        # postprocessor sees r(t) at times inside [S, S + eps]
        times = np.unique(np.concatenate([
            np.linspace(B, S + eps, TIME_RES_1),
            np.linspace(S + eps, S, RAMP_RES),
        ]))[::-1]

        for name, make_r in [("right", right_target), ("left", left_target)]:
            r = make_r(r_A, r_B, eps)
            tic = timer.time()
            V = solve_reach_avoid(model, grid, times, r, q, r(B))[-1]
            print(f"{name} eps={eps:g}: {timer.time() - tic:.1f}s")
            results[f"V_{name}_eps{eps:g}"] = V
            rows.append((name, eps, error_stats(V, V_true, q)))

    # report
    keys = ["max_abs", "mean_abs", "rms", "max_abs_feasible", "sign_mismatch"]
    print("\nError against V_true(., 40) = min{q, max{r_A, r_B, V_RA[r_A,q;60](., 40)}}")
    print(f"{'approx':<7}{'eps':>7}" + "".join(f"{k:>18}" for k in keys))
    for name, eps, st in rows:
        print(f"{name:<7}{eps:>7g}" + "".join(f"{st[k]:>18.4e}" for k in keys))

    np.savez_compressed(
        OUT_FILE,
        **results,
        epsilons=np.array(EPSILONS),
        r_A=np.asarray(r_A), r_B=np.asarray(r_B), q=np.asarray(q),
        grid_lo=np.array([0., 0., 0., 0.]), grid_hi=np.array([10., 10., 10., 10.]),
        space_res=SPACE_RES, t=S,
        error_table=np.array([[st[k] for k in keys] for _, _, st in rows]),
        error_rows=np.array([f"{n}_eps{e:g}" for n, e, _ in rows]),
        error_cols=np.array(keys),
    )
    print(f"\nsaved value functions to {OUT_FILE}")
