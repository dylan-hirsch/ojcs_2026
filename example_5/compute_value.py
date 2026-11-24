"""
Example 5: a planar drone in a state-dependent wind, comparing the
generalized reach-avoid (GRA) value function with the naive combination of
its reach-avoid (RA) and constrained-terminal (CT) parts.

The drone x = (x, y) obeys x' = u + G_d(x) d with ||u|| <= 1 and
d in [-1, 1]^2 (see util.drone_drift). Over the horizon [-T, 0] it must stay
in the safe corridor q >= 0 and EITHER reach the left region r >= 0
(x <= -3) at some time OR be in the right region l >= 0 (x >= 3) at the
final time. Three value functions are computed:

    V_RA = V_GRA[r, q, r; T]      (reach the left region)
    V_CT = V_GRA[-inf, q, l; T]   (be in the right region at time 0)
    V_GRA = V_GRA[r, q, l; T]      (either of the two)

The example is built so that, from x0 = (0, 0), both RA and CT fail
(V < 0), hence so does the naive max(V_RA, V_CT), while GRA succeeds
(V > 0): the drone can hedge between the two goals.

Outputs data/values.npz, read by plot_results.py: the value functions at
t = -T and the sampled closed-loop trajectories under each controller.
"""

from pathlib import Path

import hj_reachability as hj
import jax.numpy as jnp
import numpy as np

from util.closed_loop import ClosedLoopTrajectoryAdaptive
from util.drone_drift import DroneWithDriftDynamics, safety_margin


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "values.npz"

T = 7.0  # time horizon
X_MIN, X_MAX = -15.0, 15.0
Y_MIN, Y_MAX = -15.0, 15.0
X_VOXELS = 1000
Y_VOXELS = 1000
T_VOXELS = 2000

X0 = np.array([0.0, 0.0])  # initial state of the closed loops
STEPS = 100  # sample-and-hold steps of the closed loops
N_SAMPLES = 1000  # time samples of each closed loop saved for plotting

METHODS = ("RA", "CT", "GRA")


def make_model():
    return DroneWithDriftDynamics()


def make_grid():
    return hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([X_MIN, Y_MIN]), jnp.array([X_MAX, Y_MAX])),
        jnp.array([X_VOXELS + 1, Y_VOXELS + 1]),
    )


def make_times():
    return np.linspace(0.0, -T, T_VOXELS + 1)


def set_functions(grid):
    """Target r, terminal-set l, and safety q functions (positive inside)."""
    x = grid.states[..., 0]
    y = grid.states[..., 1]
    q = safety_margin(x, y)
    r = -x - 3
    l = x - 3
    return r, l, q


def solve_gra(grid, model, q, r, l, times):
    """V_GRA[r, q, l; T] on the whole time grid."""

    def value_postprocessor(t, v):
        return jnp.minimum(q, jnp.maximum(r, v))

    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", value_postprocessor=value_postprocessor
    )
    V_terminal = jnp.minimum(q, jnp.maximum(r, l))
    return np.asarray(hj.solve(solver_settings, model, grid, times, V_terminal))


def main():
    model = make_model()
    grid = make_grid()
    times = make_times()
    r, l, q = set_functions(grid)

    problems = {
        "RA": (r, r),
        "CT": (-jnp.inf, l),
        "GRA": (r, l),
    }
    ts = np.linspace(-T, 0.0, N_SAMPLES)
    out = {"ts": ts, "x0": X0}

    for name, (target, terminal) in problems.items():
        print(f"Solving {name} ...")
        V = solve_gra(grid, model, q, target, terminal, times)

        print(f"Simulating {name} closed loop ...")
        cl = ClosedLoopTrajectoryAdaptive(
            model, grid, times, V, initial_state=X0, steps=STEPS
        )
        out[f"{name}_x"] = np.asarray([cl.x(float(t)) for t in ts], dtype=float)
        out[f"{name}_u"] = np.asarray([cl.u(float(t)) for t in ts], dtype=float)
        out[f"{name}_d"] = np.asarray([cl.d(float(t)) for t in ts], dtype=float)
        out[f"{name}_V0"] = V[-1]  # value function at t = -T
        del V, cl  # each full value function is several GB

    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA, **out)
    print(f"Saved {DATA}")

    # value comparison at the same (x0, -T)
    x0 = jnp.asarray(X0)
    v = {name: float(grid.interpolate(jnp.asarray(out[f"{name}_V0"]), x0))
         for name in METHODS}
    v_naive = float(grid.interpolate(
        jnp.maximum(out["RA_V0"], out["CT_V0"]), x0))
    print("\n=== Value comparison at the same (x0, t0) ===")
    print(f"x0    = {X0}")
    print(f"time  = {T:.4f}")
    for name in ("GRA", "RA", "CT"):
        print(f"{name:<5} = {v[name]:.4f}")
    print(f"Naive = {v_naive:.4f}")

    print("\n=== Desired paper-style condition ===")
    print(f"RA < 0:   {v['RA'] < 0}")
    print(f"CT < 0:   {v['CT'] < 0}")
    print(f"Naive < 0: {v_naive < 0}")
    print(f"GRA > 0:   {v['GRA'] > 0}")
    success = v["RA"] < 0 and v["CT"] < 0 and v_naive < 0 and v["GRA"] > 0
    print(f"\nDesired example found: {success}")


if __name__ == "__main__":
    main()
