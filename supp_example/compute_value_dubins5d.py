# fundamental numerical library
import random
import time
from pathlib import Path

import hj_reachability as hj
import numpy as np
import jax
import jax.numpy as jnp

# my dynamics
from util.dubins5d import Dubins5DDynamics

SEED = 0


def reset_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def std_err(samples):
    return samples.std(ddof=1) / np.sqrt(len(samples))


def gra_tvp(r, q, l):
    """
    Solver settings and terminal values of (GRA-TVP) for V_GRA[r, q, l; T].
    """

    def value_postprocessor(t, V):
        return jnp.minimum(jnp.maximum(V, r), q)

    # specify the accuracy with which to solve the HJ Partial Differential Equation
    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", value_postprocessor=value_postprocessor
    )

    return solver_settings, jnp.minimum(jnp.maximum(l, r), q)


def simplified_tvp(terminal, postprocessor=None):
    """
    Solver settings and terminal values of the simplified HJ-TVP for a single
    canonical task: the terminal condition and (optional) value postprocessor
    written directly, without reference to r, q, l.
    """

    if postprocessor is None:
        solver_settings = hj.SolverSettings.with_accuracy("very_high")
    else:
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high", value_postprocessor=lambda t, V: postprocessor(V)
        )

    return solver_settings, terminal


def compute_value(model, grid, times, solver_settings, terminal_values):
    """
    Solve an HJ-TVP backwards from the terminal time, resetting the seed first.
    """
    reset_seed()

    # solve for the value function
    V = hj.solve(
        solver_settings,
        model,
        grid,
        times,
        terminal_values,
    )

    return np.asarray(V)


def timed_solve(model, grid, times, solver_settings, terminal_values, n_runs):
    """
    Compute the value function n_runs times and time each run. JAX's
    compilation caches are cleared before every run, so each run is a first
    pass that includes JIT compilation. Returns the value function from the
    last run, the array of run times (seconds), and whether every run
    returned bitwise identical values.
    """

    V, run_times, repeatable = None, [], True
    for _ in range(n_runs):
        jax.clear_caches()
        start = time.perf_counter()
        V_k = compute_value(model, grid, times, solver_settings, terminal_values)
        run_times.append(time.perf_counter() - start)
        repeatable &= V is None or np.array_equal(V, V_k)
        V = V_k

    return V, np.array(run_times), repeatable


def write_table(table_rows, n_runs):
    """
    LaTeX table of solve times (mean +/- standard error over n_runs solves,
    each a first pass including JIT compilation) and the max error between
    the two solutions.
    """

    def fmt_time(t):
        return rf"${t.mean():.2f} \pm {std_err(t):.2f}$"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Wall-clock time (s) to compute the value function of each"
        r" canonical task for the 5D Dubins car via (GRA-TVP) using"
        r" Observation~2 and via the simplified HJ-TVP in Table~2 (mean $\pm$"
        rf" standard error over {n_runs} solves, each a first pass including JIT"
        r" compilation, on"
        r" the CPU of an Apple M3 Pro), together with the maximum absolute"
        r" difference between the two solutions over the full space-time grid.}",
        r"\label{tab:supp_timing_dubins5d}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Task & Value function & (GRA-TVP) & Simplified TVP & Max error \\",
        r"\midrule",
    ]
    for name, label, times_GRA, times_TVP, err in table_rows:
        lines.append(rf"{name} & {label} & {fmt_time(times_GRA)} & "
                     rf"{fmt_time(times_TVP)} & ${err:.1f}$ \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    table_path = Path(__file__).with_name("timing_table_dubins5d.tex")
    table_path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {table_path}")


if __name__ == "__main__":

    # specify the time horizon of the problem
    T = 1.5

    # specify the number of voxels to divide the spatial and temporal axes
    XY_RES = 51
    V_RES = 11
    THETA_RES = 24
    OMEGA_RES = 11
    TIME_RES = 31

    # number of timed solves per method
    N_RUNS = 3

    # Specify PDE grid corners (theta is periodic on [-pi, pi))
    XY_MAX = 4
    V_MIN, V_MAX = 0, 2
    OMEGA_MAX = 2

    # (v, theta, omega) at which the value function is sliced for plotting
    V_SLICE = 1.0
    THETA_SLICE = 0.0
    OMEGA_SLICE = 0.0

    # build the dynamics model
    model = Dubins5DDynamics()

    # build the grid
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([-XY_MAX, -XY_MAX, V_MIN, -np.pi, -OMEGA_MAX]),
                    jnp.array([XY_MAX, XY_MAX, V_MAX, np.pi, OMEGA_MAX])),
        [XY_RES, XY_RES, V_RES, THETA_RES, OMEGA_RES],
        periodic_dims=3,
    )
    times = np.linspace(T, 0, TIME_RES)

    # target and terminal set functions are the same signed distance function
    # to an object (positive inside a disk); the constraint set function is
    # the signed distance function to an obstacle (positive outside a disk)
    xy = grid.states[..., :2]
    r = 0.75 - jnp.linalg.norm(xy - jnp.array([1.0, 0.0]), axis=-1)
    l = r
    q = jnp.linalg.norm(xy - jnp.array([-1.0, 0.0]), axis=-1) - 0.5

    pos_inf = jnp.full_like(r, jnp.inf)
    neg_inf = jnp.full_like(r, -jnp.inf)

    # Observation 2 asks for l <= r(., T) in (ii), (v), which l = r already
    # satisfies, and l >= q(., T) in (iii), which it does not, so for the
    # avoid task we clip the terminal set function to satisfy the inequality
    l_below_r = jnp.minimum(l, r)
    l_above_q = jnp.maximum(l, q)

    # (name, value function label, which of (r, q, l) are active, i.e. not
    #  simplified out by Observation 2, GRA arguments (r, q, l) per
    #  Observation 2, simplified HJ-TVP (terminal condition, postprocessor)
    #  per Table 2)
    tasks = [
        ("(I) Terminal-set",
         r"$V_{\mathrm{T}}[\ell;T]$",
         (False, False, True),
         (neg_inf, pos_inf, l),
         (l, None)),
        ("(II) Reach",
         r"$V_{\mathrm{R}}[r;T]$",
         (True, False, False),
         (r, pos_inf, l_below_r),
         (r, lambda V: jnp.maximum(V, r))),
        ("(III) Avoid",
         r"$V_{\mathrm{A}}[q;T]$",
         (False, True, False),
         (neg_inf, q, l_above_q),
         (q, lambda V: jnp.minimum(V, q))),
        ("(IV) Constrained terminal-set",
         r"$V_{\mathrm{CT}}[\ell,q;T]$",
         (False, True, True),
         (neg_inf, q, l),
         (jnp.minimum(q, l), lambda V: jnp.minimum(V, q))),
        ("(V) Reach-avoid",
         r"$V_{\mathrm{RA}}[r,q;T]$",
         (True, True, False),
         (r, q, l_below_r),
         (jnp.minimum(q, r), lambda V: jnp.minimum(jnp.maximum(V, r), q))),
    ]

    # grid indices of the plotted (v, theta, omega) slice
    i_v = int(np.argmin(np.abs(grid.coordinate_vectors[2] - V_SLICE)))
    i_theta = int(np.argmin(np.abs(grid.coordinate_vectors[3] - THETA_SLICE)))
    i_omega = int(np.argmin(np.abs(grid.coordinate_vectors[4] - OMEGA_SLICE)))
    xy_slice = np.s_[:, :, i_v, i_theta, i_omega]

    results = []
    table_rows = []
    for name, label, active, (r_i, q_i, l_i), (terminal, postprocessor) in tasks:
        V_GRA, times_GRA, repeat_GRA = timed_solve(
            model, grid, times, *gra_tvp(r_i, q_i, l_i), N_RUNS)
        V_TVP, times_TVP, repeat_TVP = timed_solve(
            model, grid, times, *simplified_tvp(terminal, postprocessor), N_RUNS)

        # max error over the full space-time grid, and on the plotted slice
        err = np.max(np.abs(V_GRA - V_TVP))
        slice_err = np.max(np.abs(V_GRA[-1][xy_slice] - V_TVP[-1][xy_slice]))
        bitwise = np.array_equal(V_GRA, V_TVP)
        print(f"{name:31s} max error over all (t, x) = {err:.3e}, "
              f"on plotted slice = {slice_err:.3e}   bitwise identical: {bitwise}")
        print(f"{'':31s} GRA-TVP:        {times_GRA.mean():.2f} +/- "
              f"{std_err(times_GRA):.2f} s   runs repeatable: {repeat_GRA}")
        print(f"{'':31s} simplified TVP: {times_TVP.mean():.2f} +/- "
              f"{std_err(times_TVP):.2f} s   runs repeatable: {repeat_TVP}")
        results.append((label, active, V_GRA[-1][xy_slice],
                        V_TVP[-1][xy_slice], slice_err))
        table_rows.append((name, label, times_GRA, times_TVP, err))
        del V_GRA, V_TVP

    write_table(table_rows, N_RUNS)

    # save the (y_1, y_2) slice of each value function at the initial time and
    # fixed (v, theta, omega), along with the set functions on that slice, for
    # plot_dubins5d.py
    slices_path = Path(__file__).with_name("dubins5d_slices.npz")
    np.savez(
        slices_path,
        Y1=np.asarray(grid.states[xy_slice][..., 0]),
        Y2=np.asarray(grid.states[xy_slice][..., 1]),
        r=np.asarray(r[xy_slice]),
        q=np.asarray(q[xy_slice]),
        l=np.asarray(l[xy_slice]),
        labels=np.array([res[0] for res in results]),
        active=np.array([res[1] for res in results]),
        V0_GRA=np.stack([res[2] for res in results]),
        V0_TVP=np.stack([res[3] for res in results]),
        slice_err=np.array([res[4] for res in results]),
        v=float(grid.coordinate_vectors[2][i_v]),
        theta=float(grid.coordinate_vectors[3][i_theta]),
        omega=float(grid.coordinate_vectors[4][i_omega]),
    )
    print(f"wrote {slices_path}")
