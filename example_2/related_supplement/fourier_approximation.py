"""
Failure of a non-monotonic approximation scheme (Section S.IX.B).

Scalar system  x' = 1  with specification
    psi := (q U_[0,0.5] r_A) or (q U_[0,1] r_B),   q = +inf, r_A = 1, r_B = 0,
so the target-set function is the step  r(t) = 1 for t <= 0.5, 0 for t > 0.5,
and the true value at x = 0 is  V(0, t) = 1 for t <= 0.5 and 0 for t > 0.5.

We approximate the step by
  * a ramp:     r_ramp_i(t) = 1 on t <= 0.5, 1 - (t - 0.5) i on [0.5, 0.5 + 1/i),
                              0 on t >= 0.5 + 1/i
  * a Fourier partial sum (of the 2*pi-periodic function equal to 1 on
    [0, 0.5], 0 on (0.5, pi), 1 on [pi, 2*pi), which matches the step on [0, 1]):
        r_fourier_i(t) = 1/2 + 1/(4 pi) + sum_{j=1}^{i} (a_j cos(j t) + b_j sin(j t)),
        a_j = sin(j/2) / (pi j),   b_j = ((-1)^j - cos(j/2)) / (pi j),
and solve the reach-avoid HJ-PDE for V_RA[r_i, q; 1] for i = 10, 50, 250.
The ramp approaches the step monotonically from above and converges; the
Fourier sum does not (Gibbs overshoot), so its value at t = 0 stays near 1.09.

Results are saved to fourier_approximation_values.npz and plotted in
fourier_approximation.pdf.
"""

from pathlib import Path

import hj_reachability as hj
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from hj_reachability import dynamics, sets

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "fourier_approximation_values.npz"
FIG_FILE = HERE / "fourier_approximation.pdf"

T = 1.0
SWITCH = 0.5
ORDERS = [10, 50, 250]
SPACE_RES = 10           # spatial grid points (value is constant in x anyway)
TIME_RES = 20001         # dt = 5e-5: resolves the i = 250 ramp (width 4e-3)
                         # and Fourier oscillations (period 2 pi / 250 ~ 0.025)


class UnitDrift(dynamics.ControlAndDisturbanceAffineDynamics):
    """x' = 1, with trivial (zero) control and disturbance."""

    def __init__(self):
        zero = sets.Box(jnp.zeros(1), jnp.zeros(1))
        super().__init__("max", "min", zero, zero)

    def open_loop_dynamics(self, state, time):
        return jnp.ones(1)

    def control_jacobian(self, state, time):
        return jnp.zeros((1, 1))

    def disturbance_jacobian(self, state, time):
        return jnp.zeros((1, 1))


def r_step(t):
    return np.where(t <= SWITCH, 1.0, 0.0)


def r_ramp(i):
    return lambda t: jnp.clip(1.0 - (t - SWITCH) * i, 0.0, 1.0)


def r_fourier(i):
    j = jnp.arange(1, i + 1) * 1.0
    a = jnp.sin(j / 2) / (jnp.pi * j)
    b = ((-1.0) ** j - jnp.cos(j / 2)) / (jnp.pi * j)

    def r(t):
        t = jnp.asarray(t)[..., None]
        return 0.5 + 1 / (4 * jnp.pi) + jnp.sum(a * jnp.cos(j * t) + b * jnp.sin(j * t), axis=-1)
    return r


def solve_reach(model, grid, times, r):
    """V_RA[r, +inf; T]: terminal value r(T), postprocessor max{V, r(t)}."""
    settings = hj.SolverSettings.with_accuracy(
        "very_high", value_postprocessor=lambda t, V: jnp.maximum(V, r(t))
    )
    V0 = jnp.full(grid.shape, r(times[0]))
    return np.asarray(hj.solve(settings, model, grid, times, V0, progress_bar=False))


if __name__ == "__main__":
    model = UnitDrift()
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(np.array([0.0]), np.array([1.0])), (SPACE_RES,), periodic_dims=0
    )
    i0 = int(np.argmin(np.abs(np.asarray(grid.coordinate_vectors[0]))))  # x = 0
    times = np.linspace(T, 0.0, TIME_RES)        # backward in time
    t_plot = times[::-1]
    V_true = r_step(t_plot)                      # V(0, t) = sup_{s in [t, 1]} r(s)

    results = {"t": t_plot, "V_true": V_true, "r_true": r_step(t_plot),
               "orders": np.array(ORDERS), "x": np.asarray(grid.coordinate_vectors[0])}
    away = np.abs(t_plot - SWITCH) > 0.05        # away from the discontinuity

    print(f"{'scheme':<9}{'i':>5}{'V(0,0)':>12}{'|V(0,0)-1|':>14}"
          f"{'sup err (|t-0.5|>0.05)':>26}{'spread in x':>14}")
    for name, make_r in [("ramp", r_ramp), ("fourier", r_fourier)]:
        for i in ORDERS:
            r = make_r(i)
            V = solve_reach(model, grid, times, r)[::-1]   # shape (time, x), t ascending
            V0 = V[:, i0]
            results[f"V_{name}_{i}"] = V0
            results[f"r_{name}_{i}"] = np.asarray(r(jnp.asarray(t_plot)))
            sup_err = np.max(np.abs(V0 - V_true)[away])
            spread = np.max(V.max(axis=1) - V.min(axis=1))
            print(f"{name:<9}{i:>5}{V0[0]:>12.5f}{abs(V0[0] - 1):>14.2e}"
                  f"{sup_err:>26.2e}{spread:>14.1e}")

    np.savez_compressed(DATA_FILE, **results)
    print(f"\nsaved to {DATA_FILE}")

    # plot: target functions (top) and value at x = 0 (bottom)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(ORDERS)))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey="row",
                             layout="constrained")
    for col, (name, title) in enumerate([("ramp", "Ramp"), ("fourier", "Fourier")]):
        ax_r, ax_v = axes[0, col], axes[1, col]
        for c, i in zip(colors, ORDERS):
            ax_r.plot(t_plot, results[f"r_{name}_{i}"], color=c, lw=1.5, label=f"$i = {i}$")
            ax_v.plot(t_plot, results[f"V_{name}_{i}"], color=c, lw=2, label=f"$i = {i}$")
        for ax in (ax_r, ax_v):
            ax.plot(t_plot, V_true, "k--", lw=1.5, label="exact")
            ax.axhline(1.0, color="gray", lw=0.5)
        ax_r.set_title(f"{title} approximation")
        ax_r.set_ylabel(f"$r^{{\\mathrm{{{name}}}}}_i(t)$")
        ax_v.set_ylabel(f"$V_{{RA}}[r^{{\\mathrm{{{name}}}}}_i, q; 1](0, t)$")
        ax_v.set_xlabel("$t$")
    axes[1, 0].legend(loc="lower left")
    fig.savefig(FIG_FILE)
    print(f"saved figure to {FIG_FILE}")
    plt.show()
