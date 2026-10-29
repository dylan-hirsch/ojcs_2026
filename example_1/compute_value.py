# fundamental numerical library
import hj_reachability as hj
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrow
from matplotlib.legend_handler import HandlerPatch

# my dynamics
from util.drone import DroneDynamics
from util.warf_builder import Warf, Zone, Pad
from util.closed_loop import ClosedLoopTrajectory

def compute_value(model, grid, times, r, q, l):

    def value_postprocessor(t, V):
        return jnp.minimum(jnp.maximum(V, r), q)

    # specify the accuracy with which to solve the HJ Partial Differential Equation
    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", value_postprocessor=value_postprocessor
    )

    # solve for the value function
    V = hj.solve(
        solver_settings,
        model,
        grid,
        times,
        jnp.minimum(jnp.maximum(l, r), q),
    )

    return V


def optimal_disturbance_field(model, grid, Vf, time=0.0, skip=6):
    """
    Optimal disturbance at each grid point, driven by the spatial gradient of
    the value function at the initial time, subsampled every `skip` voxels so
    the quiver stays readable.
    """
    grad = grid.grad_values(Vf[-1])
    flat_states = grid.states.reshape(-1, grid.states.shape[-1])
    flat_grad = grad.reshape(-1, grad.shape[-1])
    d = jax.vmap(
        lambda x, p: model.optimal_disturbance(x, time, p)
    )(flat_states, flat_grad)
    d = d.reshape(grad.shape)

    sl = np.s_[::skip, ::skip]
    return (np.asarray(grid.states[sl][..., 0]),
            np.asarray(grid.states[sl][..., 1]),
            np.asarray(d[sl][..., 0]),
            np.asarray(d[sl][..., 1]))


if __name__ == "__main__":

    # specify the time horizons of the problem
    T = 10

    # Initial state and number of steps to take in closed loop
    X0 = (1.0, 1.0)
    STEPS = 100

    # specify the number of voxels to divide the spatial and temporal axes
    SPACE_RES = 100
    TIME_RES = 100

    # Specify PDE grid corners
    X_MIN = -12
    Y_MIN = -12

    X_MAX = 12
    Y_MAX = 12

    # Build the warf
    warf = Warf()
    warf.add_pad(Pad(-4.0, 4.0, 6.0, 6.0))
    warf.add_pad(Pad(4.0, -4.0, 6.0, 6.0))
    warf.add_zone(Zone(0.0, 0.0, 9.0))

    # build the dynamics model 
    model = DroneDynamics()

    # build the grid 
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box([X_MIN, Y_MIN],
                    [X_MAX, Y_MAX]),
        [SPACE_RES, SPACE_RES],
        periodic_dims=None
    )
    times = np.linspace(T, 0, TIME_RES)

    r = warf.pad_sdf(grid.states)
    q = warf.zone_sdf(grid.states)
    r_vacuous = -jnp.inf * jnp.ones_like(r)

    # compute the downstream value function
    V = compute_value(model, grid, times, r, q, q)
    cl = ClosedLoopTrajectory(model, grid, times, V, X0, steps = STEPS) 
    V_RA = compute_value(model, grid, times, r, q, r)
    V_A = compute_value(model, grid, times, r_vacuous, q, q)
    cl_RA = ClosedLoopTrajectory(model, grid, times, V_RA, X0, steps = STEPS)
    cl_A = ClosedLoopTrajectory(model, grid, times, V_A, X0, steps = STEPS)

    # 2x2 comparison: correct value function (left) vs the wrong one (right).
    # Row 1: value function at the initial time. Row 2: environment with the
    # induced closed-loop trajectory.
    columns = [
        ("correct", V),
        ("wrong", np.maximum(V_A, V_RA)),
    ]

    X = np.asarray(grid.states[..., 0])
    Y = np.asarray(grid.states[..., 1])

    # one color scale across both columns, spanning the full data range with a
    # piecewise-linear norm: [vmin, 0] fills the lower half of the colorbar and
    # [0, vmax] the upper half, so white sits at V = 0 and the (small) positive
    # range stays readable
    V0s = [np.asarray(Vf[-1]) for _, Vf in columns]
    vmin = min(V0.min() for V0 in V0s)
    vmax = max(max(V0.max() for V0 in V0s), 1e-3)

    # pick round tick locations separately on each side of zero, and stretch
    # the color range so the bar starts and ends exactly on round ticks
    neg_ticks = MaxNLocator(nbins=5).tick_values(vmin, 0.0)
    pos_ticks = MaxNLocator(nbins=4).tick_values(0.0, vmax)
    vmin, vmax = neg_ticks[0], pos_ticks[-1]
    cbar_ticks = np.concatenate([neg_ticks[neg_ticks < 0.0],
                                 pos_ticks[pos_ticks >= 0.0]])

    norm_V = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    cmap_V = LinearSegmentedColormap.from_list("bwr_zero", ["red", "white", "blue"])

    fig, axes = plt.subplots(3, 2, figsize=(10, 13.5), layout="constrained")

    for i, (ax_V, (name, _), V0) in enumerate(zip(axes[0], columns, V0s)):
        pcm = ax_V.pcolormesh(X, Y, V0, cmap=cmap_V, norm=norm_V, shading="auto")
        ax_V.contour(X, Y, V0, levels=[0.0], colors="black", linewidths=2)
        ax_V.contour(X, Y, np.asarray(r), levels=[0.0], colors="green", linewidths=1)
        ax_V.contour(X, Y, -np.asarray(q), levels=[0.0], colors="red", linewidths=1)
        ax_V.set_aspect("equal")
        ax_V.set_xlabel(r"$x$", fontsize=25)
        ax_V.set_ylabel(r"$y$", fontsize=25)
        if i == 0:
            ax_V.set_title("True value function\n"+ r"$V_{\mathrm{GRA}}[r,q,l;T]$", fontsize=16.5)
        if i == 1:
            ax_V.set_title("Naive value function\n" + r"$\max\{V_{\mathrm{RA}}[r,q;T], V_{\mathrm{A}}[q;T]\}$", fontsize=16.5)

    fig.colorbar(pcm, ax=axes[0], shrink=0.85, label=r"$V(x, 0)$",
                 ticks=cbar_ticks)

    # single legend shared by the three closed-loop panels, drawn in the
    # otherwise-empty lower-left slot; the disturbance entry is a real arrow
    def _legend_arrow(legend, orig_handle, xdescent, ydescent, width, height,
                      fontsize):
        return FancyArrow(0, 0.5 * height, width, 0,
                          length_includes_head=True,
                          head_width=0.75 * height,
                          color=orig_handle.get_facecolor())

    legend_handles = [
        Line2D([], [], color="green", lw=2, label="landing pad"),
        Line2D([], [], color="red", lw=2, label="dead zone"),
        Line2D([], [], marker="o", color="orange", linestyle="", label="antenna"),
        Line2D([], [], color="black", linestyle="-", lw=2,
               label="closed loop (GRA)"),
        Line2D([], [], color="black", linestyle=":", lw=2,
               label="closed loop (RA)"),
        Line2D([], [], color="black", linestyle="--", lw=2,
               label="closed loop (A)"),
        FancyArrow(0, 0, 1, 0, color="purple", alpha=0.6,
                   label="optimal disturbance"),
    ]

    # the reach task ends upon reaching: cut the GRA trajectory at the first
    # sample where the pad SDF exceeds 0.9
    traj_GRA = cl.states
    reached = np.nonzero(warf.pad_sdf(traj_GRA) > 0.9)[0]
    if reached.size:
        traj_GRA = traj_GRA[:reached[0] + 1]

    # one closed-loop panel per value function; the AT problem has no pads,
    # so they are not drawn there
    cl_panels = [
        (axes[1, 0], traj_GRA, V, "-", True,
         "Closed loop under GRA value"),
        (axes[1, 1], cl_RA.states, V_RA, ":", True,
         "Closed loop under RA value"),
        (axes[2, 1], cl_A.states, V_A, "--", False,
         "Closed loop under A value"),
    ]
    axes[2, 0].axis("off")

    for ax_cl, traj, Vf, linestyle, show_pads, title in cl_panels:
        warf.plot_value_functions(grid.states, ax_cl, show_pads=show_pads)

        Xq, Yq, Du, Dv = optimal_disturbance_field(model, grid, Vf)
        ax_cl.quiver(Xq, Yq, Du, Dv, color="purple", alpha=0.6,
                     angles="xy", scale_units="xy", scale=2.4, width=0.004)

        ax_cl.plot(traj[:, 0], traj[:, 1], color="black", linestyle=linestyle,
                   lw=2)

        ax_cl.plot(*traj[0], "o", color="black")
        ax_cl.plot(*traj[-1], "x", color="black", markersize=8)

        ax_cl.set_xlim(X_MIN, X_MAX)
        ax_cl.set_ylim(Y_MIN, Y_MAX)
        ax_cl.set_xlabel(r"$x$", fontsize=25)
        ax_cl.set_ylabel(r"$y$", fontsize=25)
        ax_cl.set_title(title, fontsize=16.5)

    axes[2, 0].legend(
        handles=legend_handles, loc="center", fontsize=16.8,
        handler_map={FancyArrow: HandlerPatch(patch_func=_legend_arrow)},
    )

    fig.savefig("/Users/dylanhirsch/Desktop/value_comparison.pdf")
    plt.show()
