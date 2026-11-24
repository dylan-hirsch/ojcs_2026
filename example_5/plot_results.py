"""
Figures for Example 5, built from data/values.npz (written by
compute_value.py).

results.pdf is a single 3 x 2 figure laid out as Supplementary Figure S1:

    GRA value function      | naive max(V_CT, V_RA) value function
    GRA closed loop         | RA closed loop
    shared legend           | CT closed loop

The value maps are at t = 0 and share one color scale; each closed-loop panel
shows the trajectory over the set boundaries with the optimal disturbance
field at t = 0 (arrows share one scale across methods).

diagnostics.pdf holds, for each method, the position, control, disturbance and
set-margin time series of its closed loop, one per page. The script also
prints the RA / CT / GRA scores of each trajectory.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

import compute_value as cv


OUT = cv.HERE / "results.pdf"
DIAGNOSTICS = cv.HERE / "diagnostics.pdf"

plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"
plt.rc("font", size=12)

SET_COLORS = ("#29c437", "#6534CE", "#C55340")  # r, l, q
SET_STYLES = ("--", "-.", "-")
UNSAFE_COLOR = "#F6D7D5"
WIND_COLOR = "#7D409D"
PATH_COLOR = "#15232F"
PATH_STYLES = {"GRA": "-", "RA": ":", "CT": "--"}
COMPONENT_COLORS = ("#0072B2", "#A34D8E")
GOAL_SETS = {"GRA": (0, 1), "RA": (0,), "CT": (1,)}  # indices into (r, l, q)


def load_controllers():
    data = np.load(cv.DATA)
    ts = data["ts"]
    controllers = {
        name: SimpleNamespace(
            name=name,
            path=data[f"{name}_x"],
            u=data[f"{name}_u"],
            d=data[f"{name}_d"],
            V0=data[f"{name}_V0"],
        )
        for name in cv.METHODS
    }
    for c in controllers.values():
        if c.path.shape != (len(ts), 2) or c.u.shape != c.path.shape:
            raise ValueError("Expected two-dimensional states and controls.")
        if not np.all(np.isfinite(c.path)):
            raise ValueError(f"The {c.name} closed-loop path contains nonfinite states.")
    return ts, controllers


def sampled_indices(axis, limits, count, interior=False):
    first = max(int(interior), int(np.searchsorted(axis, limits[0], side="right")) - 1)
    last = min(len(axis) - 1 - int(interior),
               int(np.searchsorted(axis, limits[1], side="left")))
    return np.unique(np.linspace(first, last, min(count, last - first + 1)).round().astype(int))


def sample(field, i, j):
    return np.asarray(field[i[:, None], j[None, :]])


def zero_contour(ax, field, color, coords, linestyle="-", linewidth=1.4):
    finite = field[np.isfinite(field)]
    if finite.size and finite.min() < 0.0 < finite.max():
        ax.contour(*coords, np.ma.masked_invalid(field), levels=[0.0],
                   colors=[color], linestyles=linestyle, linewidths=linewidth)


def decorate_space(ax, coords, boundaries, bounds, shown=(0, 1, 2)):
    """Draw the zero level sets of the (r, l, q) fields whose indices are in `shown`."""
    for k in shown:
        zero_contour(ax, boundaries[k], SET_COLORS[k], coords, SET_STYLES[k])
    ax.set(xlim=bounds[0], ylim=bounds[1], xlabel="$y_1$", ylabel="$y_2$")
    ax.set_aspect("equal", adjustable="box")


def panel_title(ax, title, subtitle):
    ax.set_title(f"{title}\n", fontsize=13, fontweight="semibold")
    ax.set_title(subtitle, loc="left", fontsize=9)


def colorbar_ticks(vmin, vmax, min_gap=0.1):
    """
    Ticks for a TwoSlopeNorm colorbar centered at 0: nice values on each side
    plus both ends (vmin, vmax). Interior ticks closer than `min_gap` (as a
    fraction of the bar) to an end or to 0 are dropped so labels don't overlap.
    """
    def side(lo, hi):
        return [t for t in MaxNLocator(nbins=4).tick_values(lo, hi) if lo < t < hi]

    neg = [t for t in side(vmin, 0.0) if t / vmin > 2 * min_gap and 1 - t / vmin > 2 * min_gap]
    pos = [t for t in side(0.0, vmax) if t / vmax > 2 * min_gap and 1 - t / vmax > 2 * min_gap]
    return [vmin, *neg, 0.0, *pos, vmax]


def max_speed(field):
    speeds = np.linalg.norm(field, axis=-1)
    finite = speeds[np.isfinite(speeds)]
    return float(finite.max()) if finite.size else 0.0


def main():
    T = cv.T
    model = cv.make_model()
    grid = cv.make_grid()
    set_fields = cv.set_functions(grid)
    coordinates = [np.asarray(axis) for axis in grid.coordinate_vectors]
    ts, controllers = load_controllers()
    tt = ts + T

    # Common view of the closed-loop panels: all three closed loops plus a margin.
    points = np.concatenate([c.path for c in controllers.values()])
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2
    half_span = np.maximum((hi - lo) / 2 + 5, 1)
    limits = [(max(float(axis[0]), mid - span), min(float(axis[-1]), mid + span))
              for axis, mid, span in zip(coordinates, center, half_span)]

    ix, iy = [sampled_indices(axis, extent, 400)
              for axis, extent in zip(coordinates, limits)]
    xx, yy = np.meshgrid(coordinates[0][ix], coordinates[1][iy], indexing="ij")
    fields = [sample(field, ix, iy) for field in set_fields]

    # Optimal disturbance velocity at t = 0 on a coarse set of interior nodes,
    # using the same first-order centered gradient as the controller.
    qi, qj = [sampled_indices(axis, extent, 18, interior=True)
              for axis, extent in zip(coordinates, limits)]
    qx, qy = np.meshgrid(coordinates[0][qi], coordinates[1][qj], indexing="ij")
    arrow_states = jnp.asarray(np.stack((qx, qy), axis=-1).reshape(-1, 2))
    field_time = -T  # solver time -T is the paper's t = 0

    def velocity_field(value):
        center = sample(value, qi, qj)
        grad_x = 0.5 * ((center - sample(value, qi - 1, qj)) / grid.spacings[0]
                        + (sample(value, qi + 1, qj) - center) / grid.spacings[0])
        grad_y = 0.5 * ((center - sample(value, qi, qj - 1)) / grid.spacings[1]
                        + (sample(value, qi, qj + 1) - center) / grid.spacings[1])

        def optimal_velocity(state, gradient):
            disturbance = model.optimal_disturbance(state, field_time, gradient)
            return model.disturbance_jacobian(state, field_time) @ disturbance

        gradients = jnp.asarray(np.stack((grad_x, grad_y), axis=-1).reshape(-1, 2))
        velocity = np.asarray(jax.vmap(optimal_velocity)(arrow_states, gradients))
        velocity = velocity.reshape(*qx.shape, 2)
        return np.where((np.isfinite(grad_x) & np.isfinite(grad_y))[..., None],
                        velocity, np.nan)

    velocities = {name: velocity_field(c.V0) for name, c in controllers.items()}
    reference = max(max_speed(v) for v in velocities.values())  # shared arrow scale
    arrow_length = min(high - low for low, high in limits) / 24.0

    interpolate = jax.jit(jax.vmap(grid.interpolate, in_axes=(None, 0)))

    def disturbance_velocity(state, time, disturbance):
        return model.disturbance_jacobian(state, time) @ disturbance

    fig, axes = plt.subplots(3, 2, figsize=(11, 12.5), layout="constrained")

    # --- closed-loop panels and diagnostics ---
    closed_loop_axes = {"GRA": axes[1, 0], "RA": axes[1, 1], "CT": axes[2, 1]}
    sets_handles = None
    with PdfPages(DIAGNOSTICS) as pdf:
        for name in ("RA", "CT", "GRA"):
            c = controllers[name]
            path = c.path
            state_jax = jnp.asarray(path)
            r_path, l_path, q_path = [
                np.asarray(interpolate(field, state_jax)) for field in set_fields
            ]
            wind_path = np.asarray(jax.vmap(disturbance_velocity)(
                state_jax, jnp.asarray(ts), jnp.asarray(c.d)))

            ax = closed_loop_axes[name]
            panel_title(ax, f"Closed loop under the {name} value",
                        r"Initial disturbance field at $t=0$")
            ax.set_axisbelow(True)
            ax.grid(color="#CDD5DE", linewidth=0.6, alpha=0.55)
            safe = fields[2] >= 0
            ax.contourf(xx, yy, fields[2] < 0, levels=[0.5, 1.5], colors=[UNSAFE_COLOR])
            # goal regions relevant to this controller: RA has only the reach
            # set r, CT only the terminal set l, GRA both
            goals = GOAL_SETS[name]
            for k in goals:
                ax.contourf(xx, yy, (fields[k] >= 0) & safe, levels=[0.5, 1.5],
                            colors=[SET_COLORS[k]], alpha=0.12)
            decorate_space(ax, (xx, yy), fields, limits, shown=(*goals, 2))
            velocity = velocities[name]
            finite = np.all(np.isfinite(velocity), axis=-1)
            if reference > 0.0 and finite.any():
                arrows = ax.quiver(qx[finite], qy[finite], velocity[finite, 0],
                                   velocity[finite, 1], color=WIND_COLOR, alpha=0.72,
                                   angles="xy", scale_units="xy",
                                   scale=reference / arrow_length, width=0.0033,
                                   pivot="middle", minlength=0, zorder=2)
                ax.quiverkey(arrows, 0.99, 1.025, reference, f"{reference:.2g} units/s",
                             coordinates="axes", labelpos="W",
                             fontproperties={"size": 9})
            ax.plot(*path.T, color="white", linewidth=5.2, zorder=4)
            ax.plot(*path.T, color=PATH_COLOR, linestyle=PATH_STYLES[name],
                    linewidth=2.8, dash_capstyle="round", zorder=5)
            ax.scatter(*path[0], s=70, facecolor="white", edgecolor=PATH_COLOR,
                       linewidth=1.8, zorder=6)
            ax.scatter(*path[-1], s=76, marker="X", facecolor=PATH_COLOR,
                       edgecolor="white", linewidth=1.1, zorder=6)

            for key, title, ylabel, values, labels, colors in (
                ("position", "Position", "State coordinate", path,
                 ("$y_1$", "$y_2$"), COMPONENT_COLORS),
                ("control", "Held control", "Control input", c.u,
                 ("$u_1$", "$u_2$"), COMPONENT_COLORS),
                ("disturbance", "Realized disturbance velocity", "Velocity (units/s)",
                 wind_path, ("$w_1$", "$w_2$"), COMPONENT_COLORS),
                ("sets", "Set margins", "Signed margin",
                 np.column_stack((r_path, l_path, q_path)),
                 (r"Reach $r$", r"Terminal $\ell$", r"Safety $q$"), SET_COLORS),
            ):
                dfig, panel = plt.subplots(figsize=(8.2, 3.9), layout="constrained")
                dfig.suptitle(f"{name} | {title}", fontsize=16, fontweight="semibold")
                panel.axhline(0.0, color="#88919A", linewidth=0.8)
                styles = SET_STYLES if key == "sets" else ("-", "--")
                for column, label, color, style in zip(values.T, labels, colors, styles):
                    panel.plot(tt, column, label=label, color=color, linestyle=style,
                               linewidth=2.1,
                               drawstyle="steps-post" if key == "control" else "default")
                panel.set(xlim=(0, T), xlabel="Elapsed time (s)", ylabel=ylabel)
                panel.set_axisbelow(True)
                panel.grid(color="#CDD5DE", linewidth=0.7, alpha=0.65)
                panel.legend(loc="best", fontsize=10)
                if key == "sets":
                    sets_handles = panel.get_legend_handles_labels()[0]
                pdf.savefig(dfig)
                plt.close(dfig)

            safe_prefix = np.minimum.accumulate(q_path)
            rat_score = np.max(np.minimum(r_path, safe_prefix))
            rct_score = min(l_path[-1], np.min(q_path))
            print(f"Drone {name} | min q={np.min(q_path):.3f}, max r={np.max(r_path):.3f}, "
                  f"terminal l={l_path[-1]:.3f}")
            print(f"RA trajectory score: {rat_score:.4f}")
            print(f"CT trajectory score: {rct_score:.4f}")
            print(f"GRA trajectory score: {max(rat_score, rct_score):.4f}")

    # --- top row: GRA vs naive value functions at t = 0 ---
    # Enlarge the sampled domain so the value maps contain data throughout the view.
    value_limits = [
        (max(float(axis[0]), (low + high) / 2 - 1.5 * (high - low) / 2),
         min(float(axis[-1]), (low + high) / 2 + 4 * (high - low) / 2))
        for axis, (low, high) in zip(coordinates, limits)
    ]
    value_limits[1] = (
        max(float(coordinates[1][0]), value_limits[1][0] - 1.0),
        min(float(coordinates[1][-1]), value_limits[1][1] - 1.0),
    )
    value_ix, value_iy = [sampled_indices(axis, extent, 400)
                          for axis, extent in zip(coordinates, value_limits)]
    value_coords = np.meshgrid(coordinates[0][value_ix], coordinates[1][value_iy],
                               indexing="ij")
    value_boundaries = [sample(field, value_ix, value_iy) for field in set_fields]
    gra = sample(controllers["GRA"].V0, value_ix, value_iy)
    naive = np.maximum(sample(controllers["RA"].V0, value_ix, value_iy),
                       sample(controllers["CT"].V0, value_ix, value_iy))
    valid = np.concatenate([field[np.isfinite(field)] for field in (gra, naive)])
    lo, hi = (float(valid.min()), float(valid.max())) if valid.size else (-1.0, 1.0)
    eps = max(abs(lo), abs(hi), 1.0) * 1e-6
    norm = TwoSlopeNorm(vmin=min(lo, -eps), vcenter=0.0, vmax=max(hi, eps))

    for ax, label, field, formula in (
        (axes[0, 0], "GRA", gra, r"$V_{\mathrm{GRA}}[\ell,r,q;T]$"),
        (axes[0, 1], "Naive", naive, r"$\max(V_{\mathrm{CT}},V_{\mathrm{RA}})$"),
    ):
        panel_title(ax, f"{label} value function", formula + r" at $t=0$")
        mesh = ax.pcolormesh(*value_coords, np.ma.masked_invalid(field),
                             cmap="RdBu", norm=norm, shading="auto")
        decorate_space(ax, value_coords, value_boundaries, value_limits)
        zero_contour(ax, field, PATH_COLOR, value_coords, linewidth=1.9)
        cbar = fig.colorbar(mesh, ax=ax, label="Value (white = 0)", shrink=0.84)
        cbar.set_ticks(colorbar_ticks(norm.vmin, norm.vmax))
        cbar.ax.yaxis.set_major_formatter(lambda t, _: f"{t:.3g}")
        cbar.minorticks_off()

    # --- bottom left: shared legend ---
    spatial_handles = [
        Line2D([], [], color=color, ls=style, lw=1.7, label=label)
        for color, style, label in zip(
            SET_COLORS, SET_STYLES,
            (r"Reach boundary $r=0$", r"Terminal boundary $\ell=0$",
             r"Safety boundary $q=0$"),
        )
    ]
    spatial_handles.extend([
        Patch(facecolor=UNSAFE_COLOR, edgecolor=SET_COLORS[2],
              label=r"Unsafe region $q<0$"),
        Line2D([], [], color=PATH_COLOR, linewidth=1.9, label=r"Zero value $V=0$"),
    ])
    trajectory_handles = [
        Line2D([], [], color=PATH_COLOR, ls=style, lw=2.8, dash_capstyle="round",
               label=f"{method} trajectory")
        for method, style in PATH_STYLES.items()
    ]
    trajectory_handles.extend([
        Line2D([], [], color=PATH_COLOR, marker="o", ls="none",
               markerfacecolor="white", markeredgewidth=1.4, label="Start"),
        Line2D([], [], color=PATH_COLOR, marker="X", ls="none", label="End"),
    ])
    if reference > 0.0:
        trajectory_handles.append(Line2D(
            [], [], color=WIND_COLOR, marker=">", lw=1.2,
            label=r"Optimal disturbance $G_d d^*$",
        ))
    legend_ax = axes[2, 0]
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[*spatial_handles, *trajectory_handles, *sets_handles],
        loc="center", fontsize=12, frameon=True, framealpha=1, edgecolor="#DDDDDD",
        borderpad=0.7, handlelength=2.4,
    )

    fig.savefig(OUT)
    plt.close(fig)
    print(f"Saved {OUT}")
    print(f"Saved {DIAGNOSTICS}")


if __name__ == "__main__":
    main()
