from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D


if __name__ == "__main__":

    # slices saved by compute_value_dubins5d.py
    data = np.load(Path(__file__).with_name("dubins5d_slices.npz"))
    Y1, Y2 = data["Y1"], data["Y2"]
    r, q, l = data["r"], data["q"], data["l"]

    cmap_V = LinearSegmentedColormap.from_list("bwr_zero", ["red", "white", "blue"])

    # set colors matching Fig. 1 of the paper; the obstacle (complement of
    # the constraint set) is drawn light red and dashed
    COLOR_R = "#85cb5a"  # target set (light green)
    COLOR_Q = "#ff8a8a"  # obstacle (light red)
    COLOR_L = "#915bab"  # terminal set (purple)
    STYLE_R, STYLE_Q, STYLE_L = "solid", "dashed", "solid"

    # 5x2 comparison: GRA value function (left) vs simplified HJ-TVP (right),
    # both at the initial time on the (y_1, y_2) slice at fixed (v, theta, omega)
    fig, axes = plt.subplots(5, 2, figsize=(9, 20), layout="constrained")

    subtitles = [
        "Computed by (GRA-TVP)",
        "Computed using Simplified TVP",
    ]

    for row, label, active, V0_GRA, V0_TVP, slice_err in zip(
            axes, data["labels"], data["active"], data["V0_GRA"],
            data["V0_TVP"], data["slice_err"]):
        V0s = [V0_GRA, V0_TVP]
        vmin = min(min(V0.min() for V0 in V0s), -1e-3)
        vmax = max(max(V0.max() for V0 in V0s), 1e-3)

        # pick round tick locations separately on each side of zero, and stretch
        # the color range so the bar starts and ends exactly on round ticks
        neg_ticks = MaxNLocator(nbins=4).tick_values(vmin, 0.0)
        pos_ticks = MaxNLocator(nbins=4).tick_values(0.0, vmax)
        vmin, vmax = neg_ticks[0], pos_ticks[-1]
        cbar_ticks = np.concatenate([neg_ticks[neg_ticks < 0.0],
                                     pos_ticks[pos_ticks >= 0.0]])
        norm_V = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

        for j, (ax, V0) in enumerate(zip(row, V0s)):
            pcm = ax.pcolormesh(Y1, Y2, V0, cmap=cmap_V, norm=norm_V,
                                shading="auto", rasterized=True)
            ax.contour(Y1, Y2, V0, levels=[0.0], colors="black", linewidths=2)
            # only draw the sets whose functions are active for this task
            for show, fn, color, style in zip(active, (r, q, l),
                                              (COLOR_R, COLOR_Q, COLOR_L),
                                              (STYLE_R, STYLE_Q, STYLE_L)):
                if show:
                    ax.contour(Y1, Y2, fn, levels=[0.0], colors=color,
                               linewidths=2, linestyles=style)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$y_1$", fontsize=14)
            ax.set_ylabel(r"$y_2$", fontsize=14)
            title = str(label) + "\n" + subtitles[j]
            if j == 1:
                title += f"\nmax error on slice $=$ {slice_err:.1e}"
            ax.set_title(title, fontsize=10)

        fig.colorbar(pcm, ax=row, shrink=0.85, label=r"$V(x, 0)$",
                     ticks=cbar_ticks, format="%g")

    # single legend shared by all panels, drawn below the grid
    legend_handles = [
        Line2D([], [], color=COLOR_R, lw=2, linestyle=STYLE_R,
               label=r"target set $\{r > 0\}$"),
        Line2D([], [], color=COLOR_Q, lw=2, linestyle=STYLE_Q,
               label=r"obstacle $\{q \leq 0\}$"),
        Line2D([], [], color=COLOR_L, lw=2, linestyle=STYLE_L,
               label=r"terminal set $\{\ell > 0\}$"),
        Line2D([], [], color="black", lw=2, label=r"$\{V(x, 0) = 0\}$"),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=2,
               fontsize=12)
    v_s, theta_s, omega_s = (round(float(data[k]), 3) + 0.0
                             for k in ("v", "theta", "omega"))
    fig.suptitle(rf"$(y_1, y_2)$ slice at $v = {v_s:g}$, $\theta = {theta_s:g}$, "
                 rf"$\omega = {omega_s:g}$",
                 fontsize=13)

    fig.savefig(Path(__file__).with_name("supp_value_comparison_dubins5d.pdf"))
    plt.show()
