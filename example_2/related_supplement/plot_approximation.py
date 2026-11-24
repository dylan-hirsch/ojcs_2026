"""
Plots the left- vs. right-hand approximations from approximation.py at the
switching time t = 40 on a 2D slice of the state space.

Slice: robot B is held fixed at the centre of the shelf (already in the
target), and robot A's position varies over the warehouse floor. Wherever
robot A is too far from the shelf to arrive in the remaining 20 seconds
(V_RA[r_A, q; 60](., 40) < 0, outlined in purple), the true value is still
positive because robot B is in the target -- the right-hand approximation
recovers this, while the left-hand approximation does not.

Rows: eps = 1, 0.1, 0.01. Columns: left-hand, right-hand approximation.
Run approximation.py first to produce approximation_values.npz.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from util.warehouse_builder import Warehouse, Wall, Shelf

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "approximation_values.npz"
OUT_FILE = HERE / "approximation_slice.pdf"

B_POS = (5.0, 8.5)       # robot B at the centre of the shelf
V_CLIP = -2.0            # colour scale floor (values below are saturated)


def build_warehouse():
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
    return warehouse


if __name__ == "__main__":
    data = np.load(DATA_FILE)
    n = int(data["space_res"])
    axis = np.linspace(data["grid_lo"][0], data["grid_hi"][0], n)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    grid_states = np.stack([X, Y], axis=-1)

    # grid indices of robot B's fixed position
    iB = tuple(int(np.argmin(np.abs(axis - p))) for p in B_POS)
    sl = lambda V: V[:, :, iB[0], iB[1]]

    V_true = sl(data["V_true"])
    V_A_only = sl(data["V_RA_A_40"])   # can robot A reach the shelf in 20 s?
    epsilons = data["epsilons"]

    norm = TwoSlopeNorm(vmin=V_CLIP, vcenter=0.0, vmax=float(V_true.max()))
    levels = np.linspace(V_CLIP, norm.vmax, 31)
    warehouse = build_warehouse()

    fig, axes = plt.subplots(
        len(epsilons), 2, figsize=(9, 13), sharex=True, sharey=True,
        layout="constrained",
    )
    for row, eps in enumerate(epsilons):
        for col, name in enumerate(["left", "right"]):
            ax = axes[row, col]
            V = sl(data[f"V_{name}_eps{eps:g}"])
            err = np.max(np.abs(V - V_true))

            cf = ax.contourf(X, Y, np.maximum(V, V_CLIP), levels=levels,
                             cmap="RdBu", norm=norm, extend="min")
            warehouse.plot_value_functions(grid_states, ax, linewidth=1.5)
            ax.contour(X, Y, V_A_only, levels=[0.0], colors="purple",
                       linewidths=2, linestyles=":")
            ax.contour(X, Y, V_true, levels=[0.0], colors="black",
                       linewidths=2.5, linestyles="--")
            ax.contour(X, Y, V, levels=[0.0], colors="gold", linewidths=2)
            ax.plot(*B_POS, "o", color="tab:orange", markersize=10,
                    markeredgecolor="black")

            side = "Left" if name == "left" else "Right"
            ax.set_title(f"{side}-hand, $\\varepsilon = {eps:g}$\n"
                         f"max error on slice: {err:.2e}")
            ax.set_xlabel("$x_{1,A}$" if row == len(epsilons) - 1 else "")
            ax.set_ylabel("$x_{2,A}$" if col == 0 else "")

    fig.colorbar(cf, ax=axes, shrink=0.5, label="$V(x, 40)$")
    fig.legend(
        handles=[
            Line2D([], [], color="gold", lw=2, label="approx. zero level set"),
            Line2D([], [], color="black", lw=2.5, ls="--", label="true zero level set"),
            Line2D([], [], color="purple", lw=2, ls=":",
                   label="A reaches shelf in 20 s ($V_{RA}[r_A,q;60]=0$)"),
            Line2D([], [], marker="o", color="tab:orange", mec="black", ls="",
                   markersize=10, label="robot B (fixed, on shelf)"),
        ],
        loc="outside lower center", ncol=2,
    )
    fig.savefig(OUT_FILE)
    print(f"saved figure to {OUT_FILE}")
    plt.show()
