"""
Summary figure for Example 4 (Fig. 6), built from the saved outputs of
compute_value.py:

  * data/values.npz       (value functions)
  * data/closed_loop.npz  (closed-loop trajectories)

Panels (2 x 2):
  (top left)     V(x, y, theta, omega; t = 0), computed with Theorem 1
  (other panels) one closed loop each: touring targets 1, 2, 3 during
                 [5, 10], reaching target 0 (grocery store) before t = 5,
                 and being at the truck site at exactly t = 5

Targets are drawn as their boxes with an emoji: target 1 is a kiwi, target 2
a tomato, target 3 a lemon, target 0 shows all three, and the truck site
shows a delivery truck. The emoji are
rendered with the macOS Apple Color Emoji font; if it is unavailable the
targets are labelled with their numbers instead.
"""

import functools
from fractions import Fraction
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, Rectangle

import compute_value as cv


# slice of the 4D value function shown in the top-left panel
THETA = -np.pi / 2
OMEGA = 0.0
V_MIN, V_MAX = -1.0, 0.6  # color limits of the value plots (clipped beyond)

# where to put the label of each trajectory tick, relative to the tick (x, y)
LABEL_OFFSET = {"tour": (1.4, -0.7), "target 0": (-1.6, -1.1),
                "truck": (-2.2, 0.9)}
EVENT_LABEL = {"target 0": "grocery store", "truck": "truck"}

TARGET_COLORS = cv.TARGET_COLORS

EMOJI_FONT = Path("/System/Library/Fonts/Apple Color Emoji.ttc")
EMOJI = {1: ["\U0001F95D"],                              # kiwi
         2: ["\U0001F345"],                              # tomato
         3: ["\U0001F34B"],                              # lemon
         0: ["\U0001F95D", "\U0001F345", "\U0001F34B"],  # all three
         "truck": ["\U0001F69A"]}                         # delivery truck
TRUCK_COLOR = cv.TRUCK_COLOR


@functools.lru_cache(maxsize=None)
def emoji_image(char):
    """RGBA array of a color emoji, or None if the emoji font is missing."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(str(EMOJI_FONT), 160)  # a native bitmap size
    except (ImportError, OSError):
        return None
    image = Image.new("RGBA", (180, 180), (0, 0, 0, 0))
    ImageDraw.Draw(image).text((0, 0), char, font=font, embedded_color=True)
    return np.asarray(image.crop(image.getbbox()))


def draw_emoji(ax, char, center, size):
    image = emoji_image(char)
    if image is None:
        return False
    h = size / 2
    ax.imshow(image, extent=(center[0] - h, center[0] + h, center[1] - h,
                             center[1] + h), zorder=6, interpolation="antialiased")
    return True


def draw_scene(ax):
    """Arena, obstacles (gray disks) and targets (boxes with emoji)."""
    ax.add_patch(Rectangle((-cv.ARENA, -cv.ARENA), 2 * cv.ARENA, 2 * cv.ARENA,
                           fill=False, edgecolor="black", lw=1.5, zorder=3))
    for c, rad in cv.OBSTACLES:
        ax.add_patch(Circle(c, rad, facecolor="0.55", edgecolor="black",
                            lw=1, zorder=3))
    for i, (c, hw) in enumerate(cv.TARGETS):
        corner = (c[0] - hw, c[1] - hw)
        ax.add_patch(Rectangle(corner, 2 * hw, 2 * hw, facecolor="white",
                               alpha=0.75, zorder=3))
        ax.add_patch(Rectangle(corner, 2 * hw, 2 * hw, fill=False,
                               edgecolor=TARGET_COLORS[i], lw=1.8, zorder=3))
        chars = EMOJI[i]
        if len(chars) == 1:
            placed = [(c, 1.5 * hw)]
        else:  # three small emoji in a triangle
            s = 0.95 * hw
            placed = [((c[0] - 0.48 * hw, c[1] + 0.42 * hw), s),
                      ((c[0] + 0.48 * hw, c[1] + 0.42 * hw), s),
                      ((c[0], c[1] - 0.45 * hw), s)]
        ok = all(draw_emoji(ax, ch, pos, size)
                 for ch, (pos, size) in zip(chars, placed))
        if not ok:
            ax.text(*c, str(i), ha="center", va="center", fontsize=11,
                    fontweight="bold", zorder=6)
    (c, hw) = cv.TRUCK
    corner = (c[0] - hw, c[1] - hw)
    ax.add_patch(Rectangle(corner, 2 * hw, 2 * hw, facecolor="white",
                           alpha=0.75, zorder=3))
    ax.add_patch(Rectangle(corner, 2 * hw, 2 * hw, fill=False,
                           edgecolor=TRUCK_COLOR, lw=1.8, ls="--", zorder=3))
    if not draw_emoji(ax, EMOJI["truck"][0], c, 0.9 * hw):
        ax.text(*c, "truck", ha="center", va="center", fontsize=10,
                fontweight="bold", zorder=6)
    ax.set_xlim(-cv.XY_MAX, cv.XY_MAX)
    ax.set_ylim(-cv.XY_MAX, cv.XY_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def pi_fraction(angle):
    """LaTeX for an angle as a rational multiple of pi, e.g. -\\pi/2."""
    frac = Fraction(float(angle) / np.pi).limit_denominator(36)
    if frac == 0:
        return "0"
    sign = "-" if frac < 0 else ""
    num, den = abs(frac.numerator), frac.denominator
    top = r"\pi" if num == 1 else rf"{num}\pi"
    return sign + (top if den == 1 else rf"{top}/{den}")


def value_slice(grid, V, theta, omega):
    states = np.asarray(grid.states)
    thetas = states[0, 0, :, 0, 2]
    omegas = states[0, 0, 0, :, 3]
    it = int(np.argmin(np.abs(np.angle(np.exp(1j * (thetas - theta))))))
    io = int(np.argmin(np.abs(omegas - omega)))
    return (states[:, 0, 0, 0, 0], states[0, :, 0, 0, 1], V[:, :, it, io],
            thetas[it], omegas[io])


def plot_value(ax, grid, V, title):
    xs, ys, v, theta, omega = value_slice(grid, V, THETA, OMEGA)
    mesh = ax.pcolormesh(xs, ys, v.T, cmap="RdBu", shading="gouraud",
                         norm=TwoSlopeNorm(0.0, V_MIN, V_MAX))
    ax.contour(xs, ys, v.T, levels=[0], colors="black", linewidths=1.2)
    draw_scene(ax)
    ax.set_title(title)
    ax.text(0.98, 0.02,
            rf"$t=0$, $\theta={pi_fraction(theta)}$, $\omega={omega:.0f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    return mesh


def tick(ax, traj, t, t_mark, label, label_offset, length=0.5):
    """Short tick across the trajectory at time t_mark, with a text label."""
    i = int(np.argmin(np.abs(t - t_mark)))
    p = traj[i, :2]
    heading = np.array([np.cos(traj[i, 2]), np.sin(traj[i, 2])])
    normal = np.array([-heading[1], heading[0]])
    ax.plot(*np.stack([p - length / 2 * normal, p + length / 2 * normal]).T,
            color="black", lw=2.5, solid_capstyle="butt", zorder=7)
    ax.annotate(label, xy=p, xytext=p + np.asarray(label_offset),
                ha="center", va="center", fontsize=9, zorder=8,
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8,
                                shrinkA=2, shrinkB=4),
                bbox=dict(facecolor="white", edgecolor="0.3", lw=0.6, pad=2))


def plot_closed_loops(ax, sims, title, norm, cmap="viridis"):
    """Draw one or more closed loops (traj, t, events) on the scene."""
    draw_scene(ax)
    ax.set_title(title)
    for traj, t, events in sims:
        points = traj[:, :2].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2.2,
                            capstyle="round", zorder=4)
        lc.set_array(t[:-1])
        ax.add_collection(lc)
        ax.plot(*traj[0, :2], "o", color="black", markersize=5, zorder=7)

        name = events[0][1] if events else None
        if name == "target 0":
            # the task is complete before t = 5, so the trajectory ends there;
            # mark the arrival time instead
            te = events[0][0]
            tick(ax, traj, t, te, rf"$t={te:.1f}$: {EVENT_LABEL[name]}",
                 LABEL_OFFSET[name])
        elif name == "truck":
            tick(ax, traj, t, cv.T_SWITCH, rf"$t={cv.T_SWITCH:.0f}$: truck",
                 LABEL_OFFSET[name])
        else:
            tick(ax, traj, t, cv.T_SWITCH, rf"$t={cv.T_SWITCH:.0f}$",
                 LABEL_OFFSET["tour"])
    return lc


if __name__ == "__main__":

    model, grid, q, rs = cv.build_problem()

    values = np.load(cv.HERE / "data" / "values.npz")
    closed_loop = np.load(cv.HERE / "data" / "closed_loop.npz")

    # V_up is stored backward in time on [0, T_SWITCH], so [-1] is t = 0
    V0 = values["V_up"][-1]

    sims = []
    for i in range(len(cv.INITIAL_STATES)):
        events = list(zip(closed_loop[f"event_times_{i}"],
                          closed_loop[f"event_names_{i}"]))
        sims.append((closed_loop[f"x_{i}"], closed_loop[f"t_{i}"], events))

    def outcome(sim):
        names = [n for _, n in sim[2]]
        return names[0] if names and names[0] in ("target 0", "truck") else "tour"

    by_outcome = {outcome(sim): sim for sim in sims}
    titles = {"tour": "Closed loop: tour targets 1, 2, 3",
              "target 0": "Closed loop: grocery store",
              "truck": "Closed loop: truck"}

    fig, ((ax_val, ax_tour), (ax_store, ax_truck)) = plt.subplots(
        2, 2, figsize=(10, 9.4), layout="constrained")

    mesh = plot_value(ax_val, grid, V0, "Value function")
    fig.colorbar(mesh, ax=ax_val, shrink=0.9, extend="both",
                 label=r"$V(x, y, \theta, \omega; 0)$")

    norm = Normalize(0.0, cv.T_FINAL)
    lc = None
    for ax, key in ((ax_tour, "tour"), (ax_store, "target 0"), (ax_truck, "truck")):
        if key in by_outcome:
            lc = plot_closed_loops(ax, [by_outcome[key]], titles[key], norm)
        else:
            draw_scene(ax)
            ax.set_title(titles[key] + " (no trajectory)")
    if lc is not None:
        fig.colorbar(lc, ax=[ax_tour, ax_truck], shrink=0.9, label="time")

    fig.savefig(cv.HERE / "results.png", dpi=200)
    fig.savefig(cv.HERE / "results.pdf")
    plt.show()
