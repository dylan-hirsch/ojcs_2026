# fundamental numerical library
from pathlib import Path

import hj_reachability as hj
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

# my dynamics
from util.robot import RobotDynamics
from util.warehouse_builder import Warehouse, Wall, Shelf
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


HERE = Path(__file__).resolve().parent


if __name__ == "__main__":

    # specify the dynamics parameters
    uA_max = 10/60
    uB_max = 5/60
    d_max = 2/60

    # specify the time horizons of the problem
    B = 60
    S = 40

    # Initial state and number of steps to take in closed loop
    X0 = (5.0, 1.5, 5.0, 3.5)
    STEPS = 100

    # specify the number of voxels to divide the spatial and temporal axes
    SPACE_RES = 41
    TIME_RES_1 = 21
    TIME_RES_2 = 41

    # Specify PDE grid corners
    X_MIN = 0
    Y_MIN = 0

    X_MAX = 10
    Y_MAX = 10

    # Build the warehouse
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

    # build the dynamics model 
    model = RobotDynamics(uA_max = uA_max, uB_max = uB_max , d_max = d_max)

    # build the grid 
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box([X_MIN, Y_MIN, X_MIN, Y_MIN],
                    [X_MAX, Y_MAX, X_MAX, Y_MAX]),
        [SPACE_RES, SPACE_RES, SPACE_RES, SPACE_RES],
        periodic_dims=None
    )
    times_downstream = np.linspace(B, S, TIME_RES_1)
    times_upstream = np.linspace(S, 0, TIME_RES_2)


    wall_sdf = warehouse.wall_sdf(grid.states[:,:,0,0,:])
    shelf_sdf = warehouse.shelf_sdf(grid.states[:,:,0,0,:])
    r_A = np.broadcast_to(shelf_sdf[:, :, None, None], [SPACE_RES]*4)
    r_B = np.broadcast_to(shelf_sdf[None, None, :, :], [SPACE_RES]*4)
    q_A = np.broadcast_to(wall_sdf[:, :, None, None], [SPACE_RES]*4)
    q_B = np.broadcast_to(wall_sdf[None, None, :, :], [SPACE_RES]*4)
    positions = np.asarray(grid.states[:,:,0,0,:])
    SAFE_DIST = 0.5
    q_AB = np.linalg.norm(
        positions[:, :, None, None, :] - positions[None, None, :, :, :], axis=-1
    )
    r = np.maximum(r_A, r_B)
    q = np.minimum.reduce([q_A, q_B, q_AB]) - SAFE_DIST

    # compute the downstream value function
    V0 = compute_value(model, grid, times_downstream, r_A, q, r_A)
    V = compute_value(model, grid, times_upstream, r, q, V0[-1,...])
    cl = ClosedLoopTrajectory(model, grid, times_upstream, V, X0, steps = STEPS)
    cl0 = ClosedLoopTrajectory(model, grid, times_downstream, V0, cl.states[-1,:], steps = STEPS)
    
    # Visualize solution: trajectories on the left, collision distance on the
    # right
    TEXT_SIZE = 1.3 * 1.25 * plt.rcParams["font.size"]
    LEGEND_SIZE = TEXT_SIZE
    TITLE_SIZE = 1.2 * TEXT_SIZE  # matplotlib titles default to 1.2x body text
    MATH_LABEL_SIZE = 1.3 * 1.6 * plt.rcParams["font.size"]
    fig, (ax, ax_d) = plt.subplots(
        1, 2, figsize=(12, 6), layout="constrained"
    )
    ax = warehouse.plot_value_functions(grid.states[:,:,0,0,:], ax, linewidth=4)

    # project the closed-loop trajectory onto each robot's position, with the
    # line hue darkening over time so equal shades mark simultaneous positions
    traj = np.vstack([cl.states, cl0.states[1:]])
    t_traj = np.concatenate([cl.times, cl0.times[1:]])
    norm = Normalize(t_traj[0], t_traj[-1])

    for cols, base_cmap, label in [((0, 1), "Blues", "robot A"),
                                   ((2, 3), "Oranges", "robot B")]:
        # truncate the light end of the colormap so early times stay visible
        cmap = LinearSegmentedColormap.from_list(
            base_cmap + "Trunc", plt.get_cmap(base_cmap)(np.linspace(0.35, 1.0, 256))
        )

        points = traj[:, cols].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=4,
                            capstyle="round")
        lc.set_array(t_traj[:-1])
        ax.add_collection(lc)

        ax.plot(*traj[0, list(cols)], "o", color=cmap(0.0))
        ax.plot(*traj[-1, list(cols)], "x", color=cmap(1.0), markersize=16, mew=4)

    # robot legend in two columns beneath the warehouse map
    ax.legend(handles=[Line2D([], [], color=plt.get_cmap("Blues")(0.7), lw=4, label="robot A"),
                       Line2D([], [], color=plt.get_cmap("Oranges")(0.7), lw=4, label="robot B")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
              fontsize=LEGEND_SIZE)

    dist = np.linalg.norm(traj[:, :2] - traj[:, 2:], axis=1)
    ax_d.plot(t_traj, dist, color="black", lw=3)
    ax_d.axhline(SAFE_DIST, color="red", linestyle="--", lw=3, label="collision distance")
    ax_d.set_xlabel("time")
    ax_d.set_ylabel("inter-robot distance")
    ax_d.set_ylim(bottom=0.0)
    ax_d.legend(fontsize=LEGEND_SIZE)
    # match the square warehouse map so both panels are the same size
    ax_d.set_box_aspect(1)

    # enlarge titles, axis labels, and tick labels
    for a in (ax, ax_d):
        for text in (a.title, a.xaxis.label, a.yaxis.label):
            text.set_fontsize(TITLE_SIZE if text is a.title else TEXT_SIZE)
        a.tick_params(labelsize=TEXT_SIZE)
    # mathtext renders smaller than plain text, so bump the map's labels further
    ax.xaxis.label.set_fontsize(MATH_LABEL_SIZE)
    ax.yaxis.label.set_fontsize(MATH_LABEL_SIZE)

    fig.savefig(HERE / "warehouse.pdf")

    # Animate the closed loop: each robot is a disk of radius SAFE_DIST / 2,
    # so the constraint q_AB >= 0 holds exactly when the disks never touch.
    # A companion plot tracks the inter-robot distance against the threshold.
    fig2, (ax_map, ax_dist) = plt.subplots(
        1, 2, figsize=(11, 5), width_ratios=[1.2, 1], layout="constrained"
    )
    warehouse.plot_value_functions(grid.states[:, :, 0, 0, :], ax_map)

    ax_dist.plot(t_traj, dist, color="black")
    ax_dist.axhline(SAFE_DIST, color="red", linestyle="--", label="collision distance")
    ax_dist.set_xlabel("time")
    ax_dist.set_ylabel("inter-robot distance")
    ax_dist.set_ylim(bottom=0.0)
    ax_dist.legend()

    disk_A = Circle(traj[0, :2], SAFE_DIST / 2, color="tab:blue", alpha=0.5)
    disk_B = Circle(traj[0, 2:], SAFE_DIST / 2, color="tab:orange", alpha=0.5)
    ax_map.add_patch(disk_A)
    ax_map.add_patch(disk_B)
    trail_A, = ax_map.plot([], [], color="tab:blue", lw=2, label="robot A")
    trail_B, = ax_map.plot([], [], color="tab:orange", lw=2, label="robot B")
    cursor = ax_dist.axvline(t_traj[0], color="gray")
    ax_map.legend(loc="lower right")

    def update(f):
        disk_A.center = traj[f, :2]
        disk_B.center = traj[f, 2:]
        trail_A.set_data(traj[:f + 1, 0], traj[:f + 1, 1])
        trail_B.set_data(traj[:f + 1, 2], traj[:f + 1, 3])
        cursor.set_xdata([t_traj[f], t_traj[f]])
        ax_map.set_title(f"t = {t_traj[f]:5.1f}")
        return disk_A, disk_B, trail_A, trail_B, cursor

    anim = FuncAnimation(fig2, update, frames=len(t_traj), interval=50)
    anim.save(HERE / "closed_loop.gif", writer="pillow", fps=20, dpi=90)

    plt.show()
