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
from util.pk import PKDynamics
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

    # specify the number of voxels to divide the spatial and temporal axes
    SPACE_RES = 21
    TIME_RES = 6

    # Specify PDE grid corners
    X_MIN = -.1
    Y_MIN = -.1

    X_MAX = 2.
    Y_MAX = 2.

    # Specify thresholds
    THRESH_THERAPEUTIC = 1.
    THRESH_TOXIC = 1.

    # Initial state and number of steps to take in closed loop
    X0 = (0., 0., 0., 0.)
    STEPS = 10


    # build the dynamics model 

    # build the grid 
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box([X_MIN, Y_MIN, X_MIN, Y_MIN],
                    [X_MAX, Y_MAX, X_MAX, Y_MAX]),
        [SPACE_RES, SPACE_RES, SPACE_RES, SPACE_RES],
        periodic_dims=None
    )

    # compute value functions
    Vs = []
    models = []
    timess = []
    b = 10
    a = 8
    r = grid.states[..., 2] - THRESH_THERAPEUTIC
    q = THRESH_TOXIC - grid.states[..., 3]
    l = r
    for i in range(5):
        times = np.linspace(b, a, TIME_RES)
        if i % 2:
            model = PKDynamics(u_max = 0.0)
        else:
            model = PKDynamics(u_max = 1.0)
        V = compute_value(model, grid, times, r, q, l)
        l = V[-1,...]
        Vs.append(V)
        models.append(model)
        timess.append(times)
        b = b - 2
        a = a - 2

    # Compute closed loop trajectories forward in time using the value function
    CLs = []
    for V, model, times in zip(Vs[::-1], models[::-1], timess[::-1]):
        cl = ClosedLoopTrajectory(model, grid, times, V, X0, steps = STEPS)
        X0 = cl.states[-1,:]
        CLs.append(cl)

    # Plot closed-loop trajectory, inputs, and disturbances
    # Stitch the per-phase trajectories into one forward-time history
    traj = np.vstack([CLs[0].states] + [cl.states[1:] for cl in CLs[1:]])
    t_traj = np.concatenate([CLs[0].times] + [cl.times[1:] for cl in CLs[1:]])
    t_u = np.concatenate([cl.times[:-1] for cl in CLs])
    us = np.concatenate(
        [np.array([cl.u(t) for t in cl.times[:-1]]) for cl in CLs]
    )
    ds = np.concatenate(
        [np.array([cl.d(t) for t in cl.times[:-1]]) for cl in CLs]
    )

    # append the final sample so step plots extend to the end of the horizon
    t_step = np.append(t_u, t_traj[-1])
    u_step = np.vstack([us, us[-1:]])
    d_step = np.vstack([ds, ds[-1:]])

    fig, (ax_drug, ax_resp, ax_u, ax_d) = plt.subplots(
        4, 1, figsize=(6.5, 10), sharex=True, layout="constrained"
    )

    ax_drug.plot(t_traj, traj[:, 0], color="tab:blue", label="$D_1$")
    ax_drug.plot(t_traj, traj[:, 1], color="tab:orange", label="$D_2$")
    ax_drug.set_ylabel("drug concentration")
    ax_drug.legend()

    ax_resp.plot(t_traj, traj[:, 2], color="tab:green", label="$E$ (effect)")
    ax_resp.plot(t_traj, traj[:, 3], color="tab:red", label="$Y$ (toxicity)")
    ax_resp.axhline(THRESH_THERAPEUTIC, color="tab:green", linestyle="--",
                    label="therapeutic threshold")
    ax_resp.axhline(THRESH_TOXIC, color="tab:red", linestyle=":",
                    label="toxic threshold")
    ax_resp.set_ylabel("response")
    ax_resp.legend()

    ax_u.step(t_step, u_step[:, 0], where="post", color="tab:blue",
              label="$u_1$")
    ax_u.step(t_step, u_step[:, 1], where="post", color="tab:orange",
              label="$u_2$")
    ax_u.set_ylabel("dose rate")
    ax_u.set_ylim(-0.05, max(m.u_max for m in models) + 0.05)
    ax_u.legend()

    ax_d.step(t_step, d_step[:, 0], where="post", color="tab:blue",
              label="$d_1$")
    ax_d.step(t_step, d_step[:, 1], where="post", color="tab:orange",
              label="$d_2$")
    ax_d.set_xlabel("time")
    ax_d.set_ylabel("disturbance")
    ax_d.legend()

    # shade the phases in which dosing is unavailable
    for cl, model in zip(CLs, models[::-1]):
        if model.u_max == 0.0:
            for ax in (ax_drug, ax_resp, ax_u, ax_d):
                ax.axvspan(cl.times[0], cl.times[-1], color="gray",
                           alpha=0.15, zorder=0)

    fig.savefig(HERE / "PK.pdf")
    plt.show()
