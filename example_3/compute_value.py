# fundamental numerical library
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

if __name__ == "__main__":

    # specify the number of voxels to divide the spatial and temporal axes
    SPACE_RES = 21
    TIME_RES = 11

    # Specify PDE grid corners
    X_MIN = 0.
    Y_MIN = 0.

    X_MAX = 2.
    Y_MAX = 2.

    # Specify thresholds
    THRESH_THERAPEUTIC = 1.
    THRESH_TOXIC = 1.

    # Initial state and number of steps to take in closed loop
    X0 = (0., 0., 0., 0.)
    STEPS = 100


    # build the dynamics model 
    model = PKDynamics(u_max = 1.0)

    # build the grid 
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box([X_MIN, Y_MIN, X_MIN, Y_MIN],
                    [X_MAX, Y_MAX, X_MAX, Y_MAX]),
        [SPACE_RES, SPACE_RES, SPACE_RES, SPACE_RES],
        periodic_dims=None
    )
    times = np.linspace(10, 8, TIME_RES)

    # compute the downstream value function
    r = grid.states[..., 2] - THRESH_THERAPEUTIC
    q = THRESH_TOXIC - grid.states[..., 3]
    V = compute_value(model, grid, times, r, q, r)

    # Run closed loop
    cl = ClosedLoopTrajectory(model, grid, times, V, X0, steps=STEPS)
    traj = cl.states
    t_traj = cl.times
    us = np.array([cl.u(t) for t in t_traj[:-1]])

    # Visualize the closed loop: drug concentrations on top, therapeutic
    # effect and toxicity against their thresholds in the middle, and the
    # sample-and-hold dosing schedule on the bottom
    fig, (ax_drug, ax_resp, ax_u) = plt.subplots(
        3, 1, figsize=(6.5, 8), sharex=True, layout="constrained"
    )

    ax_drug.plot(t_traj, traj[:, 0], color="tab:blue", label="$D_1$")
    ax_drug.plot(t_traj, traj[:, 1], color="tab:orange", label="$D_2$")
    ax_drug.set_ylabel("drug concentration")
    ax_drug.legend()

    ax_resp.plot(t_traj, traj[:, 2], color="tab:green", label="$E$ (effect)")
    ax_resp.plot(t_traj, traj[:, 3], color="tab:red", label="$T$ (toxicity)")
    ax_resp.axhline(THRESH_THERAPEUTIC, color="tab:green", linestyle="--",
                    label="therapeutic threshold")
    ax_resp.axhline(THRESH_TOXIC, color="tab:red", linestyle=":",
                    label="toxic threshold")
    ax_resp.set_ylabel("response")
    ax_resp.legend()

    ax_u.step(t_traj, np.concatenate([us[:, 0], us[-1:, 0]]), where="post",
              color="tab:blue", label="$u_1$")
    ax_u.step(t_traj, np.concatenate([us[:, 1], us[-1:, 1]]), where="post",
              color="tab:orange", label="$u_2$")
    ax_u.set_xlabel("time")
    ax_u.set_ylabel("dose rate")
    ax_u.set_ylim(-0.05, model.control_space.hi[0] + 0.05)
    ax_u.legend()

    fig.savefig("closed_loop.png", dpi=150)
    plt.show()
