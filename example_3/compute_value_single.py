# fundamental numerical library
import hj_reachability as hj
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# my dynamics
from util.pk import PKDynamics
from util.closed_loop import ClosedLoopTrajectory


class ScheduledPKDynamics(PKDynamics):
    """PK dynamics with explicitly time-dependent dosing availability.

    Dosing alternates between available and unavailable in blocks of length
    `phase`, starting available at t = 0. The schedule enters the dynamics
    through the control Jacobian, so a single backward solve over the full
    horizon captures the switching without manually stitching value
    functions together at the points of discontinuity.
    """

    def __init__(self, phase=2., **kwargs):
        self.phase = phase
        super().__init__(**kwargs)

    def dosing_gate(self, time):
        # closed on the right so dosing is active at the end of an on-phase,
        # matching the phase boundaries of the stitched formulation
        return jnp.where(jnp.mod(time, 2 * self.phase) <= self.phase, 1., 0.)

    def control_jacobian(self, state, time):
        return self.dosing_gate(time) * super().control_jacobian(state, time)


def compute_value(model, grid, times, r, q, l):

    def value_postprocessor(t, V):
        return jnp.minimum(jnp.maximum(V, r), q)

    # specify the accuracy with which to solve the HJ Partial Differential Equation
    solver_settings = hj.SolverSettings.with_accuracy(
        "medium", value_postprocessor=value_postprocessor
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

    # specify the number of voxels to divide the spatial and temporal axes;
    # TIME_RES is chosen so the slice times (spacing 10/11) are deliberately
    # OUT of phase with the switching times t = 2, 4, 6, 8, while keeping a
    # total slice count comparable to the stitched run in compute_value.py
    SPACE_RES = 21
    N_PHASES = 5
    PHASE = 2.
    TIME_RES = 12
    STEPS_PER_PHASE = 100

    T_FINAL = N_PHASES * PHASE
    STEPS = N_PHASES * STEPS_PER_PHASE

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

    # build the dynamics model with the dosing schedule baked in
    model = ScheduledPKDynamics(phase=PHASE, u_max=1.0)

    # build the grid
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box([X_MIN, Y_MIN, X_MIN, Y_MIN],
                    [X_MAX, Y_MAX, X_MAX, Y_MAX]),
        [SPACE_RES, SPACE_RES, SPACE_RES, SPACE_RES],
        periodic_dims=None
    )
    times = np.linspace(T_FINAL, 0, TIME_RES)

    # compute the value function over the whole horizon in one solve
    r = grid.states[..., 2] - THRESH_THERAPEUTIC
    q = THRESH_TOXIC - grid.states[..., 3]
    V = compute_value(model, grid, times, r, q, r)

    # Run a single closed loop over the whole horizon
    cl = ClosedLoopTrajectory(model, grid, times, V, X0, steps=STEPS)
    traj = cl.states
    t_traj = cl.times
    t_u = t_traj[:-1]
    us = np.array([cl.u(t) for t in t_u])
    ds = np.array([cl.d(t) for t in t_u])

    # show the dose rate actually delivered: the optimal control is
    # arbitrary while the gate is closed, so mask it with the schedule
    gates = np.array([model.dosing_gate(t) for t in t_u])
    us_eff = gates[:, None] * us

    # print a numerical summary for comparison with the stitched run
    crossed = traj[:, 2] >= THRESH_THERAPEUTIC
    t_cross = t_traj[np.argmax(crossed)] if crossed.any() else np.nan
    print(f"final state:              {traj[-1]}")
    print(f"E crosses threshold at t: {t_cross:.3f}")
    print(f"max toxicity:             {traj[:, 3].max():.4f}")
    print(f"value along trajectory:   "
          f"min {min(cl.value(t) for t in t_traj):.4f}, "
          f"final {cl.value(t_traj[-1]):.4f}")

    # Plot closed-loop trajectory, inputs, and disturbances
    fig, (ax_drug, ax_resp, ax_u, ax_d) = plt.subplots(
        4, 1, figsize=(6.5, 10), sharex=True, layout="constrained"
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

    t_step = np.append(t_u, t_traj[-1])
    u_step = np.vstack([us_eff, us_eff[-1:]])
    d_step = np.vstack([ds, ds[-1:]])

    ax_u.step(t_step, u_step[:, 0], where="post", color="tab:blue",
              label="$u_1$")
    ax_u.step(t_step, u_step[:, 1], where="post", color="tab:orange",
              label="$u_2$")
    ax_u.set_ylabel("dose rate")
    ax_u.set_ylim(-0.05, model.u_max + 0.05)
    ax_u.legend()

    ax_d.step(t_step, d_step[:, 0], where="post", color="tab:blue",
              label="$d_1$")
    ax_d.step(t_step, d_step[:, 1], where="post", color="tab:orange",
              label="$d_2$")
    ax_d.set_xlabel("time")
    ax_d.set_ylabel("disturbance")
    ax_d.legend()

    # shade the phases in which dosing is unavailable
    for k in range(N_PHASES):
        start = k * PHASE
        if model.dosing_gate(start + PHASE / 2) == 0.:
            for ax in (ax_drug, ax_resp, ax_u, ax_d):
                ax.axvspan(start, start + PHASE, color="gray",
                           alpha=0.15, zorder=0)

    fig.savefig("closed_loop_single.png", dpi=150)
    plt.show()
