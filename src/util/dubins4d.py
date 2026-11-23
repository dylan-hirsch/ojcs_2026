import jax.numpy as jnp
from hj_reachability import dynamics, sets


class Dubins4DDynamics(dynamics.ControlAndDisturbanceAffineDynamics):
    """
    Four-state Dubins car with state (x, y, theta, omega):

        x'     = (v + d) cos(theta)
        y'     = (v + d) sin(theta)
        theta' = omega
        omega' = u

    The controller sets the angular acceleration u in [-u_max, u_max] but has
    no authority over the forward speed. The disturbance perturbs the speed by
    d in [-d_max, d_max]; requiring d_max < v ensures the disturbance can slow
    the car down or speed it up, but never stop or reverse it.
    """

    def __init__(
        self,
        v = 1.0,
        u_max = 1.0,
        d_max = 0.25,
        control_mode="max",
        disturbance_mode="min",
    ):
        if not 0 <= d_max < v:
            raise ValueError("Need 0 <= d_max < v so the car can never reverse.")

        self.v = v
        self.u_max = u_max
        self.d_max = d_max

        control_space = sets.Box(
            jnp.array([-self.u_max]), jnp.array([self.u_max])
        )
        disturbance_space = sets.Box(
            jnp.array([-self.d_max]), jnp.array([self.d_max])
        )

        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        _, _, theta, omega = state
        return jnp.array([
            self.v * jnp.cos(theta),
            self.v * jnp.sin(theta),
            omega,
            0.0,
        ])

    def control_jacobian(self, state, time):
        return jnp.array([[0.0], [0.0], [0.0], [1.0]])

    def disturbance_jacobian(self, state, time):
        theta = state[2]
        return jnp.array([
            [jnp.cos(theta)],
            [jnp.sin(theta)],
            [0.0],
            [0.0],
        ])
