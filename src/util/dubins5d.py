import jax.numpy as jnp
from hj_reachability import dynamics, sets


class Dubins5DDynamics(dynamics.ControlAndDisturbanceAffineDynamics):
    """
    Five-state Dubins car with state (x, y, v, theta, omega):

        x'     = v cos(theta)
        y'     = v sin(theta)
        v'     = u_1 + d
        theta' = omega
        omega' = u_2

    The controller sets the forward and angular accelerations u = (u_1, u_2)
    in the ball |u|_2 <= u_max. The disturbance perturbs the forward
    acceleration by d in [-d_max, d_max].
    """

    def __init__(
        self,
        u_max = 1.0,
        d_max = 0.25,
        control_mode="max",
        disturbance_mode="min",
    ):
        self.u_max = u_max
        self.d_max = d_max

        control_space = sets.Ball(
            jnp.array([0.0, 0.0]), self.u_max
        )
        disturbance_space = sets.Box(
            jnp.array([-self.d_max]), jnp.array([self.d_max])
        )

        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        _, _, v, theta, omega = state
        return jnp.array([
            v * jnp.cos(theta),
            v * jnp.sin(theta),
            0.0,
            omega,
            0.0,
        ])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[0.0], [0.0], [1.0], [0.0], [0.0]])
