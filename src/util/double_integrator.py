import jax.numpy as jnp
from hj_reachability import dynamics, sets


class DoubleIntegratorDynamics(dynamics.ControlAndDisturbanceAffineDynamics):
    """
    Planar double integrator x1' = x2, x2' = u + d with bounded scalar
    control and disturbance.
    """

    def __init__(
        self,
        u_max = 1.,
        d_max = 0.3,
        control_mode="max",
        disturbance_mode="min",
    ):
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
        return jnp.array([state[1], 0.0])

    def control_jacobian(self, state, time):
        return jnp.array([[0.0], [1.0]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[0.0], [1.0]])
