import jax.numpy as jnp
from hj_reachability import dynamics, sets


class RobotDynamics(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(
        self,
        control_mode="max",
        disturbance_mode="min",
        uMax=+1.0,
    ):
        self.uMax = uMax

        control_space = sets.Ball(
            jnp.array([0.0, 0.0]), self.uMax
        )
        disturbance_space = sets.Box(
            jnp.array([0.0]), jnp.array([0.0])
        )  # no disturbance

        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        f = jnp.zeros((4,1))

        return f.reshape((4, ))

    def control_jacobian(self, state, time):

        g = jnp.eye(4)

        return g

    def disturbance_jacobian(self, state, time):
        h = jnp.eye(4)

        return h
