import jax.numpy as jnp
from hj_reachability import dynamics, sets


def safety_margin(x, y):
    """
    Signed safety margin q(x, y) of the drone example: positive inside the
    safe corridor between the two V-shaped walls

        y <= 1.745 + |x|   and   y >= |x| - 5.255.
    """
    return jnp.minimum(1.745 + jnp.abs(x) - y, 5.255 - jnp.abs(x) + y)


class DroneWithDriftDynamics(dynamics.ControlAndDisturbanceAffineDynamics):
    """
    Planar drone (single integrator) pushed by a state-dependent wind:

        x' = u + G_d(x) d,   ||u||_2 <= 1,   d in [-1, 1]^2,

    where the wind gain G_d(x) = diag(h, 0.05 h), h = q(x) - 0.4, grows with
    the distance from the walls of the safe corridor.
    """

    def __init__(
        self,
        control_mode="max",
        disturbance_mode="min",
        dxMin=-1.0,
        dxMax=1.0,
        dyMin=-1.0,
        dyMax=1.0,
    ):
        control_space = sets.Ball(jnp.array([0.0, 0.0]), 1.0)
        disturbance_space = sets.Box(
            jnp.array([dxMin, dyMin]),
            jnp.array([dxMax, dyMax]),
        )
        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        return jnp.array([0.0, 0.0])

    def control_jacobian(self, state, time):
        return jnp.array([[1.0, 0.0], [0.0, 1.0]])

    def disturbance_jacobian(self, state, time):
        x, y = state
        hx = safety_margin(x, y) - 0.4
        hy = 0.05 * hx
        return jnp.array([
            [hx, 0.0],
            [0.0, hy],
        ])
