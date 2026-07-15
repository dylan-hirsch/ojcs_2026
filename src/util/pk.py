import jax.numpy as jnp
from hj_reachability import dynamics, sets


class RobotDynamics(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(
        self,
        u_max = 1.,
        d_max = 0.,
        alpha_1 = 1.,
        alpha_2 = 1.,
        eta_1 = 1.,
        eta_2 = 1.,
        kappa = 4.,
        c_1 = 1.,
        D_1 = 1.,
        k_1 = 1.,
        beta = 1.
        k_2 = 1.,
        gamma = .1

        control_mode="max",
        disturbance_mode="min",
    ):
        attrs = locals().copy()
        attrs.pop("self")
        self.__dict__.update(attrs)

        control_space = sets.Box(
            jnp.array([0.0, 0.0]), jnp.array([u_max, u_max])
        )
        )
        disturbance_space = sets.Ball(
            jnp.array([0.0, 0.0]), self.d_max
        )

        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        D_1, D_2, E, T = state

        f = jnp.zeros((4,))
        f[0] = -self.alpha_1 * D_1
        f[1] = -self.alpha_2 * D_2
        f[2] = self.eta_1 * D_1 / (self.c_1 + D_1) + \
                self.eta_2 * D_2 / (self.c_2 + D_2) + \
                self.kappa * D_1 * D_2 / (self.c_1 + D_1) / (self.c_2 + D_2) \
                - self.beta * E
        f[2] = D_1 * D_2 / (self.c_1 + D_1) / (self.c_2 + D_2) \ self.gamma * T
        return f

    def control_jacobian(self, state, time):

        g = jnp.zeros((4,2))

        g[0,0] = 1
        g[1,1] = 1

        return g

    def disturbance_jacobian(self, state, time):
        
        h = jnp.zeros((4,2))

        D_1, D_2, _, _ = state
        
        h[0,0] = -self.alpha_1
        h[1,1] = -self.alpha_2

        return h
