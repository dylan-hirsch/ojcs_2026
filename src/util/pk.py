import jax.numpy as jnp
from hj_reachability import dynamics, sets


class PKDynamics(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(
        self,
        u_max = 1.,
        d_max = .1,
        alpha_1 = 1.,
        alpha_2 = 1.,
        eta_1 = .2,
        eta_2 = .1,
        eta_3 = 1.,
        kappa_1 = .8,
        kappa_2 = .4,
        kappa_3 = 4.,
        c_1 = 1.,
        c_2 = .1,
        beta = 0.1,
        k_1 = 1.,
        k_2 = 1.,
        gamma = .3,

        control_mode="max",
        disturbance_mode="min",
    ):
        attrs = locals().copy()
        attrs.pop("self")
        self.__dict__.update(attrs)

        control_space = sets.Box(
            jnp.array([0.0, 0.0]), jnp.array([u_max, u_max])
        )

        disturbance_space = sets.Ball(
            jnp.array([0.0, 0.0]), self.d_max
        )

        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        D_1, D_2, E, T = state

        return jnp.array([
            -self.alpha_1 * D_1,
            -self.alpha_2 * D_2,
            jnp.maximum(self.eta_1 * D_1**2 / (self.c_1**2 + D_1**2) +
                self.eta_2 * D_2**2 / (self.c_2**2 + D_2**2) +
                self.eta_3 * D_1**2 * D_2**2 / (self.c_1**2 + D_1**2) / (self.c_2**2 + D_2**2) -
                self.beta * E, 0.),
            jnp.maximum(self.kappa_1 * D_1**2 / (self.k_1**2 + D_1**2) +
                self.kappa_2 * D_2**2 / (self.k_2 + D_2**2) +
                self.kappa_3 * D_1**2 * D_2**2 / (self.k_1**2 + D_1**2) / (self.k_2**2 + D_2**2) -
                self.gamma * T, 0.),
        ])

    def control_jacobian(self, state, time):

        return jnp.array([
            [1., 0.],
            [0., 1.],
            [0., 0.],
            [0., 0.],
        ])

    def disturbance_jacobian(self, state, time):
        
        return jnp.array([
            [-self.alpha_1, 0.],
            [0., -self.alpha_2],
            [0., 0.],
            [0., 0.],
        ])
