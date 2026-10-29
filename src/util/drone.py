import jax.numpy as jnp
from flax import struct
from hj_reachability import dynamics, sets


@struct.dataclass
class BallProduct(sets.BoundedSet):
    """
    Cartesian product of two Euclidean balls, so each factor is bounded
    independently (e.g. per-robot speed limits).
    """
    ball_a: sets.Ball
    ball_b: sets.Ball

    def extreme_point(self, direction):
        n = self.ball_a.ndim
        return jnp.concatenate([
            self.ball_a.extreme_point(direction[..., :n]),
            self.ball_b.extreme_point(direction[..., n:]),
        ], axis=-1)

    @property
    def bounding_box(self):
        box_a = self.ball_a.bounding_box
        box_b = self.ball_b.bounding_box
        return sets.Box(jnp.concatenate([box_a.lo, box_b.lo], axis=-1),
                        jnp.concatenate([box_a.hi, box_b.hi], axis=-1))


class DroneDynamics(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(
        self,
        u_max = 1.,
        d_max_NE = 1.5,
        d_max_SW = 1.2,
        control_mode="max",
        disturbance_mode="min",
    ):
        self.u_max = u_max
        self.d_max_NE = d_max_NE
        self.d_max_SW = d_max_SW

        control_space = sets.Ball(
            jnp.array([0.0, 0.0]), self.u_max
        )
        disturbance_space = sets.Box(
            jnp.array([-self.d_max_SW, -self.d_max_SW]), jnp.array([self.d_max_NE, self.d_max_NE])
        )

        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        f = jnp.zeros((2,1))

        return f.reshape((2, ))

    def control_jacobian(self, state, time):

        g = jnp.eye(2)

        return g

    def disturbance_jacobian(self, state, time):
        h = jnp.eye(2)

        return h
