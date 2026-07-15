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


class RobotDynamics(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(
        self,
        uA_max = 1.0,
        uB_max = 1.0,
        d_max = 0.2,
        control_mode="max",
        disturbance_mode="min",
    ):
        self.uA_max = uA_max
        self.uB_max = uB_max
        self.d_max = d_max

        control_space = BallProduct(
            sets.Ball(jnp.array([0.0, 0.0]), self.uA_max),
            sets.Ball(jnp.array([0.0, 0.0]), self.uB_max),
        )
        disturbance_space = sets.Ball(
            jnp.array([0.0, 0.0, 0.0, 0.0]), self.d_max
        )

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
