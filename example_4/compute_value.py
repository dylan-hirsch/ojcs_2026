"""
Example 4: a 4D Dubins car with a composite timed temporal-logic task.

The car (a grocery-pickup robot) must EITHER reach target 0 (the grocery
store) before time 5, OR be at the truck drop-off site at exactly time 5, OR
(while obeying the constraints until time 5) reach targets 1, 2, and 3 (the
kiwi, tomato and lemon stands), in any order, between times 5 and 10. The constraints are always: stay inside the arena, stay out of
the obstacles, and keep the turn rate |omega| <= OMEGA_MAX.

Every target function r_i(x), and the truck's terminal-set function l(x), is
time invariant: it is just the signed distance to the (spatial) square box of
that site, positive inside. Likewise q(x) is time
invariant. The only time dependence in the task is WHICH phase a target belongs
to: r0 is used in the upstream phase [0, 5], while r1, r2, r3 are used in the
downstream phase [5, 10].

In timed temporal-logic notation the specification is

    psi := ((q U_[0,5] r0) v (q U_[5,5] l))  v  ( AND_{k=1,2,3} q U_[5,10] r_k ),

and (because the dynamics are time invariant) its robustness at t = 0 can be
rewritten as

    rho_psi(x, 0) = max{ J_GRA[l, r0, q; 5](x, 0),
                         min{ J_URA[{r1, r2, r3}, q; 10](x, 5), J_A[q; 5](x, 0) } }

where the terminal-set function l(x) is the signed distance to the truck
drop-off site: the GRA's terminal-set part is exactly "be at the truck at time
5 while obeying the constraints until then". This is exactly the form required by
Theorem 1 of the OJCS paper, with the downstream performance functional
J0 := J_URA[{r1,r2,r3},q;10] evaluated at T := 5. Hence

    V[rho_psi](., 0) = V_GRA[l~, r0, q; 5](., 0),
    l~(x) := max{ l(x), V[J0](x, 5) } = max{ l_truck(x), V_URA(x, 5) }.

So the downstream computation is unchanged by the truck; only the terminal-set
function handed to the upstream GRA changes.

The downstream (unordered reach-avoid) value function V_URA is itself computed
with Algorithm 2 of "Exact Decomposition of Value Functions for Two-Player
Games in Hamilton-Jacobi Reachability", i.e. by backward induction over the
subsets K of the remaining targets:

    V_{k}  = V_RA[r_k, q; 10]
    V_K    = V_RA[ max_{k in K} min{r_k, V_{K \\ {k}}}, q; 10 ].

Although every r_k is time invariant, the composite target that Algorithm 2
builds for |K| >= 2,

    r~_K(x, t) := max_{k in K} min{ r_k(x), V_{K \\ {k}}(x, t) },

inherits a time dependence from the value functions V_{K \\ {k}}(x, t) (the
time left to reach the remaining targets shrinks as t -> 10). The
hj_reachability value postprocessor is a static (compile-time) field, so we
implement our own jitted stepping loop in which this composite target enters as
a traced argument and is linearly interpolated in time between the saved time
slices. The single-target solves (|K| = 1) and the upstream GRA solve use the
time-invariant r_k directly.
"""

import functools
import itertools
import time as timer
from pathlib import Path

import hj_reachability as hj
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

from util.dubins4d import Dubins4DDynamics


HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "values.npz"


# ---------------------------------------------------------------------------
# GRA solver (target may be time invariant, or a time-varying composite
# target built by Algorithm 2)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("dynamics",))
def _gra_step(dynamics, grid, t_start, values, t_end, r_start, r_end, q):
    """
    Integrate the GRA HJ-TVP from t_start back to t_end. The target function is
    linearly interpolated in time between r_start = r(., t_start) and
    r_end = r(., t_end); the constraint q is time invariant.
    """

    def value_postprocessor(t, v):
        w = (t - t_start) / (t_end - t_start)
        r = (1 - w) * r_start + w * r_end
        return jnp.minimum(jnp.maximum(v, r), q)

    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", value_postprocessor=value_postprocessor
    )

    def sub_step(time_values):
        return solver_settings.time_integrator(
            solver_settings, dynamics, grid, *time_values, t_end
        )

    return jax.lax.while_loop(
        lambda time_values: jnp.abs(t_end - time_values[0]) > 0,
        sub_step,
        (t_start, values),
    )[1]


def compute_value(model, grid, times, r, q, l):
    """
    GRA value function V_GRA[l, r, q; T] on the (decreasing) time grid
    `times`, with T = times[0].

    r is either a single array (a time-invariant target such as the signed
    distance functions r_k) or an array with a leading time axis aligned with
    `times` (the composite targets r~_K of Algorithm 2).
    """
    r = jnp.asarray(r)
    if r.ndim == grid.ndim:
        r = jnp.broadcast_to(r, (len(times),) + r.shape)

    v = jnp.minimum(jnp.maximum(l, r[0]), q)
    V = [np.asarray(v)]
    for j in range(len(times) - 1):
        v = _gra_step(model, grid, times[j], v, times[j + 1], r[j], r[j + 1], q)
        V.append(np.asarray(v))
    return np.stack(V)


def compute_value_ra(model, grid, times, r, q):
    """
    Reach-avoid value function V_RA[r, q; T], obtained as the GRA value
    function with terminal-set function l := r(., T).
    """
    r = np.asarray(r)
    l = r[0] if r.ndim > grid.ndim else r
    return compute_value(model, grid, times, r, q, l)


def unordered_reach_avoid(model, grid, times, rs, q, verbose=True):
    """
    Algorithm 2 (Unordered Reach-Avoid) of the arXiv technical note.

    Returns a dict mapping each nonempty frozenset K of target indices to
    V_URA[{r_k : k in K}, q; T] on the time grid `times`. All the intermediate
    value functions are returned because the closed-loop controller switches
    between them as targets are reached.
    """
    N = len(rs)
    V = {}

    # lines 1-2: single-target reach-avoid value functions
    for i in range(N):
        start = timer.perf_counter()
        V[frozenset([i])] = compute_value_ra(model, grid, times, rs[i], q)
        if verbose:
            print(f"  V_{{{i + 1}}} done in {timer.perf_counter() - start:.1f} s")

    # lines 3-6: build up to larger subsets of targets
    for size in range(2, N + 1):
        for K in itertools.combinations(range(N), size):
            K = frozenset(K)
            start = timer.perf_counter()
            # line 5: composite target; time varying only through V_{K \ {k}}
            r_tilde = np.max(
                [np.minimum(rs[k], V[K - {k}]) for k in K], axis=0
            )
            V[K] = compute_value_ra(model, grid, times, r_tilde, q)
            if verbose:
                name = ",".join(str(k + 1) for k in sorted(K))
                print(f"  V_{{{name}}} done in {timer.perf_counter() - start:.1f} s")

    return V


# ---------------------------------------------------------------------------
# Closed-loop simulation with mode switching
# ---------------------------------------------------------------------------

class InterpolatedValue:
    """
    Value function on a time grid with time-linear interpolation of the value
    and its spatial gradient. Gradients are computed lazily per time slice.
    """

    def __init__(self, grid, times, V):
        order = np.argsort(times)
        self.grid = grid
        self.times = np.asarray(times)[order]
        self.V = V[order]
        self._grads = {}
        self._interp = jax.jit(grid.interpolate)

    def _slice_weights(self, t):
        t = np.clip(t, self.times[0], self.times[-1])
        j = int(np.clip(np.searchsorted(self.times, t, side="right") - 1,
                        0, len(self.times) - 2))
        w = (t - self.times[j]) / (self.times[j + 1] - self.times[j])
        return j, w

    def _grad_slice(self, j):
        if j not in self._grads:
            self._grads[j] = self.grid.grad_values(jnp.asarray(self.V[j]))
        return self._grads[j]

    def _clip(self, x):
        x = np.array(x, dtype=float)
        lo, hi = np.asarray(self.grid.domain.lo), np.asarray(self.grid.domain.hi)
        x[[0, 1, 3]] = np.clip(x[[0, 1, 3]], lo[[0, 1, 3]], hi[[0, 1, 3]])
        return x

    def value(self, t, x):
        j, w = self._slice_weights(t)
        x = self._clip(x)
        return float((1 - w) * self._interp(jnp.asarray(self.V[j]), x)
                     + w * self._interp(jnp.asarray(self.V[j + 1]), x))

    def grad(self, t, x):
        j, w = self._slice_weights(t)
        x = self._clip(x)
        return np.asarray((1 - w) * self._interp(self._grad_slice(j), x)
                          + w * self._interp(self._grad_slice(j + 1), x))

    def clear_cache(self):
        self._grads = {}


def rk4_step(model, x, u, d, t, dt):
    f = lambda t, x: np.asarray(model(jnp.asarray(x), u, d, t))
    k1 = f(t, x)
    k2 = f(t + dt / 2, x + dt / 2 * k1)
    k3 = f(t + dt / 2, x + dt / 2 * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(model, V_up, V_down, target_fns, x0, T_switch, T_final, dt,
             substeps=4, stop_at_target0=True, truck_fn=None):
    """
    Sample-and-hold closed loop for the composite task. Both the control and
    the (worst-case) disturbance are read off the gradient of the value
    function of the currently active mode:

      * t < T_switch: the upstream GRA value function. If target 0 is reached
        the task is complete; the simulation then stops, or (if
        stop_at_target0 is False) keeps following the upstream controller until
        T_switch so the state at T_switch can be shown.
      * t = T_switch: if the car is at the truck site (truck_fn > 0), the
        task is complete.
      * t >= T_switch: the URA value function V_K for the set K of targets not
        yet visited (visits before T_switch do not count). When the car enters
        target k in K, it switches to V_{K \\ {k}}; when K is empty the task is
        complete.
    """
    x = np.array(x0, dtype=float)
    t = 0.0
    K = None  # None = upstream phase
    done0 = False
    xs, ts, us, ds, modes, events = [x.copy()], [t], [], [], [], []

    while t < T_final - 1e-9:
        # mode logic (checked at each sample time)
        if K is None and not done0 and target_fns[0](x) > 0:
            events.append((t, "target 0"))
            done0 = True
            if stop_at_target0:
                break
        if done0 and t >= T_switch - 1e-9:
            break
        if (K is None and not done0 and truck_fn is not None
                and abs(t - T_switch) < 1e-9 and truck_fn(x) > 0):
            events.append((t, "truck"))
            break
        if K is None and t >= T_switch - 1e-9:
            K = frozenset(range(len(target_fns) - 1))
            V_up.clear_cache()
        if K is not None:
            reached = [k for k in K if target_fns[k + 1](x) > 0]
            for k in reached:
                events.append((t, f"target {k + 1}"))
                V_down[K].clear_cache()
                K = K - {k}
            if not K:
                break

        Vf = V_up if K is None else V_down[K]
        grad = jnp.asarray(Vf.grad(t, x))
        u = model.optimal_control(jnp.asarray(x), t, grad)
        d = model.optimal_disturbance(jnp.asarray(x), t, grad)

        # do not hold an upstream input across the switching time
        h = dt if K is not None else min(dt, T_switch - t)
        for _ in range(substeps):
            x = rk4_step(model, x, u, d, t, h / substeps)
            t += h / substeps

        xs.append(x.copy())
        ts.append(t)
        us.append(float(u[0]))
        ds.append(float(d[0]))
        modes.append("up" if K is None else ",".join(str(k + 1) for k in sorted(K)))

    return dict(x=np.array(xs), t=np.array(ts), u=np.array(us),
                d=np.array(ds), modes=modes, events=events)


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

# dynamics parameters: speed in [V - D_MAX, V + D_MAX], never reversing
V = 2.0
D_MAX = 0.4
U_MAX = 4.0
OMEGA_MAX = 2.0

# time horizons
T_SWITCH = 5.0
T_FINAL = 10.0

# grid
XY_MAX = 6.0
OMEGA_GRID_MAX = 2.5
XY_RES = 51
THETA_RES = 36
OMEGA_RES = 21
TIME_RES = 26   # time slices per 5-second phase

# arena (square of half-width ARENA), disk obstacles (center, radius) and
# square box targets (center, half-width; areas roughly match the earlier
# disk targets).
# Target 0 sits in the far corner from the cluster of targets 1-3, so from
# much of the arena it cannot be reached by t = 5 and the car must instead
# loiter safely (it cannot stop) and then tour targets 1-3 during [5, 10].
ARENA = 5.5
OBSTACLES = [((1.2, 1.2), 1.3), ((-3.3, -3.3), 0.8)]
TARGETS = [((4.3, 4.3), 0.9),     # target 0
           ((-4.0, -1.5), 0.6),   # target 1
           ((-1.5, -4.0), 0.6),   # target 2
           ((-1.0, -1.0), 0.6)]   # target 3

# truck drop-off site (center, half-width), only usable at exactly t = T_SWITCH.
# It is placed where, at t = 5, neither the grocery store (closed) nor the
# tour of the fruit stands (too far away) can complete the task. The site is
# large enough for the car (which cannot stop) to loiter inside it while it
# waits for t = 5, despite the speed disturbance.
TRUCK = ((3.9, -3.6), 1.3)

# closed-loop initial states (x, y, theta, omega)
INITIAL_STATES = [(-4.5, 4.0, -np.pi / 2, 0.0),   # tours the fruit stands
                  (-1.5, 4.0, 0.0, 0.0),          # reaches the grocery store
                  (-0.7, 2.6, -np.pi / 2, 0.0)]   # meets the truck at t = 5
DT = 0.05


def disk_sdf(points, center, radius):
    """Positive inside the disk."""
    return radius - np.linalg.norm(points - np.asarray(center), axis=-1)


def box_sdf(points, center, half_width):
    """Signed distance to an axis-aligned square box, positive inside."""
    d = np.abs(points - np.asarray(center)) - half_width
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    inside = np.minimum(np.max(d, axis=-1), 0.0)
    return -(outside + inside)


def constraint_xy(points):
    """Positive inside the arena and outside every obstacle."""
    wall = ARENA - np.max(np.abs(points), axis=-1)
    obs = [-disk_sdf(points, c, rad) for c, rad in OBSTACLES]
    return np.minimum.reduce([wall] + obs)


def build_problem():
    model = Dubins4DDynamics(v=V, u_max=U_MAX, d_max=D_MAX)
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([-XY_MAX, -XY_MAX, -np.pi, -OMEGA_GRID_MAX]),
                    jnp.array([XY_MAX, XY_MAX, np.pi, OMEGA_GRID_MAX])),
        [XY_RES, XY_RES, THETA_RES, OMEGA_RES],
        periodic_dims=2,
    )

    states = np.asarray(grid.states)
    xy = states[..., :2]
    omega = states[..., 3]

    q = np.minimum(constraint_xy(xy), OMEGA_MAX - np.abs(omega))
    rs = [box_sdf(xy, c, hw) for c, hw in TARGETS]
    return model, grid, q, rs


def truck_sdf(grid):
    """Terminal-set function l: signed distance to the truck site."""
    return box_sdf(np.asarray(grid.states)[..., :2], *TRUCK)


def solve_all(model, grid, q, rs):
    times_down = np.linspace(T_FINAL, T_SWITCH, TIME_RES)
    times_up = np.linspace(T_SWITCH, 0.0, TIME_RES)
    truck = np.array([*TRUCK[0], TRUCK[1]])

    V_down = None
    if CACHE.exists():
        data = np.load(CACHE)
        if data["shape"].tolist() == list(grid.shape) and data["time_res"] == TIME_RES:
            V_down = {frozenset(int(c) - 1 for c in key.split("_")[1]): data[key]
                      for key in data.files if key.startswith("K_")}
            if "truck" in data.files and np.array_equal(data["truck"], truck):
                print(f"Loading cached value functions from {CACHE}")
                return times_down, times_up, V_down, data["V_up"], data["V_r0"]
            print("Reusing cached downstream value functions; recomputing upstream")

    print(f"grid shape {grid.shape} ({np.prod(grid.shape):,} nodes)")

    # (1) downstream task via Algorithm 2: reach targets 1, 2, 3 in any order
    if V_down is None:
        print("Downstream unordered reach-avoid (Algorithm 2):")
        V_down = unordered_reach_avoid(model, grid, times_down, rs[1:], q)

    # (2) Theorem 1: l~ = max{l, V[J0](., T)} with terminal-set function
    #     l := signed distance to the truck (V_down is stored backward in
    #     time, so [-1] is t = T_SWITCH)
    l = truck_sdf(grid)
    l_tilde = np.maximum(l, V_down[frozenset(range(len(rs) - 1))][-1])

    # (3) upstream GRA: reach target 0 by T_SWITCH, or survive and hand off
    print("Upstream GRA (Theorem 1):")
    start = timer.perf_counter()
    V_up = compute_value(model, grid, times_up, rs[0], q, l_tilde)
    print(f"  V_GRA done in {timer.perf_counter() - start:.1f} s")

    # for comparison only: reaching target 0 alone (no downstream fallback)
    start = timer.perf_counter()
    V_r0 = compute_value_ra(model, grid, times_up, rs[0], q)
    print(f"  V_RA[r0, q; 5] done in {timer.perf_counter() - start:.1f} s")

    CACHE.parent.mkdir(exist_ok=True)
    np.savez(
        CACHE, shape=np.array(grid.shape), time_res=TIME_RES, truck=truck,
        V_up=V_up, V_r0=V_r0,
        **{"K_" + "".join(str(k + 1) for k in sorted(K)): v for K, v in V_down.items()},
    )
    return times_down, times_up, V_down, V_up, V_r0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

TARGET_COLORS = ["tab:green", "tab:purple", "tab:blue", "tab:orange"]
TRUCK_COLOR = "tab:brown"


def event_color(name):
    return TRUCK_COLOR if name == "truck" else TARGET_COLORS[int(name[-1])]


def draw_scene(ax):
    ax.add_patch(Rectangle((-ARENA, -ARENA), 2 * ARENA, 2 * ARENA,
                           fill=False, edgecolor="black", lw=1.5))
    for c, rad in OBSTACLES:
        ax.add_patch(Circle(c, rad, color="tab:red", alpha=0.6))
    for i, (c, hw) in enumerate(TARGETS):
        ax.add_patch(Rectangle((c[0] - hw, c[1] - hw), 2 * hw, 2 * hw,
                               color=TARGET_COLORS[i], alpha=0.35))
        ax.text(*c, str(i), ha="center", va="center", fontsize=11,
                fontweight="bold")
    (c, hw) = TRUCK
    ax.add_patch(Rectangle((c[0] - hw, c[1] - hw), 2 * hw, 2 * hw,
                           color=TRUCK_COLOR, alpha=0.35))
    ax.text(*c, "T", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(-XY_MAX, XY_MAX)
    ax.set_ylim(-XY_MAX, XY_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def plot_regions(ax, grid, V_up, V_r0, theta, omega=0.0):
    """
    Slice of the initial-state sets at a fixed heading: where target 0 can be
    reached by t = 5, where only the fallback tour of targets 1-3 works, and
    where the composite task cannot be guaranteed.
    """
    states = np.asarray(grid.states)
    it = int(np.argmin(np.abs(np.angle(np.exp(1j * (states[0, 0, :, 0, 2] - theta))))))
    io = int(np.argmin(np.abs(states[0, 0, 0, :, 3] - omega)))
    v_up = V_up[-1][:, :, it, io]
    v_r0 = V_r0[-1][:, :, it, io]
    region = np.where(v_r0 > 0, 2, np.where(v_up > 0, 1, 0))
    cmap = ListedColormap(["white", "#c6dbef", "#c7e9c0"])
    xs = states[:, 0, 0, 0, 0]
    ys = states[0, :, 0, 0, 1]
    ax.pcolormesh(xs, ys, region.T, cmap=cmap, vmin=-0.5, vmax=2.5,
                  shading="nearest")
    ax.contour(xs, ys, v_up.T, levels=[0], colors="tab:blue", linewidths=1)
    ax.contour(xs, ys, v_r0.T, levels=[0], colors="tab:green", linewidths=1)
    draw_scene(ax)
    ax.annotate("", xy=(4.8 + 0.6 * np.cos(theta), -4.8 + 0.6 * np.sin(theta)),
                xytext=(4.8, -4.8), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title(rf"$t=0$, $\theta={np.degrees(states[0, 0, it, 0, 2]):.0f}^\circ$, "
                 rf"$\omega={states[0, 0, 0, io, 3]:.1f}$")


def plot_trajectory(ax, sim, cmap_name):
    traj, t = sim["x"], sim["t"]
    points = traj[:, :2].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap_name, norm=Normalize(0, T_FINAL),
                        linewidths=2.2, capstyle="round")
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    ax.plot(*traj[0, :2], "o", color="black", markersize=5)
    ax.plot(*traj[-1, :2], "x", color="black", markersize=7)
    i5 = int(np.argmin(np.abs(t - T_SWITCH)))
    if t[-1] >= T_SWITCH - 1e-9:
        ax.plot(*traj[i5, :2], "s", color="black", markersize=5, fillstyle="none")
    return lc


if __name__ == "__main__":

    model, grid, q, rs = build_problem()
    times_down, times_up, V_down, V_up, V_r0 = solve_all(model, grid, q, rs)

    full = frozenset(range(len(rs) - 1))
    Vf_up = InterpolatedValue(grid, times_up, V_up)
    Vf_down = {K: InterpolatedValue(grid, times_down, v) for K, v in V_down.items()}
    target_fns = [lambda x, c=c, hw=hw: float(box_sdf(x[:2], c, hw))
                  for c, hw in TARGETS]
    truck_fn = lambda x: float(box_sdf(x[:2], *TRUCK))

    sims = []
    for x0 in INITIAL_STATES:
        print(f"x0 = {x0}: V_GRA = {Vf_up.value(0.0, x0):+.3f}, "
              f"V_RA[r0] = {InterpolatedValue(grid, times_up, V_r0).value(0.0, x0):+.3f}")
        sim = simulate(model, Vf_up, Vf_down, target_fns, x0, T_SWITCH, T_FINAL, DT,
                       truck_fn=truck_fn)
        print("   events:", ", ".join(f"{name} @ t={t:.2f}" for t, name in sim["events"]))
        q_traj = np.minimum(constraint_xy(sim["x"][:, :2]),
                            OMEGA_MAX - np.abs(sim["x"][:, 3]))
        print(f"   min constraint value along trajectory: {q_traj.min():+.3f}")
        sims.append(sim)

    # save the closed-loop trajectories for plot_results.py
    np.savez(
        HERE / "data" / "closed_loop.npz",
        **{f"x_{i}": sim["x"] for i, sim in enumerate(sims)},
        **{f"t_{i}": sim["t"] for i, sim in enumerate(sims)},
        **{f"event_times_{i}": np.array([te for te, _ in sim["events"]]) for i, sim in enumerate(sims)},
        **{f"event_names_{i}": np.array([n for _, n in sim["events"]]) for i, sim in enumerate(sims)},
    )

    # --- Figure 1: initial-state regions and closed-loop trajectories ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), layout="constrained")
    for ax, sim, x0, cmap_name in zip(axes, sims, INITIAL_STATES, ["viridis", "viridis"]):
        plot_regions(ax, grid, V_up, V_r0, theta=x0[2], omega=x0[3])
        lc = plot_trajectory(ax, sim, cmap_name)
    fig.colorbar(lc, ax=axes, label="time", shrink=0.8)
    axes[0].legend(handles=[
        Patch(color="#c7e9c0", label="reach 0 by $t=5$"),
        Patch(color="#c6dbef", label="else tour 1,2,3 in $[5,10]$"),
        Line2D([], [], marker="s", ls="", color="black", fillstyle="none", label="$t=5$"),
    ], loc="upper center", fontsize=8, framealpha=0.9)
    fig.savefig(HERE / "closed_loop.png", dpi=200)

    # --- Figure 2: states and inputs over time for each initial state ---
    fig2, axs = plt.subplots(3, len(sims), figsize=(11, 6.5), sharex=True,
                             layout="constrained", squeeze=False)
    for col, sim in enumerate(sims):
        t = sim["t"]
        dist = np.array([[f(x) for f in target_fns] for x in sim["x"]])
        for i in range(len(TARGETS)):
            axs[0, col].plot(t, dist[:, i], color=TARGET_COLORS[i], label=f"$r_{i}$")
        axs[0, col].axhline(0, color="gray", lw=0.8)
        q_traj = np.minimum(constraint_xy(sim["x"][:, :2]),
                            OMEGA_MAX - np.abs(sim["x"][:, 3]))
        axs[1, col].plot(t, q_traj, color="tab:red", label="$q$")
        axs[1, col].plot(t, sim["x"][:, 3], color="black", lw=1, label=r"$\omega$")
        axs[1, col].axhline(0, color="gray", lw=0.8)
        axs[2, col].step(t[:-1], sim["u"], where="post", color="black", label="$u$")
        axs[2, col].step(t[:-1], sim["d"] * U_MAX / D_MAX, where="post",
                         color="tab:gray", ls="--",
                         label=rf"$d$ (scaled $\times{U_MAX / D_MAX:.0f}$)")
        for ax in axs[:, col]:
            ax.axvline(T_SWITCH, color="gray", ls=":", lw=1)
            for te, name in sim["events"]:
                ax.axvline(te, color=event_color(name), ls="--", lw=0.8)
        axs[2, col].set_xlabel("time")
    axs[0, 0].set_ylabel("target functions")
    axs[1, 0].set_ylabel("constraint")
    axs[2, 0].set_ylabel("inputs")
    for ax in axs[:, -1]:
        ax.legend(fontsize=8, loc="upper right")
    fig2.savefig(HERE / "signals.png", dpi=200)

    # --- Animation of the tour trajectory ---
    sim = sims[0]
    traj, t_traj = sim["x"], sim["t"]
    fig3, ax3 = plt.subplots(figsize=(5.5, 5.5), layout="constrained")
    draw_scene(ax3)
    trail, = ax3.plot([], [], color="black", lw=1.5)
    car = ax3.quiver([traj[0, 0]], [traj[0, 1]], [np.cos(traj[0, 2])],
                     [np.sin(traj[0, 2])], angles="xy", scale_units="xy",
                     scale=1.5, color="black", width=0.012)

    def update(f):
        trail.set_data(traj[:f + 1, 0], traj[:f + 1, 1])
        car.set_offsets([traj[f, :2]])
        car.set_UVC([np.cos(traj[f, 2])], [np.sin(traj[f, 2])])
        phase = "upstream" if t_traj[f] < T_SWITCH else "downstream"
        ax3.set_title(f"t = {t_traj[f]:5.2f} ({phase})")
        return trail, car

    anim = FuncAnimation(fig3, update, frames=len(t_traj), interval=50)
    anim.save(HERE / "closed_loop.gif", writer="pillow", fps=20, dpi=90)

    plt.show()
