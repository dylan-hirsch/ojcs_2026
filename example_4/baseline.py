"""
Example 4 baseline: a correct but naive decomposition of the same task.

Instead of solving the downstream unordered reach-avoid (URA) task once and
then attaching a single upstream GRA via Theorem 1 (compute_value.py), the
baseline folds target 0 (grocery store) and the truck into each of targets
1, 2, 3. For k = 1, 2, 3 define the space-time target

    R_k := {(x, t) : x in target 0, t in [0, 5]}  u  {(x, 5) : x in truck}
           u  {(x, t) : x in target k, t in [5, 10]},

i.e. target function r0(x) during the upstream phase [0, 5], the truck at the
single instant t = 5, and r_k(x) during the downstream phase [5, 10] (each of
these is a time-invariant signed distance function; only the phase in which it
is used depends on time). Reaching target 0 before time 5, or being at the
truck at time 5, hits all three R_k at once, so

    rho_psi(x, 0) = J_URA[{R1, R2, R3}, q; 10](x, 0),

and the value function can be computed by Algorithm 2 of the arXiv note on the
full horizon [0, 10], with no intermediate hand-off between tasks. Algorithm 2
needs one reach-avoid value function V_K = V_RA[r~_K, q; 10] per nonempty
subset K of {1, 2, 3} (2^3 - 1 = 7 of them), where

    r~_K = R_k                                       if K = {k},
    r~_K = max_{k in K} min{R_k, V_{K \\ {k}}}        otherwise.

Because R_k switches from r0 to r_k at t = 5, each of these reach-avoid
problems is in turn split as in Example 2 (Theorem 1 of the OJCS paper):

  * downstream: V_K on [5, 10]  = V_RA[r~_K|_[5,10], q; 10]
  * upstream:   V_K on [0, 5]   = V_GRA[l~_K, r~_K|_[0,5], q; 5],
                l~_K := max{ l_K, V_K(., 5) },
                l_K  := max{ l_truck, r~_K(., 5) },

where l_K is the terminal-set function of the upstream GRA: being at the truck
at t = 5 completes every R_k, and r~_K(., 5) carries the upstream target to the
switching time.

So the baseline performs 2 x 7 = 14 HJ solves (7 reach-avoid solves on
[5, 10] and 7 GRA solves on [0, 5]), versus 7 + 1 = 8 for the proposed
approach. The 7 downstream solves are identical in both approaches; the
difference is that the baseline must carry every subset K back through the
upstream phase, while the proposed approach carries back only the full set.

This script runs both pipelines in the same process (after a JIT warm-up),
times every solve, and checks that both give the same value function at t = 0.
"""

import itertools
import time as timer

import jax.numpy as jnp
import numpy as np

import compute_value as cv
from make_timing_table import write_table


def timed(log, label, fn, *args):
    start = timer.perf_counter()
    out = fn(*args)
    log.append((label, timer.perf_counter() - start))
    print(f"  {label:32s} {log[-1][1]:6.1f} s")
    return out


def subsets(N):
    for size in range(1, N + 1):
        for K in itertools.combinations(range(N), size):
            yield frozenset(K)


def name(K):
    return "{" + ",".join(str(k + 1) for k in sorted(K)) + "}"


def proposed(model, grid, times_down, times_up, rs, q, l_truck, log):
    """Algorithm 2 on [5, 10] for targets 1-3, then one GRA via Theorem 1."""
    r0, rs_down = rs[0], rs[1:]
    V_down = {}
    for K in subsets(len(rs_down)):
        if len(K) == 1:
            (k,) = K
            r = rs_down[k]
        else:
            r = np.max([np.minimum(rs_down[k], V_down[K - {k}]) for k in K], axis=0)
        V_down[K] = timed(log, f"RA  [5,10] V_{name(K)}",
                          cv.compute_value_ra, model, grid, times_down, r, q)

    full = frozenset(range(len(rs_down)))
    l_tilde = np.maximum(l_truck, V_down[full][-1])
    V_up = timed(log, "GRA [0,5]  V_GRA (Theorem 1)",
                 cv.compute_value, model, grid, times_up, r0, q, l_tilde)
    return V_up


def baseline(model, grid, times_down, times_up, rs, q, l_truck, log):
    """Algorithm 2 on [0, 10] with targets R_k, each RA split at t = 5."""
    r0, rs_down = rs[0], rs[1:]
    V_down, V_up = {}, {}
    for K in subsets(len(rs_down)):
        # composite target on each phase: R_k = r_k on [5,10], r0 on [0,5]
        if len(K) == 1:
            (k,) = K
            r_down, r_up = rs_down[k], r0
        else:
            r_down = np.max([np.minimum(rs_down[k], V_down[K - {k}]) for k in K], axis=0)
            r_up = np.max([np.minimum(r0, V_up[K - {k}]) for k in K], axis=0)

        # downstream reach-avoid on [5, 10]
        V_down[K] = timed(log, f"RA  [5,10] V_{name(K)}",
                          cv.compute_value_ra, model, grid, times_down, r_down, q)

        # upstream GRA on [0, 5] (Theorem 1 / Example 2):
        # l_K = max{l_truck, r~_K(., 5)}
        l = np.maximum(l_truck, r_up[0] if np.ndim(r_up) > grid.ndim else r_up)
        l_tilde = np.maximum(l, V_down[K][-1])
        V_up[K] = timed(log, f"GRA [0,5]  V_{name(K)}",
                        cv.compute_value, model, grid, times_up, r_up, q, l_tilde)

    return V_up[frozenset(range(len(rs_down)))]


def summarize(log):
    t = np.array([s for _, s in log])
    return len(t), t.sum(), t.mean(), t.std(ddof=1)


if __name__ == "__main__":

    model, grid, q, rs = cv.build_problem()
    times_down = np.linspace(cv.T_FINAL, cv.T_SWITCH, cv.TIME_RES)
    times_up = np.linspace(cv.T_SWITCH, 0.0, cv.TIME_RES)
    q = jnp.asarray(q)
    l_truck = cv.truck_sdf(grid)
    print(f"grid shape {grid.shape} ({np.prod(grid.shape):,} nodes), "
          f"{cv.TIME_RES} time slices per phase")

    # JIT warm-up so that neither pipeline is charged for compilation
    start = timer.perf_counter()
    v = jnp.asarray(rs[0])
    cv._gra_step(model, grid, times_up[0], v, times_up[0] - 1e-3, v, v, q).block_until_ready()
    print(f"JIT warm-up: {timer.perf_counter() - start:.1f} s\n")

    print("Proposed (Algorithm 2 downstream + Theorem 1 upstream):")
    log_prop = []
    V_prop = proposed(model, grid, times_down, times_up, rs, q, l_truck, log_prop)

    print("\nBaseline (Algorithm 2 over [0,10], each RA split at t = 5):")
    log_base = []
    V_base = baseline(model, grid, times_down, times_up, rs, q, l_truck, log_base)

    # both must give the same value function (up to numerical error)
    v0_prop, v0_base = V_prop[-1], V_base[-1]
    err = np.abs(v0_prop - v0_base)
    feasible = np.asarray(q) > 0
    print("\nAgreement at t = 0:")
    print(f"  max |V_prop - V_base|            = {err.max():.3e}")
    print(f"  max |V_prop - V_base| where q > 0 = {err[feasible].max():.3e}")
    print(f"  states with different sign       = "
          f"{np.sum((v0_prop > 0) != (v0_base > 0)):,} of {v0_prop.size:,}")

    n_p, tot_p, mean_p, std_p = summarize(log_prop)
    n_b, tot_b, mean_b, std_b = summarize(log_base)
    print("\nTiming (wall clock, JIT excluded):")
    print(f"  {'method':10s} {'solves':>6s} {'total [s]':>10s} {'per solve [s]':>15s}")
    print(f"  {'proposed':10s} {n_p:6d} {tot_p:10.1f} {mean_p:8.1f} +/- {std_p:.1f}")
    print(f"  {'baseline':10s} {n_b:6d} {tot_b:10.1f} {mean_b:8.1f} +/- {std_b:.1f}")
    print(f"  speed-up of proposed over baseline: {tot_b / tot_p:.2f}x")

    # save the t = 0 value functions and timings for plot_results.py
    np.savez(
        cv.HERE / "data" / "comparison.npz",
        V_prop_t0=v0_prop, V_base_t0=v0_base,
        times_prop=np.array([t for _, t in log_prop]),
        times_base=np.array([t for _, t in log_base]),
    )

    write_table([t for _, t in log_prop], [t for _, t in log_base])
