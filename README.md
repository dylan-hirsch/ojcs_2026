# Extending and Unifying the Fundamental Tasks in Hamilton-Jacobi Reachability Analysis

Code for the numerical examples in the paper and its supplemental material.
Every value function is computed with
[`hj_reachability`](https://github.com/StanfordASL/hj_reachability) (v0.7.0).
Shared dynamics and helpers live in `src/util/`. The scripts listed below import them.

## Setup

```bash
uv sync
```

Run each script from the repository root with `uv run python <path>`. The exception is the
scripts in `example_2/related_supplement/`: run those from inside that folder, because they import
each other.

## Scripts → figures

Figures 1 and 2 (the graphical abstract and the counter-example sketch) are schematics, so no
script makes them.

### Main text

| Figure | Example | Script(s) | Output |
|---|---|---|---|
| **Fig. 3** | Example 1: drone over water (Sec. V.A) | `example_1/compute_value.py` | `example_1/value_comparison.pdf` (true vs. naive value function and closed loop) |
| **Fig. 4** | Example 2: warehouse robots (Sec. V.B) | `example_2/compute_value.py` | `example_2/warehouse.pdf`, plus the animation `example_2/closed_loop.gif` |
| **Fig. 5** | Example 3: optimal dosing (Sec. V.C) | `example_3/compute_value.py` | `example_3/PK.pdf` (value functions stitched across phases with Algorithm 1) |
| **Fig. 6** | Example 4: 4D Dubins car with a timed temporal-logic task | `example_4/compute_value.py` → `example_4/plot_results.py` | `compute_value.py` writes `data/values.npz` and `data/closed_loop.npz`, plus `closed_loop.png`, `signals.png` and `closed_loop.gif`. `plot_results.py` builds Fig. 6 as `results.pdf` / `results.png`: the t = 0 value slice and one closed loop per outcome |

Each script saves its outputs next to itself, whatever directory you run it from.

Approximation study for Example 2, mentioned in Sec. V.B:

| Script | Purpose |
|---|---|
| `example_2/related_supplement/approximation_t0.py` | Exact value vs. right-hand ramp approximation at t = 0 for ε ∈ {10, 1, 0.1}. It prints an error table and saves `approximation_t0_values.npz`. It makes no figure |

### Supplemental material

| Figure / Table | Topic | Script(s) | Output |
|---|---|---|---|
| **Fig. S1**, Table S6 | Example 5: canoe in a windy lake, a GRA task using ℓ, r and q (Sec. S.X) | `example_5/compute_value.py` → `example_5/plot_results.py` | `data/values.npz`, then `results.pdf` (Fig. S1) and `diagnostics.pdf` (time series for each method, not in the paper). Table S6 is `example_5/numerics_table.tex` |
| **Fig. S2** | Error over time of the left- vs. right-hand ramp approximations (Sec. S.XI.A) | `example_2/related_supplement/approximation_over_time.py` | `approximation_over_time.pdf` (+ `.npz`) |
| **Fig. S3** | Left- vs. right-hand approximations at t = 40 on a 2D slice (Sec. S.XI.A) | `example_2/related_supplement/approximation.py` → `example_2/related_supplement/plot_approximation.py` | `approximation_values.npz`, then `approximation_slice.pdf` |
| **Fig. S4** | Ramp-based vs. Fourier-based approximation (Sec. S.XI.B) | `example_2/related_supplement/fourier_approximation.py` | `fourier_approximation.pdf` (+ `fourier_approximation_values.npz`) |
| **Fig. S5**, Table S2 | 5D Dubins car: GRA-TVP vs. the simplified HJ-TVPs of Table 3 | `supp_example/compute_value_dubins5d.py` → `supp_example/plot_dubins5d.py` | `timing_table_dubins5d.tex` (Table S2) and `dubins5d_slices.npz`, then `supp_value_comparison_dubins5d.pdf` (Fig. S5) |

Tables S4–S8 list the grids and solver settings each script uses.
