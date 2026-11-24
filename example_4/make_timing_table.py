"""
Timing table for the two approaches of Example 4: number of HJ solves and total
computation time (wall clock, JIT excluded), built from the timings saved by
baseline.py in data/comparison.npz. Writes timing_table.tex.

baseline.py calls write_table itself after timing both approaches; this script
regenerates the table from the saved timings without recomputing anything.
"""

import numpy as np

import compute_value as cv


def write_table(t_prop, t_base, path=None):
    table = "\n".join([
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Approach & HJ solves & Total time (s) \\",
        r"\midrule",
        rf"Efficient & {len(t_prop)} & {np.sum(t_prop):.1f} \\",
        rf"Naive & {len(t_base)} & {np.sum(t_base):.1f} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]) + "\n"
    path = cv.HERE / "timing_table.tex" if path is None else path
    path.write_text(table)
    return table


if __name__ == "__main__":

    data = np.load(cv.HERE / "data" / "comparison.npz")
    print(write_table(data["times_prop"], data["times_base"]))
