"""SIDM probe 3: matched solutions followed in r1 (jeans.branch), X derived from the rate condition.

Haloes at z = 0.6: (log M200, alpha) = (11.8, 1.5), (12.3, 1.0), (12.3, 1.5), (13.0, 1.5), with and
without stars. Prints the solution every few r1 and the largest X reached before the solver fails.
Result (2026-10-24): with stars the branch ends at X = 4.8, 11.5, 6.5, 63 respectively (X still
rising at the end, no turnover in X); without stars it continues to r1 = 150 kpc. The end is a fold
of the model, not a solver failure: see probe_branch_end.py.
    python sidm/probe_continue_r1.py
"""
import jeans as J
import numpy as np

z = 0.6
r1s = np.geomspace(0.02, 150, 140)
for lm, alpha in [(11.8, 1.5), (12.3, 1.0), (12.3, 1.5), (13.0, 1.5)]:
    m200, ms, re = J.galaxy(lm, z, alpha)
    for stars in (True, False):
        b = J.branch(m200, z, ms if stars else 1.0, re, r1s)
        print(f"log M200 {lm}  alpha {alpha}  stars {stars}:  M* {ms:.2e}  R_e {re:.2f} kpc  "
              f"-> {len(b)}/{len(r1s)} r1 values solved, max X = {b[:, 3].max():.2f} at r1 = {b[-1, 0]:.1f} kpc")
        if stars:
            for row in b[::15]:
                print("     r1 %7.3f  rho0 %.3e  sigma0 %6.1f  X %8.3f" % tuple(row))
