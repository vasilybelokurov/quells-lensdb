"""Diagnostics of the matched SIDM branches (jeans.branch) over the forecast's halo grid:
monotonicity of X(r1) (assumed when interpolating along a branch), whether each branch ends at a fold
or runs to r200, and sigma/m at the end (X_end / cosmic age at z_l).

    python sidm/check_branches.py
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jeans as J
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lenscore import cosmology as cosmo  # noqa: E402

CELLS = [(lm, z, a) for lm in (11.0, 11.6, 12.2, 12.8, 13.4) for z in (0.25, 0.85, 1.45) for a in (2.16, 1.0)]


def run(cell):
    lm, z, a = cell
    m200, ms, re = J.galaxy(lm, z, a)
    r200 = J.r200(m200, z)
    b = J.branch(m200, z, ms, re, np.geomspace(0.02, r200, 100))
    return cell, bool(np.all(np.diff(b[:, 3]) > 0)), b[-1, 0] >= 0.999 * r200, b[-1, 3] / cosmo.COSMO.age(z)


if __name__ == "__main__":
    with ProcessPoolExecutor(13) as ex:
        res = list(ex.map(run, CELLS))
    print("log M200   z    alpha  X(r1) monotonic  end        sigma/m at end [cm^2/g]")
    for (lm, z, a), mono, full, s in res:
        print(f"  {lm:5.1f}  {z:5.2f}  {a:5.2f}  {str(mono):16s} {'r200' if full else 'fold':10s} {s:8.2f}")
    print(f"all monotonic: {all(r[1] for r in res)}; branches reaching r200: {sum(r[2] for r in res)}/{len(res)}")


def show_non_monotonic(lm, z, a):
    """Print where X(r1) decreases along the branch of one halo."""
    m200, ms, re = J.galaxy(lm, z, a)
    b = J.branch(m200, z, ms, re, np.geomspace(0.02, J.r200(m200, z), 100))
    d = np.diff(b[:, 3])
    for i in np.nonzero(d <= 0)[0]:
        print(f"  log M200 {lm} z {z} alpha {a}: X decreases from {b[i, 3]:.6g} to {b[i + 1, 3]:.6g} "
              f"(relative {d[i] / b[i, 3]:+.1e}) between r1 = {b[i, 0]:.4g} and {b[i + 1, 0]:.4g} kpc "
              f"(branch ends at r1 = {b[-1, 0]:.4g}, X = {b[-1, 3]:.4g})")
