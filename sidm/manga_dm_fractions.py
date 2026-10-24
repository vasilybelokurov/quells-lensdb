"""Step 0 of the SIDM study: how much dark matter is there where sub-arcsecond lenses form?

Number-density-weighted mean dark-matter fraction f_DM(<r) = 1 - M_*(<r)/M_tot(<r) (3D, enclosed)
at r = 1, 2, 3 kpc (the Einstein radii of 0.1-1" lenses at z_l ~ 0.5-1), for MaNGA DynPop galaxies
(manga_population.py: stars = MGE x JAM M/L, gNFW halo) split by Sersic index and sigma_e.

Result (2026-10-24):
    early types (n >= 2.5) sigma_e 70-100   N=317   0.14 0.19 0.25
    early types sigma_e 100-160             N=1111  0.08 0.11 0.15
    early types sigma_e > 160               N=1810  0.04 0.06 0.07
    discs (n < 2.5) sigma_e 70-160          N=1891  0.39 0.41 0.43
    python sidm/manga_dm_fractions.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
from lenscore.lensing import R3  # noqa: E402

d = np.load(paths.RAW / "manga" / "manga_profiles.npz", allow_pickle=True)
mt, ms, n, sig, sn = d["mt"], d["ms"], d["n"], d["sigma"], d["sersic_n"]
good = np.isfinite(sig)


def f_dm(r, sel):
    i = np.searchsorted(R3, r)
    return np.average(1 - ms[sel, i] / mt[sel, i], weights=n[sel])


for label, sel in [("early types (n >= 2.5) sigma_e 70-100", (sn >= 2.5) & (sig >= 70) & (sig < 100)),
                   ("early types sigma_e 100-160", (sn >= 2.5) & (sig >= 100) & (sig <= 160)),
                   ("early types sigma_e > 160", (sn >= 2.5) & (sig > 160)),
                   ("discs (n < 2.5) sigma_e 70-160", (sn < 2.5) & (sig >= 70) & (sig <= 160))]:
    sel &= good
    print(f"{label:40s} N={sel.sum():5d}  f_DM(<1, 2, 3 kpc) = " + " ".join(f"{f_dm(r, sel):.2f}" for r in (1, 2, 3)))
