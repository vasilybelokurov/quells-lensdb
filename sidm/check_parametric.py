"""Smoothness of the parametric SIDM model across the forecast's halo-mass grid: progenitor fit (a, c, rms),
F_t, tau, and Einstein radius and cross-section at fixed z_l, for the SLACS-matched alphas of the forecast.
Jumps between neighbouring masses would show up as spikes in the separation distribution.

Result (2026-10-24): with the free fit, a crossed 1 at log M200 <~ 11.05 and >~ 13.45 (z = 0.6) and theta_E
jumped by up to x17 (|d ln theta_E| = 2.8) between neighbouring masses -- f_in's exponent 1/(a - 1); a and c
also stepped every ~0.1 dex because the fit points changed with mass. After restricting a >= 1.05 and fitting
on fixed radii (parametric.py) everything is smooth: max |d ln theta_E| = 0.12-0.22 per 0.05 dex, fit rms
0.035-0.067; c sits at its lower bound 1 for 13/55 masses of the contracted-CDM limit (as in the author's code).

    python sidm/check_parametric.py      -> plots/sidm_check_parametric.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import parametric as P

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
from lenscore import cosmology as cosmo  # noqa: E402
from lenscore import lensing  # noqa: E402
from lenscore import profiles as prof  # noqa: E402

Z = 0.6
LG = np.arange(11.0, 13.71, 0.05)
CASES = [(1.517, 0.0), (1.383, 1.0), (1.211, 5.0)]          # (alpha, sigma/m): pref, param_s1, param_s5
scr, dl = cosmo.sigma_crit(Z)
age = cosmo.COSMO.age(Z)

rows = {k: [] for k in CASES}
for lm in LG:
    m200 = 10**lm
    ms0 = prof.mstar_moster13(m200, Z)
    re = prof.re_shen03(ms0)
    for a, s in CASES:
        h = P.Halo(m200, Z, a * ms0, re)
        tau = h.tau(s, age)
        row = h.m_tot(tau)[None, :]
        te = lensing.einstein_radius(row, dl, scr)[0] / cosmo.ARCSEC
        br = lensing.images(row, dl, scr)[0][0] / cosmo.ARCSEC
        rows[(a, s)].append((h.a, h.c, h.fit_rms, h.Ft, tau, te, br))

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
labels = ["inner slope a", "concentration c", "fit rms (ln M)", "F_t", "tau", "theta_E [arcsec]"]
for (a, s), col in zip(CASES, ["k", "C1", "C3"]):
    v = np.array(rows[(a, s)])
    for i, axx in enumerate(ax.flat):
        axx.plot(LG, v[:, i], color=col, marker=".", ms=3, label=rf"$\alpha$={a}, $\sigma/m$={s:g}")
        axx.set(xlabel=r"log M$_{200}$", ylabel=labels[i])
    d = np.abs(np.diff(np.log(np.maximum(v[:, 5], 1e-6))))
    print(f"alpha {a}, sigma/m {s:g}: a {v[:, 0].min():.2f}-{v[:, 0].max():.2f}, c {v[:, 1].min():.2f}-{v[:, 1].max():.2f} "
          f"(at bound c=1: {np.sum(v[:, 1] < 1.001)}/{len(LG)}), max |d ln theta_E| between neighbours {d.max():.3f} "
          f"at log M = {LG[np.argmax(d)]:.2f}")
ax[1, 2].set_yscale("log")
ax[0, 0].legend(fontsize=7)
fig.suptitle(f"parametric SIDM model across halo mass, z_l = {Z}")
fig.tight_layout()
fig.savefig(paths.PLOTS / "sidm_check_parametric.png", dpi=100)
print(f"wrote {paths.PLOTS / 'sidm_check_parametric.png'}")
