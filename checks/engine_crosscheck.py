"""Cross-check of lenscore's projection and lens quantities against an independent method.

lenscore projects with a shell sum on fixed grids (lensing.PROJ). The reference here is the
Abel-type integral used by lens_small_separation before it moved to lenscore:
    M_cyl(R) = M(rmax) w(rmax) + int_0^T M(R cosh t) / cosh^2 t dt,
    w(r) = 1 - sqrt(1 - R^2/r^2),  T = arccosh(rmax/R),
with theta_E from mean kappa = 1 (root finding) and beta_r = max_{theta<theta_E} theta (kbar - 1).
Profiles: SIS, Hernquist, and NFW + Hernquist (no response and Cautun+20 contraction, from the
halo model in lens_population.py). Same geometry for both (z_l = 0.5, z_s = 2).

Result (2026-10-24): theta_E, beta_r and M_2D agree to <= 0.1%; the mean outer-image separation
is 0.92-1.0 x 2 theta_E for NFW + stars (so 2 theta_E overestimates separations by 2-8%).
    python checks/engine_crosscheck.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lens_population as LP  # noqa: E402
from lenscore import cosmology as cosmo  # noqa: E402
from lenscore import lensing  # noqa: E402
from lenscore import profiles as prof  # noqa: E402

ZL, ZS = 0.5, 2.0
SCR, DL = cosmo.sigma_crit(ZL, ZS)
RMAX = lensing.R3[-1]


def m_cyl(m3, R, rmax, nt=4000):
    R = np.atleast_1d(R)[:, None]
    T = np.arccosh(np.maximum(rmax / R, 1.0))
    t = np.linspace(0, 1, nt)[None, :] * T
    integral = np.trapezoid(m3(R * np.cosh(t)) / np.cosh(t) ** 2, t, axis=1)
    return m3(rmax) * (1 - np.sqrt(np.maximum(1 - (R[:, 0] / rmax) ** 2, 0))) + integral


def reference(m3):
    kbar = lambda th: m_cyl(m3, DL * th * cosmo.ARCSEC, RMAX)[0] / (np.pi * (DL * th * cosmo.ARCSEC) ** 2 * SCR)
    te = brentq(lambda th: np.log(kbar(th)), 1e-4, 50, xtol=1e-7)
    th = te * np.logspace(-5, 0, 300)
    br = max(t * (kbar(t) - 1) for t in th)
    return te, br


def compare(name, m3):
    row = m3(np.minimum(lensing.R3, RMAX))[None, :]
    te1, br1 = reference(m3)
    te2 = lensing.einstein_radius(row, DL, SCR)[0] / cosmo.ARCSEC
    beta_r, sep, _, _ = lensing.images(row, DL, SCR)
    br2 = beta_r[0] / cosmo.ARCSEC
    R = np.array([0.3, 1.0, 3.0, 10.0])
    ratio = np.interp(R, lensing.R2, lensing.project(row)[0]) / m_cyl(m3, R, RMAX)
    print(f"{name:34s} theta_E {te1:.4f} vs {te2:.4f} ({te2 / te1 - 1:+.2%})  beta_r {br1:.4f} vs {br2:.4f} "
          f"({br2 / br1 - 1:+.2%})  M2D ratio at 0.3/1/3/10 kpc " + " ".join(f"{x:.4f}" for x in ratio)
          + f"  sep/2theta_E {np.nanmin(sep) / (2 * te1):.3f}-{np.nanmax(sep) / (2 * te1):.3f}")


sis = lambda s: (lambda r: 2 * s**2 * np.asarray(r, float) / cosmo.G)
hern = lambda M, Re: (lambda r: M * prof.hernquist_m(np.asarray(r, float), Re / 1.8153))
compare("SIS 200 km/s", sis(200.0))
compare("SIS 120 km/s", sis(120.0))
compare("Hernquist 1e11, Re = 3 kpc", hern(1e11, 3.0))
compare("Hernquist 3e10, Re = 1.5 kpc", hern(3e10, 1.5))
for lm, ms, re in [(12.0, 2e10, 2.5), (12.5, 6e10, 3.5), (13.0, 1.5e11, 6.0)]:
    m200 = np.array([10**lm])
    c, r200 = prof.concentration(m200, ZL), prof.r200(m200, ZL)
    for model in ("NFW", "contracted"):
        row = LP.enclosed_profiles(m200, c, r200, np.array([ms]), np.array([re]), model)[0]
        m3 = lambda r, row=row: np.exp(np.interp(np.log(np.asarray(r, float)), np.log(lensing.R3), np.log(row)))
        compare(f"NFW+Hernquist log M={lm} {model}", m3)
