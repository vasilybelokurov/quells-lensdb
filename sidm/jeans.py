"""Isothermal-Jeans SIDM haloes with baryons (core-expansion phase), after Kaplinghat, Keeley,
Linden & Yu (2014, PRL 113, 021302) and Kaplinghat, Tulin & Yu (2016, PRL 116, 041302).

Halo before self-interactions: NFW (Diemer+19 concentration) times (1 - f), f = M_*/M200, truncated
at r200, no halo response. Stars: Hernquist, a = R_e / 1.8153, M_* = alpha x Moster+13, R_e from
Shen+03 at the Chabrier mass.

Inside r1 the dark matter is isothermal in the total potential,
    rho = rho0 exp(-h),  h = (Phi - Phi0) / sigma0^2,
    dh/dln r = G [M_DM + M_b] / (sigma0^2 r),  dM_DM/dln r = 4 pi r^3 rho0 exp(-h),
integrated with RK4 in ln r from 1e-4 kpc. (rho0, sigma0) are fixed by matching density and
enclosed mass to the CDM halo at r1, and r1 by one scattering per particle over the age t:
    rho_CDM(r1) (sigma/m) (4/sqrt(pi)) sigma0 t = 1   =>   X = (sigma/m) t = sqrt(pi) / (4 sigma0 rho_CDM(r1)).
Solutions are followed in r1 (continuation from small r1, where SIDM ~ CDM) and X is derived.
"""
import numpy as np
from scipy.optimize import root

from lenscore.cosmology import G, rho_crit
from lenscore.profiles import concentration, mstar_moster13

CM2G = 1.989e33 / 3.0857e21**2          # 1 cm^2/g in kpc^2 / Msun
KMS_GYR = 1.0227                        # 1 km/s x 1 Gyr in kpc
R_IN = 1e-4                             # kpc, inner start of the integration


def re_shen03(mstar):
    """Early-type half-light radius [kpc], Shen et al. (2003, MNRAS 343, 978): 2.88e-6 M^0.56."""
    return 2.88e-6 * mstar**0.56


def halo(m200, z, mstar, re):
    """CDM halo and stars: (M_CDM(r), rho_CDM(r), M_b(r)) as callables of r [kpc]."""
    c = float(concentration(m200, z))
    r200 = (3 * m200 / (4 * np.pi * 200 * rho_crit(z))) ** (1 / 3)
    rs, f, a = r200 / c, mstar / m200, re / 1.8153
    mu = lambda x: np.log1p(x) - x / (1 + x)
    rho_s = (1 - f) * m200 / (4 * np.pi * rs**3 * mu(c))
    m_cdm = lambda r: (1 - f) * m200 * mu(np.minimum(r, r200) / rs) / mu(c)
    rho_cdm = lambda r: rho_s / ((r / rs) * (1 + r / rs) ** 2)
    m_b = lambda r: mstar * r**2 / (r + a) ** 2
    return m_cdm, rho_cdm, m_b


def galaxy(log_m200, z, alpha):
    """M200, M_* = alpha x Moster+13 and the Shen+03 size of the Chabrier-IMF galaxy."""
    m200 = 10**log_m200
    ms0 = mstar_moster13(m200, z)
    return m200, alpha * ms0, re_shen03(ms0)


def isothermal(rho0, s0, m_b, r1, n=600, profile=False):
    """Integrate the isothermal DM from R_IN to r1 for an array of rho0 [Msun/kpc^3] at fixed
    sigma0 [km/s]. Returns rho(r1), M_DM(r1); with profile=True, (r, rho, M_DM) on the grid."""
    lr = np.linspace(np.log(R_IN), np.log(r1), n)
    dl = lr[1] - lr[0]
    rho0 = np.atleast_1d(np.asarray(rho0, float))
    h, m = np.zeros_like(rho0), 4 / 3 * np.pi * R_IN**3 * rho0

    def f(l, h, m):
        r = np.exp(l)
        return G * (m + m_b(r)) / (s0**2 * r), 4 * np.pi * r**3 * rho0 * np.exp(-h)

    out = []
    for i in range(n):
        if profile:
            out.append((np.exp(lr[i]), rho0 * np.exp(-h), m))
        if i == n - 1:
            break
        k1 = f(lr[i], h, m)
        k2 = f(lr[i] + dl / 2, h + dl / 2 * k1[0], m + dl / 2 * k1[1])
        k3 = f(lr[i] + dl / 2, h + dl / 2 * k2[0], m + dl / 2 * k2[1])
        k4 = f(lr[i] + dl, h + dl * k3[0], m + dl * k3[1])
        h = h + dl / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        m = m + dl / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    if profile:
        return (np.array([o[0] for o in out]), np.array([o[1][0] for o in out]),
                np.array([o[2][0] for o in out]))
    return rho0 * np.exp(-h), m


def x_of(s0, rho_cdm_r1):
    """X = (sigma/m) t [cm^2 g^-1 Gyr] for core dispersion s0 [km/s] and rho_CDM(r1) [Msun/kpc^3]."""
    return np.sqrt(np.pi) / (4 * s0 * rho_cdm_r1 * CM2G * KMS_GYR)


def branch(m200, z, mstar, re, r1s, tol=1e-6):
    """Follow the matched solutions along r1s (increasing). Returns rows (r1, rho0, sigma0, X) up to
    the first r1 where the solver fails (residual > tol); see probe scripts for what that means."""
    m_cdm, rho_cdm, m_b = halo(m200, z, mstar, re)

    def resid(p, r1):
        rho, mm = isothermal(np.exp(p[0]), np.exp(p[1]), m_b, r1)
        return [np.log(rho[0] / rho_cdm(r1)), np.log(mm[0] / m_cdm(r1))]

    r1 = r1s[0]
    p = np.log([3 * rho_cdm(r1), np.sqrt(G * (m_cdm(r1) + m_b(r1)) / (2 * r1))])
    out = []
    for r1 in r1s:
        sol = root(resid, p, args=(r1,), method="hybr")
        if not (sol.success and np.max(np.abs(resid(sol.x, r1))) < tol):
            break
        p = sol.x
        rho0, s0 = np.exp(p)
        out.append((r1, rho0, s0, x_of(s0, rho_cdm(r1))))
    return np.array(out)
