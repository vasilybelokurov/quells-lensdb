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
from lenscore.profiles import concentration, mstar_moster13, re_shen03

CM2G = 1.989e33 / 3.0857e21**2          # 1 cm^2/g in kpc^2 / Msun
KMS_GYR = 1.0227                        # 1 km/s x 1 Gyr in kpc
R_IN = 1e-4                             # kpc, inner start of the integration


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

    def solve(r1, p):
        sol = root(resid, p, args=(r1,), method="hybr")
        return sol.x if sol.success and np.max(np.abs(resid(sol.x, r1))) < tol else None

    r1 = r1s[0]
    p = np.log([3 * rho_cdm(r1), np.sqrt(G * (m_cdm(r1) + m_b(r1)) / (2 * r1))])
    out = []
    def done():
        # keep the expansion branch only: up to the maximum of X (the fold); past it the continuation can
        # step onto the high-density branch, where X decreases again
        b = np.array(out)
        return b[:np.argmax(b[:, 3]) + 1] if len(b) else b

    for target in r1s:
        # step from the last solution towards target, halving the step on failure (up to 2^-8 of the
        # nominal step) so that only a genuine end of the branch stops the continuation
        last = out[-1][0] if out else target
        step, sub = np.log(target / last), 0
        while True:
            r1 = last * np.exp(step) if out else target
            q = solve(r1, p)
            if q is None:
                if sub == 8 or not out:
                    return done()
                step, sub = step / 2, sub + 1
                continue
            p, last = q, r1
            rho0, s0 = np.exp(p)
            out.append((r1, rho0, s0, x_of(s0, rho_cdm(r1))))
            if r1 >= target * (1 - 1e-12):
                break
            step = np.log(target / r1)
    return done()


def r200(m200, z):
    """r_200c [kpc] as used in halo()."""
    return (3 * m200 / (4 * np.pi * 200 * rho_crit(z))) ** (1 / 3)


def solve_at_x(m200, z, mstar, re, X, b):
    """(r1, rho0, sigma0) of the matched solution at X [cm^2 g^-1 Gyr], starting from the branch b
    (output of branch for the same halo); None if X lies beyond the end (fold) of the branch."""
    if X <= 0:
        return None
    if X > b[-1, 3]:
        return None
    m_cdm, rho_cdm, m_b = halo(m200, z, mstar, re)
    rgrid = np.logspace(-4, 3.5, 4000)
    lrho = -np.log(rho_cdm(rgrid))

    def r1_of(s0):
        target = np.sqrt(np.pi) / (4 * s0 * X * CM2G * KMS_GYR)
        return np.exp(np.interp(-np.log(target), lrho, np.log(rgrid)))

    def resid(p):
        r1 = r1_of(np.exp(p[1]))
        rho, mm = isothermal(np.exp(p[0]), np.exp(p[1]), m_b, r1)
        return [np.log(rho[0] / rho_cdm(r1)), np.log(mm[0] / m_cdm(r1))]

    lx = np.log(b[:, 3])
    guess = [np.interp(np.log(X), lx, np.log(b[:, 1])), np.interp(np.log(X), lx, np.log(b[:, 2]))]
    sol = root(resid, guess, method="hybr")
    if not (sol.success and np.max(np.abs(resid(sol.x))) < 1e-6):
        return None
    rho0, s0 = np.exp(sol.x)
    return r1_of(s0), rho0, s0


def m3d_at_x(m200, z, mstar, re, X, b):
    """Total enclosed mass M(<r) (SIDM dark matter + stars) as a callable of r [kpc] at X; None beyond
    the fold. Inside r1 the isothermal solution, outside the CDM halo."""
    s = solve_at_x(m200, z, mstar, re, X, b)
    if s is None:
        return None
    r1, rho0, s0 = s
    m_cdm, _, m_b = halo(m200, z, mstar, re)
    r, _, m_in = isothermal(rho0, s0, m_b, r1, n=800, profile=True)
    lr, lm = np.log(r), np.log(m_in)
    return lambda x: (np.where(x < r1, np.exp(np.interp(np.log(np.maximum(x, R_IN)), lr, lm)), m_cdm(x))
                      + m_b(x))
