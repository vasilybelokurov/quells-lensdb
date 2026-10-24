"""Parametric SIDM haloes with baryons through core expansion AND core collapse, after Yang (2024,
PRD 110, 103044, arXiv:2405.03787; 'Cored-DZ' model), implemented from the paper's equations
(the author's GPL code, github.com/DanengYang/parametricSIDM, was used only to check coefficients).

Halo: NFW (M200c, Diemer+19 concentration), R_cut = r200. Stars: Hernquist, r_H = R_e / 1.8153,
rho_H = M_* / (2 pi r_H^3).
Progenitor: the NFW halo contracted by the stars with the relation of the paper's Appendix C,
    r_j / r_f = 1 - f_b + M_b(rbar_f) / M_i(rbar_j),   rbar = 0.85 (r / R200)^0.8 R200,
    M_DM(r_f) = (1 - f_b) M_i(r_j),   f_b = M_* / M200,
fitted with a Dekel-Zhao profile (inner slope a, concentration c = R_cut / r_x) between 0.05 and 5 r_s.
Gravothermal phase: tau = t / t_c,b with (Eqs. 8-9, C = 0.9, alpha = 1.6, gamma = 20)
    t_c,b = 150/C / [(sigma/m)(rho_s r_s / r_eff + gamma rho_H r_H^3 / (r_eff (r_eff + r_H)^2))]
            x [4 pi G (rho_s r_s^2 + alpha rho_H r_H^2 / 2)]^(-1/2),
    r_eff = (rho_s r_s^3 + alpha rho_H r_H^3 / 2) / (rho_s r_s^2 + alpha rho_H r_H^2 / 2),
    F_t = t_c,b / t_c,0 (Eq. 10), t_c,0 the same without baryons.
Here t = cosmic age at z_l and the halo and stars are held fixed (the paper integrates dt / t_c,b over an
accretion history), so tau = (sigma/m) t / t_c,b(sigma/m = 1).
Profile (Eqs. 20-22): rho = f_in rho_x f_out / [((r^k + r_c^k)^(1/k) / r_x) (1 + (r/r_x)^(1/2))^(2(3.5-a))],
    rho_x = rho_x0 g_rho(tau), r_x = r_x0 g_r(tau), r_c = r_s g_c(tau) F_t^2, k = 4 F_t^(1/2),
    f_in = (r/r_x + r_c / (0.4 r_s) [rho_x0 r_x0 / (rho_s r_s + 0.4 rho_H r_H)]^(1/(a-1)))^(1-a),
    f_out = (1 + (r / R_cut)(r_x / r_x0 - 1))^(-1/2);
frozen at tau = 1.1 beyond core collapse (as the paper does for its beta4 model).

Caveat (compare_models.py --mass-conservation): these fitted profile forms do not conserve the total mass;
without stars the Cored-DZ halo gains 7-29% inside r200 between tau = 0.01 and 1 (the published DM-only beta4
model 1-8%). The excess sits at r >~ r_s and adds ~1e-3 to the projected mass inside a few kpc.
The progenitor fit is restricted to a >= A_MIN = 1.05: f_in contains [...]^(1/(a-1)), which blows up for
a <= 1 and made Einstein radii jump by x5 between neighbouring halo masses where the free fit crossed a = 1
(check_parametric.py; log M200 <~ 11.05 and >~ 13.45 at z = 0.6). Those haloes get the best fit with a = A_MIN.
"""
import numpy as np
from scipy.optimize import brentq, least_squares

from lenscore.cosmology import G, rho_crit
from lenscore.lensing import R3
from lenscore.profiles import concentration

C_TC, ALPHA_B, GAMMA_B = 0.9, 1.6, 20.0
A_MIN = 1.05
CM2G = 1.989e33 / 3.0857e21**2          # 1 cm^2/g in kpc^2 / Msun
KMS_GYR = 1.0227                        # 1 km/s x 1 Gyr in kpc
TAU_FREEZE = 1.1
LN0 = np.log(0.001)


def g_rho(t):
    return 2.033 + 0.7381 * t + 7.264 * t**5 - 12.73 * t**7 + 9.915 * t**9 + (1 - 2.033) / LN0 * np.log(t + 0.001)


def g_r(t):
    return 0.7178 - 0.1026 * t + 0.2474 * t**2 - 0.4079 * t**3 + (1 - 0.7178) / LN0 * np.log(t + 0.001)


def g_c(t):
    return 2.555 * np.sqrt(t) - 3.632 * t + 2.131 * t**2 - 1.415 * t**3 + 0.4683 * t**4


def dz_mass(r, rho_x, r_x, a):
    """Dekel-Zhao enclosed mass (closed form): M = 4 pi rho_x r_x^3 / (3 - a) * u^(2(3-a)), u = s/(1+s), s = sqrt(r/r_x)
    (with Eq. 19 this gives exactly M_h inside R_cut)."""
    s = np.sqrt(r / r_x)
    return 4 * np.pi * rho_x * r_x**3 / (3 - a) * (s / (1 + s)) ** (2 * (3 - a))


def dz_norm(m_h, r_cut, a, c):
    """rho_x, r_x of Eq. 19 for halo mass m_h inside r_cut."""
    mu = c ** (a - 3) * (1 + c**0.5) ** (2 * (3 - a))
    return (1 - a / 3) * c**3 * mu * m_h / (4 * np.pi / 3 * r_cut**3), r_cut / c


class Halo:
    """One halo + Hernquist galaxy: progenitor fit and t_c,b for sigma/m = 1 cm^2/g. Optional: the halo
    concentration c (default Diemer+19) and a baryon profile m_b(r) used for the contraction and as the
    galaxy's mass (default the Hernquist profile; the Hernquist (M_*, R_e) still set t_c,b and f_in)."""

    def __init__(self, m200, z, mstar, re, c=None, m_b=None):
        self.c200 = float(concentration(m200, z)) if c is None else float(c)
        self.r200 = (3 * m200 / (4 * np.pi * 200 * rho_crit(z))) ** (1 / 3)
        self.rs = self.r200 / self.c200
        mu = np.log1p(self.c200) - self.c200 / (1 + self.c200)
        self.rho_s = m200 / (4 * np.pi * self.rs**3 * mu)
        self.m200, self.mstar, self.fb = m200, mstar, mstar / m200
        self.rH = re / 1.8153
        self.rhoH = mstar / (2 * np.pi * self.rH**3)
        self.m_b = m_b or (lambda r: mstar * r**2 / (r + self.rH) ** 2)
        self.m_i = lambda r: m200 * (np.log1p(r / self.rs) - (r / self.rs) / (1 + r / self.rs)) / mu
        self.a, self.c = self._fit_progenitor()
        self.rho_x0, self.r_x0 = dz_norm((1 - self.fb) * m200, self.r200, self.a, self.c)
        self.tc1, self.Ft = self._tc()

    def contracted_dm(self, r_i):
        """(r_f, M_DM(r_f)) for initial radii r_i (Appendix-C relation)."""
        rbar = lambda r: 0.85 * (r / self.r200) ** 0.8 * self.r200
        rf = np.array([brentq(lambda x: rj / x - (1 - self.fb) - self.m_b(rbar(x)) / self.m_i(rbar(rj)),
                              1e-3 * rj, 1.5 * rj) for rj in r_i])
        return rf, (1 - self.fb) * self.m_i(r_i)

    def _fit_progenitor(self):
        # contracted profile on a dense grid, interpolated onto fixed radii 0.05-5 r_s (a fixed set of fit points,
        # so the fit varies smoothly with halo mass)
        rf, md = self.contracted_dm(np.geomspace(0.005 * self.rs, 10 * self.rs, 200))
        rfit = self.rs * np.geomspace(0.05, 5, 40)
        mfit = np.exp(np.interp(np.log(rfit), np.log(rf), np.log(md)))
        mh = (1 - self.fb) * self.m200

        def res(p):
            a, lc = p
            rx, ry = dz_norm(mh, self.r200, a, np.exp(lc))
            return np.log(dz_mass(rfit, rx, ry, a) / mfit)

        sol = least_squares(res, [1.3, np.log(self.c200)], bounds=([A_MIN, 0.0], [2.8, 6.0]),
                            xtol=1e-12, ftol=1e-12, gtol=1e-12)
        self.fit_rms = np.sqrt(np.mean(sol.fun**2))
        return sol.x[0], np.exp(sol.x[1])

    def _tc(self):
        rs, rhos, rH, rhoH, a_, g_ = self.rs, self.rho_s, self.rH, self.rhoH, ALPHA_B, GAMMA_B
        reff = (rhos * rs**3 + a_ * rhoH * rH**3 / 2) / (rhos * rs**2 + a_ * rhoH * rH**2 / 2)
        sm = CM2G                                                          # sigma/m = 1 cm^2/g
        tcb = 150 / C_TC / (sm * (rhos * rs / reff + g_ * rhoH * rH**3 / (reff * (reff + rH) ** 2))) \
            / np.sqrt(4 * np.pi * G * (rhos * rs**2 + a_ * rhoH * rH**2 / 2))
        tc0 = 150 / C_TC / (sm * rhos) / np.sqrt(4 * np.pi * G * rhos * rs**2)
        return tcb / KMS_GYR, tcb / tc0                                    # Gyr, F_t

    def tau(self, sigma_m, t_gyr):
        return sigma_m * t_gyr / self.tc1

    def rho_dm(self, r, tau):
        t = min(tau, TAU_FREEZE)
        a, Ft = self.a, self.Ft
        rho_x, r_x = self.rho_x0 * g_rho(t), self.r_x0 * g_r(t)
        rc, k = self.rs * g_c(t) * Ft**2, 4 * Ft**0.5
        f_in = (r / r_x + rc / (0.4 * self.rs) * (self.rho_x0 * self.r_x0 / (self.rho_s * self.rs + 0.4 * self.rhoH * self.rH))
                ** (1 / (a - 1))) ** (1 - a)
        f_out = (1 + r / self.r200 * (r_x / self.r_x0 - 1)) ** -0.5
        return f_in * rho_x * f_out / ((r**k + rc**k) ** (1 / k) / r_x * (1 + np.sqrt(r / r_x)) ** (2 * (3.5 - a)))

    def m_dm(self, tau, r=R3):
        """Dark-matter enclosed mass on r (trapezoid in ln r from 1e-5 kpc; truncated at r200)."""
        rr = np.geomspace(1e-5, self.r200, 3000)
        y = 4 * np.pi * rr**3 * self.rho_dm(rr, tau)
        m = np.concatenate([[0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(np.log(rr)))])
        return np.interp(r, rr, m, right=m[-1])

    def m_tot(self, tau, r=R3):
        return self.m_dm(tau, r) + self.m_b(r)
