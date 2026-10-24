"""SIDM probe 2: solve the two matching conditions in (rho0, sigma0) at fixed X, continuing in X.

For log M200 = 12.3, z = 0.6, alpha = 1 and 1.5, and the same haloes without stars.
Result (2026-10-24): residuals ~1e-11 up to X ~ 5-9 (alpha = 1.5) / ~9 (alpha = 1); beyond, the
solver stops converging even with fine steps (61 X values). Without stars, solutions exist to
X = 100. Parametrising by r1 instead (probe_continue_r1.py) gives the same end points.
    python sidm/probe_continue_x.py
"""
import jeans as J
import numpy as np
from scipy.optimize import root

z = 0.6
Xs = np.geomspace(0.01, 100, 61)
for alpha, stars in [(1.0, True), (1.5, True), (1.5, False)]:
    m200, ms, re = J.galaxy(12.3, z, alpha)
    if not stars:
        ms = 1.0
    m_cdm, rho_cdm, m_b = J.halo(m200, z, ms, re)
    rgrid = np.logspace(-4, 3, 4000)

    def r1_of(s0, X):
        target = np.sqrt(np.pi) / (4 * s0 * X * J.CM2G * J.KMS_GYR)
        return np.exp(np.interp(-np.log(target), -np.log(rho_cdm(rgrid)), np.log(rgrid)))

    def resid(p, X):
        rho0, s0 = np.exp(p)
        r1 = r1_of(s0, X)
        rho, mm = J.isothermal(rho0, s0, m_b, r1)
        return [np.log(rho[0] / rho_cdm(r1)), np.log(mm[0] / m_cdm(r1))]

    s0 = 200.0
    for _ in range(5):
        r1 = r1_of(s0, Xs[0])
        s0 = np.sqrt(J.G * (m_cdm(r1) + m_b(r1)) / (2 * r1))
    p = np.log([3 * rho_cdm(r1), s0])
    print(f"alpha {alpha}, stars {stars}")
    for i, X in enumerate(Xs):
        sol = root(resid, p, args=(X,), method="hybr")
        res = np.max(np.abs(resid(sol.x, X)))
        ok = sol.success and res < 1e-6
        if ok:
            p = sol.x
        if i % 4 == 0 or not ok:
            rho0, s0 = np.exp(sol.x)
            print(f"   X {X:8.3f}  converged {ok:d}  rho0 {rho0:.3e}  sigma0 {s0:6.1f}  r1 {r1_of(s0, X):7.3f}  residual {res:.0e}")
        if not ok:
            break
