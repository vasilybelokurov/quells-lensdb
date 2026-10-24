"""SIDM probe 1: structure of the matching problem at fixed sigma0.

For log M200 = 12.3, alpha = 1.5, z = 0.6 and X = 3, 10, 30 cm^2 g^-1 Gyr: for each sigma0 on a
grid, r1 follows from the rate condition; rho0 is scanned for density matches at r1 and the
mass residual M_iso(r1)/M_CDM(r1) - 1 is printed for each match.

Result (2026-10-24): density matches exist only above sigma0 ~ 170-190 km/s, on two rho0 branches
that meet near the lowest sigma0; the mass residual has opposite signs on the two branches
(-0.12 and +0.26 at X = 3), so the matched solution lies close to that fold and a fixed sigma0
grid misses it -> solve both conditions jointly (probe_continue_r1.py).
    python sidm/probe_fixed_sigma0.py
"""
import jeans as J
import numpy as np

z = 0.6
m200, ms, re = J.galaxy(12.3, z, 1.5)
m_cdm, rho_cdm, m_b = J.halo(m200, z, ms, re)
rgrid = np.logspace(-3, 3, 2000)
print(f"M200 {m200:.2e}  M* {ms:.2e}  R_e {re:.2f} kpc")
for X in (3, 10, 30):
    rows = []
    for s0 in np.logspace(np.log10(30), np.log10(500), 60):
        target = np.sqrt(np.pi) / (4 * s0 * X * J.CM2G * J.KMS_GYR)       # rho_CDM(r1)
        if not (rho_cdm(rgrid[-1]) < target < rho_cdm(rgrid[0])):
            continue
        r1 = np.exp(np.interp(-np.log(target), -np.log(rho_cdm(rgrid)), np.log(rgrid)))
        rho0 = np.logspace(np.log10(target) - 1, np.log10(target) + 6, 300)
        rho, mm = J.isothermal(rho0, s0, m_b, r1)
        d = np.log(rho / rho_cdm(r1))
        for k in np.nonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0]:
            t = d[k] / (d[k] - d[k + 1])
            rows.append((s0, r1, rho0[k] * (rho0[k + 1] / rho0[k]) ** t, (mm[k] + t * (mm[k + 1] - mm[k])) / m_cdm(r1) - 1))
    print(f"X = {X}: {len(rows)} density matches")
    for r in rows[::max(1, len(rows) // 12)]:
        print("   sigma0 %5.0f  r1 %6.2f  rho0 %.2e  mass residual %+.3f" % r)
