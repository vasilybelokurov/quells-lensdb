"""SIDM probe 4: is the end of the matched branch with stars (probe_continue_r1.py) physical or numerical?

For each halo, at r1 just before and after the last solved point, map both matching residuals
    d_rho = ln[rho_iso(r1) / rho_CDM(r1)],   d_M = ln[M_iso(r1) / M_CDM(r1)]
over a (rho0, sigma0) grid. A solution is a crossing of the two zero-contours. If the contours cross
before the end and no longer cross after it, the branch ends at a fold of the model (no expansion-phase
solution beyond); if they still cross, the solver lost an existing solution.
Also repeats the continuation near the end with 4x finer integration (n = 2400) and finer r1 steps.

Result (2026-10-24), log M200 = 11.8 and 12.3, alpha = 1.5, z = 0.6: the end point (r1 = 11.5 / 23.4 kpc,
X = 4.77 / 6.49) is unchanged with 4x finer integration and finer r1 steps. In the maps the contours cross
at the branch solution at 0.8 x r1_end, nearly coincide over rho0 ~ 1e10-1e12 at 0.98 x (the expansion
solution merging with a second, high-density solution) and do not cross anywhere at 1.15 x. The branch
ends at a saddle-node fold: beyond X_max the isothermal-Jeans model has no equilibrium solution.
    python sidm/probe_branch_end.py      -> plots/sidm_branch_end.png
"""
import sys
from pathlib import Path

import jeans as J
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

z = 0.6
HALOES = [(11.8, 1.5), (12.3, 1.5)]
RHO0 = np.geomspace(1e7, 1e12, 260)
S0 = np.linspace(60, 320, 140)


def residual_maps(m_cdm, rho_cdm, m_b, r1, n=600):
    d_rho, d_m = np.zeros((S0.size, RHO0.size)), np.zeros((S0.size, RHO0.size))
    for i, s0 in enumerate(S0):
        rho, mm = J.isothermal(RHO0, s0, m_b, r1, n=n)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_rho[i], d_m[i] = np.log(rho / rho_cdm(r1)), np.log(mm / m_cdm(r1))
    return d_rho, d_m


fig, axes = plt.subplots(len(HALOES), 3, figsize=(15, 4.6 * len(HALOES)))
for row, (lm, alpha) in zip(axes, HALOES):
    m200, ms, re = J.galaxy(lm, z, alpha)
    m_cdm, rho_cdm, m_b = J.halo(m200, z, ms, re)
    b = J.branch(m200, z, ms, re, np.geomspace(0.02, 150, 140))
    r_end = b[-1, 0]
    # finer continuation close to the end, coarse and fine integration
    for n in (600, 2400):
        J_iso = J.isothermal
        J.isothermal = lambda *a, n_=n, **k: J_iso(*a[:4], n=n_, **{kk: v for kk, v in k.items() if kk != "n"})
        bf = J.branch(m200, z, ms, re, np.concatenate([np.geomspace(0.02, 0.9 * r_end, 80),
                                                        np.geomspace(0.9 * r_end, 3 * r_end, 200)[1:]]))
        J.isothermal = J_iso
        print(f"log M200 {lm} alpha {alpha}: integration n={n}: last solved r1 = {bf[-1, 0]:.3f} kpc, "
              f"X = {bf[-1, 3]:.3f}, rho0 = {bf[-1, 1]:.3e}, sigma0 = {bf[-1, 2]:.1f}")
    for ax, r1 in zip(row, (0.8 * r_end, 0.98 * r_end, 1.15 * r_end)):
        d_rho, d_m = residual_maps(m_cdm, rho_cdm, m_b, r1)
        ax.contour(RHO0, S0, d_rho, levels=[0], colors="C0", linewidths=1.8)
        ax.contour(RHO0, S0, d_m, levels=[0], colors="C3", linewidths=1.8)
        near = b[np.argmin(np.abs(b[:, 0] - r1))]
        if abs(near[0] / r1 - 1) < 0.05:
            ax.plot(near[1], near[2], "k*", ms=12, label="solution on branch")
        ax.plot([], [], color="C0", label=r"$\rho_{\rm iso}(r_1)=\rho_{\rm CDM}(r_1)$")
        ax.plot([], [], color="C3", label=r"$M_{\rm iso}(r_1)=M_{\rm CDM}(r_1)$")
        ax.set(xscale="log", xlabel=r"$\rho_0$ [M$_\odot$ kpc$^{-3}$]", ylabel=r"$\sigma_0$ [km s$^{-1}$]",
               title=f"log M200={lm}, α={alpha}: $r_1$={r1:.1f} kpc ({r1 / r_end:.2f} × branch end)")
        ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(paths.PLOTS / "sidm_branch_end.png", dpi=110)
print(f"wrote {paths.PLOTS / 'sidm_branch_end.png'}")
