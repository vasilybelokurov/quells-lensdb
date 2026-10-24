"""Figure for the SIDM probes: plots/sidm_probe.png

(a) dark-matter density of log M200 = 12.3, alpha = 1.5, z = 0.6: CDM (NFW, no response), stars,
    and SIDM at several X = (sigma/m) t; (b) total enclosed mass (DM + stars), SIDM / CDM, with
    the radii of 0.3" and 1" lenses (theta, not separation) at this z_l for z_s = 2;
(c) X(r1) along the matched branch (jeans.branch) for four haloes, with stars (solid) and
    without (dotted); a solid curve ends at the fold where the matched solution ceases to exist
    (probe_branch_end.py).
    python sidm/plot_probe.py
"""
import sys
from pathlib import Path

import jeans as J
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
from lenscore import cosmology as cosmo  # noqa: E402

z = 0.6
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

m200, ms, re = J.galaxy(12.3, z, 1.5)
m_cdm, rho_cdm, m_b = J.halo(m200, z, ms, re)
b = J.branch(m200, z, ms, re, np.geomspace(0.02, 60, 120))
r = np.logspace(-2, 2, 300)
a = re / 1.8153
ax[0].loglog(r, rho_cdm(r), "k", lw=2, label="CDM (NFW, no response)")
ax[0].loglog(r, ms * a / (2 * np.pi * r * (r + a) ** 3), color="0.6", lw=2, ls="--", label=r"stars (Hernquist, $\alpha$=1.5)")
_, dl = cosmo.sigma_crit(z, cosmo.ZS)
for X, col in zip([0.1, 0.5, 1.5, 3.0, 5.0], plt.cm.viridis(np.linspace(0, 0.9, 5))):
    if X > b[:, 3].max():
        continue
    r1, rho0, s0, xk = b[np.argmin(np.abs(np.log(b[:, 3] / X)))]
    R, RHO, MM = J.isothermal(rho0, s0, m_b, r1, n=800, profile=True)
    out = r > r1
    ax[0].loglog(np.concatenate([R, r[out]]), np.concatenate([RHO, rho_cdm(r[out])]), color=col, lw=1.4,
                 label=f"SIDM X={xk:.2g} cm$^2$g$^{{-1}}$Gyr ($r_1$={r1:.1f} kpc)")
    m_sidm = np.where(r < r1, np.interp(r, R, MM), m_cdm(r)) + m_b(r)
    ax[1].semilogx(r, m_sidm / (m_cdm(r) + m_b(r)), color=col, lw=1.4, label=f"X={xk:.2g}")
ax[0].set(xlim=(1e-2, 1e2), ylim=(1e4, 1e12), xlabel="r [kpc]", ylabel=r"$\rho$ [M$_\odot$ kpc$^{-3}$]",
          title=f"log M200=12.3, M*={ms:.1e}, z={z}: dark matter")
ax[0].legend(fontsize=7)
ax[1].axhline(1, color="k", lw=0.8)
for th, ls in [(0.3, ":"), (1.0, "--")]:
    ax[1].axvline(th * cosmo.ARCSEC * dl, color="0.5", ls=ls, lw=0.9)
    ax[1].text(th * cosmo.ARCSEC * dl, 0.62, f' {th:g}"', fontsize=8)
ax[1].set(xlim=(0.1, 100), ylim=(0.6, 1.5), xlabel="r [kpc]", ylabel=r"$M_{\rm tot}(<r)$ SIDM / CDM",
          title="total enclosed mass (DM + stars)")
ax[1].legend(fontsize=7)

r1s = np.geomspace(0.02, 150, 140)
for (lm, al), col in zip([(11.8, 1.5), (12.3, 1.0), (12.3, 1.5), (13.0, 1.5)], ["C0", "C1", "C2", "C3"]):
    m200, ms, re = J.galaxy(lm, z, al)
    bs = J.branch(m200, z, ms, re, r1s)
    ax[2].loglog(bs[:, 0], bs[:, 3], color=col, lw=1.6, label=rf"log M200={lm}, $\alpha$={al}: max X={bs[:, 3].max():.1f}")
    bn = J.branch(m200, z, 1.0, re, r1s)
    ax[2].loglog(bn[:, 0], bn[:, 3], color=col, lw=0.8, ls=":")
ax[2].plot([], [], "k:", lw=0.8, label="same haloes, no stars")
ax[2].set(xlabel=r"$r_1$ [kpc]", ylabel=r"$X=(\sigma/m)\,t$ [cm$^2$g$^{-1}$Gyr]",
          title="isothermal-Jeans solutions along the expansion branch")
ax[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(paths.PLOTS / "sidm_probe.png", dpi=110)
print(f"wrote {paths.PLOTS / 'sidm_probe.png'}")
