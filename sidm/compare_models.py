"""Isothermal-Jeans (jeans.py) against the parametric gravothermal model (parametric.py, Yang 2024) for the
same haloes: dark-matter density and total enclosed mass at several sigma/m, each relative to CDM.

The two models use different CDM baselines: jeans.py an NFW halo without response, parametric.py the NFW
halo contracted by the stars (its 'progenitor'). Both are shown.

    python sidm/compare_models.py                       -> plots/sidm_models_compare.png
    python sidm/compare_models.py --mass-conservation   (no-star mass check, see mass_conservation)
"""
import argparse
import sys
from pathlib import Path

import jeans as J
import matplotlib.pyplot as plt
import numpy as np
import parametric as P

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
from lenscore import cosmology as cosmo  # noqa: E402

Z = 0.6
SIGMAS = [0.2, 0.5, 1.0, 2.0, 5.0]
HALOES = [(11.8, 1.5), (12.3, 1.5), (13.0, 1.5)]
r = np.logspace(-1.5, 2, 200)
age = cosmo.COSMO.age(Z)
_, dl = cosmo.sigma_crit(Z)


def main():
    fig, ax = plt.subplots(2, len(HALOES), figsize=(5.6 * len(HALOES), 9))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(SIGMAS)))
    for j, (lm, alpha) in enumerate(HALOES):
        m200, ms, re = J.galaxy(lm, Z, alpha)
        h = P.Halo(m200, Z, ms, re)
        m_cdm, rho_cdm, m_b = J.halo(m200, Z, ms, re)
        b = J.branch(m200, Z, ms, re, np.geomspace(0.02, J.r200(m200, Z), 100))
        m_contr = h.m_tot(0.0, r)
        ax[0, j].loglog(r, rho_cdm(r), "k", lw=2, label="NFW (no response)")
        ax[0, j].loglog(r, h.rho_dm(r, 0.0), "k--", lw=2, label="contracted progenitor (DZ fit)")
        ax[1, j].semilogx(r, m_contr / (m_cdm(r) + m_b(r)), "k--", lw=2, label="contracted CDM / NFW CDM")
        print(f"log M200 {lm}, alpha {alpha}: t_c,b(1 cm2/g) = {h.tc1:.1f} Gyr, F_t = {h.Ft:.3f}, "
              f"Jeans fold at sigma/m = {b[-1, 3] / age:.2f}")
        for s, col in zip(SIGMAS, cols):
            tau = h.tau(s, age)
            ax[0, j].loglog(r, h.rho_dm(r, tau), color=col, lw=1.5, label=rf"parametric $\sigma/m$={s:g} ($\tau$={tau:.2f})")
            ax[1, j].semilogx(r, h.m_tot(tau, r) / m_contr, color=col, lw=1.5,
                              label=rf"parametric / contracted CDM, $\sigma/m$={s:g}")
            m3 = J.m3d_at_x(m200, Z, ms, re, s * age, b)
            line = f"   sigma/m {s:4g}: tau {tau:5.2f}; M_tot(<2 kpc)/CDM: parametric {np.interp(2, r, h.m_tot(tau, r) / m_contr):.2f}"
            if m3 is not None:
                ax[0, j].loglog(r, np.gradient(m3(r) - m_b(r), r) / (4 * np.pi * r**2), color=col, lw=1, ls=":")
                ax[1, j].semilogx(r, m3(r) / (m_cdm(r) + m_b(r)), color=col, lw=1, ls=":")
                line += f", Jeans {np.interp(2, r, m3(r) / (m_cdm(r) + m_b(r))):.2f}"
            else:
                line += ", Jeans: beyond fold"
            print(line)
        ax[0, j].plot([], [], "k:", lw=1, label="isothermal Jeans (dotted, where it exists)")
        ax[0, j].set(xlim=(r[0], r[-1]), ylim=(1e5, 1e11), xlabel="r [kpc]", ylabel=r"$\rho_{\rm DM}$ [M$_\odot$ kpc$^{-3}$]",
                     title=rf"log M200={lm}, $\alpha$={alpha}, z={Z}: $t_c(1\,{{\rm cm^2/g}})$={h.tc1:.0f} Gyr")
        for th in (0.3, 1.0):
            ax[1, j].axvline(th * cosmo.ARCSEC * dl, color="0.6", lw=0.8, ls="--")
        ax[1, j].axhline(1, color="k", lw=0.6)
        ax[1, j].set(xlim=(0.1, 100), ylim=(0.4, 2.5), xlabel="r [kpc]", ylabel=r"$M_{\rm tot}(<r)$ / CDM")
        ax[0, j].legend(fontsize=6)
        ax[1, j].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(paths.PLOTS / "sidm_models_compare.png", dpi=100)
    print(f"wrote {paths.PLOTS / 'sidm_models_compare.png'}")


def mass_conservation(lm=12.3, z=Z):
    """M_DM(<r) at tau relative to tau = 0, without stars (F_t = 1), for the Cored-DZ implementation and for the
    dark-matter-only beta4 model of Yang et al. (2023, ApJ 949, 67; Eq. 15 here with k = 4, rho_s g_rho, r_s g_r,
    r_c = r_s g_c) applied to the same NFW halo."""
    m200, _, re = J.galaxy(lm, z, 1.0)
    h = P.Halo(m200, z, 1.0, re)
    rr = np.geomspace(1e-4, h.r200, 4000)

    def m_beta4(t, r):
        rs, rhos, rc = h.rs * P.g_r(t), h.rho_s * P.g_rho(t), h.rs * P.g_c(t)
        y = 4 * np.pi * rr**3 * rhos / ((rr**4 + rc**4) ** 0.25 / rs * (1 + rr / rs) ** 2)
        m = np.concatenate([[0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(np.log(rr)))])
        return np.interp(r, rr, m)

    r = np.array([1.0, 10.0, 0.5 * h.r200, h.r200])
    print(f"no stars, log M200 {lm}: a = {h.a:.2f}, c = {h.c:.2f}; r = 1, 10 kpc, r200/2, r200")
    for t in (0.0, 0.01, 0.1, 0.3, 0.6, 1.0):
        print(f"  tau {t:4.2f}: Cored-DZ M/M(tau=0) {np.round(h.m_dm(t, r) / h.m_dm(0.0, r), 3)}   "
              f"beta4 M/M(tau=0) {np.round(m_beta4(t, r) / m_beta4(0.0, r), 3)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass-conservation", action="store_true")
    mass_conservation() if ap.parse_args().mass_conservation else main()
