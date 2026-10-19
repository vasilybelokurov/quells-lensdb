"""Decompose the lens separation distribution into its two factors at fixed
lens geometry (D_ls/D_s = 0.5):

    dN_lens/dln(dtheta) ∝ dn/dln(sigma)  x  pi theta_E^2,
    theta_E = 4 pi (sigma/c)^2 D_ls/D_s,  dtheta = 2 theta_E  (SIS),

so the cross-section scales as dtheta^2 and, for dn/dln(sigma) ∝ sigma^a at low
sigma, the lens distribution scales as dtheta^((a+4)/2).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

C_KMS = 299792.458
ARCSEC = np.pi / 180 / 3600
RATIO = 0.5                                   # D_ls / D_s
PIVOT = 1.0                                   # normalise every curve to 1 at 1"


def sep_of_sigma(s):
    return 8 * np.pi * (s / C_KMS) ** 2 * RATIO / ARCSEC


def sigma_of_sep(th):
    return C_KMS * np.sqrt(th * ARCSEC / (8 * np.pi * RATIO))


def dn_dlnsigma(s, sstar, a, b):
    x = s / sstar
    return x**a * np.exp(-x**b)


CASES = [
    ("Early types (Choi+07)", lambda s: dn_dlnsigma(s, 161.0, 2.32, 2.67)),
    ("All galaxies (Bernardi+10)", lambda s: dn_dlnsigma(s, 113.78, 0.94, 1.85)),
    (r"LCDM haloes, $dn/d\ln V\propto V^{-3}$ (Zwaan+10)",
     lambda s: dn_dlnsigma(s, 161.0, -3.0, 2.67)),
]

th = np.logspace(np.log10(0.03), np.log10(8), 400)
s = sigma_of_sep(th)
cross = th**2                                  # pi theta_E^2 ∝ dtheta^2 ∝ sigma^4

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
for ax, (title, vdf) in zip(axes, CASES):
    n = vdf(s)
    lens = n * cross
    for y, lab, kw in [(n, "galaxies: $dn/d\\ln\\sigma$", dict(color="C0", ls="--")),
                       (cross, r"cross-section: $\pi\theta_E^2\propto\sigma^4$",
                        dict(color="C1", ls=":")),
                       (lens, "lenses: product", dict(color="k", lw=2))]:
        ax.plot(th, y / np.interp(PIVOT, th, y), label=lab, **kw)
    for x, lab in [(0.18, "Gaia"), (0.7, "Rubin"), (2.0, "ZTF")]:
        ax.axvline(x, lw=0.6, color="0.7")
        ax.text(x, 3e3, f" {lab}", fontsize=7, color="0.5")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e4)
    ax.set_xlim(th[0], th[-1])
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"image separation $\Delta\theta$ [arcsec]")
    top = ax.secondary_xaxis("top", functions=(sigma_of_sep, sep_of_sigma))
    top.set_xticks([50, 100, 200, 300], labels=["50", "100", "200", "300"])
    top.xaxis.set_minor_locator(plt.NullLocator())
    top.set_xlabel(r"$\sigma$ [km s$^{-1}$]  ($D_{ls}/D_s=0.5$)", fontsize=8)
axes[0].set_ylabel(r"per $d\ln\Delta\theta$, normalised at $1''$")
axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout()
fig.savefig(PLOTS / "separation_contributions.png", dpi=150)

for title, vdf in CASES:
    a = np.gradient(np.log(vdf(s)), np.log(s))[0]
    print(f"{title}: low-sigma slope of dn/dlnsigma = {a:.2f}, "
          f"lens distribution ∝ dtheta^{(a + 4) / 2:.2f}; "
          f"peak at {th[np.argmax(vdf(s) * cross)]:.2f}\"")
print(f"sigma for 0.5, 1, 2 arcsec: {sigma_of_sep(np.array([0.5, 1, 2])).round(0)} km/s")
