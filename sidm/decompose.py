"""Why does SLACS-matched parametric SIDM give fewer sub-0.5" lenses? Separate the two changes between the
parametric CDM limit and SIDM at sigma/m = 5: the self-interactions (at fixed alpha) and the lower stellar mass
normalisation that SLACS then requires (alpha 1.510 -> 1.210, slacs_sidm.py --model param).

Same population and lensing as forecast.py (19 z_l x 28 M200, flux ratio >= 0.1). Prints the optical depth in
separation bands, the mean log separation, and the Einstein radius / cross-section of a log M200 = 12 halo.

    python sidm/decompose.py      -> plots/sidm_decompose.png
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import parametric as P

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
from lenscore import cosmology as cosmo  # noqa: E402
from lenscore import lensing  # noqa: E402
from lenscore import profiles as prof  # noqa: E402
from lenscore.population import SEP_BINS  # noqa: E402

CASES = {"CDM limit, alpha 1.51": (1.510, 0.0), "CDM limit, alpha 1.21": (1.210, 0.0),
         "SIDM 5, alpha 1.51": (1.510, 5.0), "SIDM 5, alpha 1.21 (SLACS-matched)": (1.210, 5.0)}
ZG, LG = np.arange(0.1, 1.91, 0.1), np.arange(11.0, 13.71, 0.1)


def task(args):
    zl, lm = args
    from colossus.lss import mass_function
    m200 = 10**lm
    n = mass_function.massFunction(m200 * cosmo.H, zl, mdef="200c", model="tinker08", q_out="dndlnM") \
        * cosmo.H**3 * 0.1 * np.log(10)
    w0 = n * cosmo.dvdz(zl) * 0.1
    scr, dl = cosmo.sigma_crit(zl)
    ms0 = prof.mstar_moster13(m200, zl)
    re = prof.re_shen03(ms0)
    rows = []
    for a, s in CASES.values():
        h = P.Halo(m200, zl, a * ms0, re)
        rows.append(h.m_tot(h.tau(s, cosmo.COSMO.age(zl))))
    br, sep, _, _ = lensing.images(np.array(rows), dl, scr, 24, 0.1)
    return [np.histogram(sep[i][np.isfinite(sep[i])], SEP_BINS,
                         weights=np.full(np.isfinite(sep[i]).sum(), w0 * np.pi * br[i] ** 2 / 24))[0]
            for i in range(len(rows))]


if __name__ == "__main__":
    with ProcessPoolExecutor(13) as ex:
        res = list(ex.map(task, [(z, lm) for z in ZG for lm in LG], chunksize=4))
    H = np.sum(res, axis=0)
    c = np.sqrt(SEP_BINS[1:] * SEP_BINS[:-1])
    band = lambda h, lo, hi: h[(c >= lo) & (c < hi)].sum()
    print(f"{'case':38s} tau(0.1-0.5\")  tau(0.5-1\")  tau(1-3\")  total     <log sep>")
    for (name, _), h in zip(CASES.items(), H):
        print(f"{name:38s} {band(h, 0.1, 0.5):.3e}    {band(h, 0.5, 1):.3e}    {band(h, 1, 3):.3e}  {h.sum():.3e}  "
              f"{np.sum(h * np.log10(c)) / h.sum():+.3f}")
    zl, m200 = 0.6, 1e12
    scr, dl = cosmo.sigma_crit(zl)
    ms0 = prof.mstar_moster13(m200, zl)
    print(f"log M200 = 12, z_l = {zl}: Einstein radius, cross-section radius beta_r, mean separation (q >= 0.1)")
    for name, (a, s) in CASES.items():
        h = P.Halo(m200, zl, a * ms0, prof.re_shen03(ms0))
        row = h.m_tot(h.tau(s, cosmo.COSMO.age(zl)))[None, :]
        te = lensing.einstein_radius(row, dl, scr)[0] / cosmo.ARCSEC
        br, sep, _, _ = lensing.images(row, dl, scr, 24, 0.1)
        print(f"  {name:38s} theta_E {te:.3f}\"  beta_r {br[0] / cosmo.ARCSEC:.3f}\"  mean sep {np.nanmean(sep):.3f}\"  "
              f"counted {np.isfinite(sep).mean():.2f}  tau = {h.tau(s, cosmo.COSMO.age(zl)):.2f}")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    dlog = np.diff(np.log10(SEP_BINS))
    for (name, _), h, sty in zip(CASES.items(), H, [dict(color="k", lw=2), dict(color="k", ls="--"),
                                                       dict(color="C3", lw=2), dict(color="C3", ls="--")]):
        ax.step(c, h / dlog, where="mid", label=name, **sty)
    ax.set(xscale="log", yscale="log", xlim=(0.08, 5), ylim=(1e-5, 2e-3), xlabel="image separation [arcsec]",
           ylabel=r"$d\tau/d\log_{10}\Delta\theta$ (flux ratio $\geq0.1$)",
           title="parametric model: self-interactions vs stellar-mass normalisation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.PLOTS / "sidm_decompose.png", dpi=110)
    print(f"wrote {paths.PLOTS / 'sidm_decompose.png'}")
