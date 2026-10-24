"""Halo model against the SLACS lenses, lens by lens.

For each lens: halo mass from the inverted Moster+13 relation at its z_l (median
relation), Diemer+19 concentration, Hernquist stars with the lens's own Chabrier
M_* and measured R_e. The model's projected mass and stellar fraction inside the
measured Einstein radius (kpc) are compared with Auger et al. 2009.

plots/halo_model_vs_slacs.png
"""
import matplotlib.pyplot as plt
import numpy as np
from colossus.halo import concentration, mass_so

import lens_population as L
import obs_lenses as O
import paths
from lenscore import cosmology as cosmo
from lenscore import lensing
from lenscore import profiles as prof

COLS = {"NFW": "k", "contracted": "C3", "cored n=0.5": "C0", "cored n=1": "C9"}
LOGM_GRID = np.linspace(10.5, 15.0, 451)


def m200_from_mstar(logms, z):
    lms = np.log10(prof.mstar_moster13(10**LOGM_GRID, z))
    return 10 ** np.interp(logms, lms, LOGM_GRID)       # monotonic over this range


def predict(s):
    ok = np.isfinite(s["logms"]) & np.isfinite(s["rein_kpc"]) & np.isfinite(s["re_kpc"])
    out = {"ok": ok}
    idx = np.nonzero(ok)[0]
    for model in COLS:
        mein, fst = np.full(ok.size, np.nan), np.full(ok.size, np.nan)
        for i in idx:
            z, ms = s["zl"][i], 10 ** s["logms"][i]
            m = np.array([m200_from_mstar(s["logms"][i], z)])
            c = concentration.concentration(m * cosmo.H, "200c", z, model="diemer19")
            r200 = mass_so.M_to_R(m * cosmo.H, z, "200c") / cosmo.H
            re = np.array([s["re_kpc"][i]])
            m3 = L.enclosed_profiles(m, c, r200, np.array([ms]), re, model)
            mst = ms * prof.hernquist_m(lensing.R3, re[0] / 1.8153)[None, :]
            tot, st = lensing.project(m3)[0], lensing.project(mst)[0]
            mein[i] = np.interp(s["rein_kpc"][i], lensing.R2, tot)
            fst[i] = np.interp(s["rein_kpc"][i], lensing.R2, st) / mein[i]
        out[model] = (mein, fst)
    return out


def running_median(x, y, bins):
    c = 0.5 * (bins[1:] + bins[:-1])
    med = np.array([np.nanmedian(y[(x >= a) & (x < b)]) if np.sum((x >= a) & (x < b)) >= 4
                    else np.nan for a, b in zip(bins[:-1], bins[1:])])
    return c, med


if __name__ == "__main__":
    s = O.slacs()
    p = predict(s)
    ok = p["ok"]
    x = s["logms"]
    bins = np.arange(10.6, 12.01, 0.2)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for model, col in COLS.items():
        mein, fst = p[model]
        r = np.log10(mein) - s["log_mein"]
        if model in ("NFW", "contracted"):
            a1.scatter(x[ok], r[ok], s=12, color=col, alpha=0.5, lw=0)
        cx, med = running_median(x[ok], r[ok], bins)
        a1.plot(cx, med, color=col, lw=2, label=f"{model}: median {np.nanmedian(r[ok]):+.2f} dex")
        cx, med = running_median(x[ok], 1 - fst[ok], bins)
        a2.plot(cx, med, color=col, lw=2, label=model)
    a1.axhline(0, color="0.5", lw=0.8)
    a1.set_xlabel(r"$\log M_*$ (Chabrier) [M$_\odot$]")
    a1.set_ylabel(r"$\log M_{\rm Ein}^{\rm model} - \log M_{\rm Ein}^{\rm SLACS}$")
    a1.legend(frameon=False, fontsize=8)
    a1.set_title("projected mass inside the measured Einstein radius", fontsize=10)

    a2.errorbar(x[ok], 1 - s["fstar_chab"][ok], yerr=s["e_fstar_chab"][ok], fmt="o", ms=3.5,
                color="0.2", ecolor="0.7", elinewidth=0.8, label="SLACS, Chabrier")
    a2.scatter(x[ok], 1 - s["fstar_salp"][ok], s=14, facecolors="none", edgecolors="0.45",
               label="SLACS, Salpeter")
    a2.set_ylim(-0.4, 1)
    a2.axhline(0, color="0.8", lw=0.8)
    a2.set_xlabel(r"$\log M_*$ (Chabrier) [M$_\odot$]")
    a2.set_ylabel(r"DM fraction inside $R_{\rm Ein}$ (projected)")
    a2.legend(frameon=False, fontsize=8, ncol=2)
    a2.set_title("model lines: running medians", fontsize=10)
    fig.tight_layout()
    fig.savefig(paths.PLOTS / "halo_model_vs_slacs.png", dpi=150)

    print(f"{ok.sum()} SLACS lenses")
    print(f"  observed DM fraction inside R_Ein: Chabrier {np.nanmedian(1 - s['fstar_chab'][ok]):.2f}, "
          f"Salpeter {np.nanmedian(1 - s['fstar_salp'][ok]):.2f}")
    for model in COLS:
        mein, fst = p[model]
        r = np.log10(mein) - s["log_mein"]
        print(f"  {model:12s} log(M_Ein model/obs) median {np.nanmedian(r[ok]):+.3f}, "
              f"16-84% {np.nanpercentile(r[ok], 16):+.2f} {np.nanpercentile(r[ok], 84):+.2f}; "
              f"model DM fraction {np.nanmedian(1 - fst[ok]):.2f}")
