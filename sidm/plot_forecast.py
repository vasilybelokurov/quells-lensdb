"""Figure and numbers for the SIDM forecast (sidm/forecast.py): plots/sidm_forecast.png

(a) separation distribution (flux ratio >= 0.1): CDM range allowed by SLACS (valley, grey), the X -> 0 reference
    at the valley alpha(nu=0), and SIDM at that alpha for several sigma/m;
(b) optical depth at 0.1-0.5" and 0.5-1": SIDM / reference against sigma/m for each alpha, with the CDM valley
    spread (relative to the valley model nearest nu = 0) as the grey band; dotted: share of the reference optical
    depth in haloes beyond the fold of the model (not counted). Ratios are drawn only where that share is below
    FLAG_MAX: beyond, the model has no prediction for most lenses and a ratio would only measure the hole;
(c) median delay of 0.1-1" lenses, SIDM / reference, with the CDM valley spread;
(d) sigma/m at the fold against halo mass, for the lowest, middle and highest lens redshift.

Result (2026-10-24, full grid: 13 z_l x 19 M200, 24 sources per lens):
  CDM valley spread relative to nu ~ 0: tau(0.1-0.5") x0.65-1.36, tau(0.5-1") x0.65-1.62, median delay of
  0.1-1" lenses x0.88-1.27 (7.5-10.8 d).
  SIDM at fixed alpha, where < 10% of the optical depth is beyond the fold: tau(0.1-0.5") x1.00-1.19,
  tau(0.5-1") x1.02-1.31, median delay x0.98-1.06 -- all inside the CDM spread.
  The model covers only sigma/m <~ 0.2 (alpha = 2.16, the SLACS valley at nu = 0), <~ 0.2-0.5 (alpha = 1.5)
  and <~ 0.5 (alpha = 1); at sigma/m = 1 a fraction 0.88 / 0.65 / 0.18 of the reference optical depth is in
  haloes beyond the fold. Not yet done: re-matching alpha to SLACS for SIDM (fixed alpha here).
    python sidm/plot_forecast.py [--quick]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
from lenscore.population import DT_E, SEP_BINS, SEP_E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
f = np.load(paths.ROOT / "data" / "sidm" / ("forecast_quick.npz" if args.quick else "forecast.npz"))
keys = list(f["keys"])
H1 = dict(zip(keys, f["h1"]))
H2 = dict(zip(keys, f["h2"]))
FLAG = dict(zip(keys, f["flag_tau"]))
alphas_sidm, sigma_m = f["alphas_sidm"], f["sigma_m"]
c1 = np.sqrt(SEP_BINS[1:] * SEP_BINS[:-1])
c2 = np.sqrt(SEP_E[1:] * SEP_E[:-1])
dc = np.sqrt(DT_E[1:] * DT_E[:-1])


def band(h, lo, hi):
    return h[(c1 >= lo) & (c1 < hi)].sum()


def median_delay(h2, lo=0.1, hi=1.0):
    p = h2[(c2 >= lo) & (c2 < hi)].sum(axis=0)
    if p.sum() == 0:
        return np.nan
    return np.exp(np.interp(0.5, np.cumsum(p) / p.sum(), np.log(dc)))


FLAG_MAX = 0.1
valid = {k: FLAG[k] < FLAG_MAX * H1[f"ref_a{k.split('_')[1][1:]}"].sum() for k in keys if k.startswith("sidm_")}
cdm = [k for k in keys if k.startswith("cdm_")]
nus = np.array([float(k[6:]) for k in cdm])
k0 = cdm[int(np.argmin(np.abs(nus)))]                       # valley model nearest nu = 0
ratio_cdm = {(lo, hi): [band(H1[k], lo, hi) / band(H1[k0], lo, hi) for k in cdm] for lo, hi in [(0.1, 0.5), (0.5, 1)]}
delay_cdm = [median_delay(H2[k]) / median_delay(H2[k0]) for k in cdm]

print(f"CDM valley (SLACS): tau(0.1-0.5\"), tau(0.5-1\"), median delay of 0.1-1\" lenses; reference {k0}")
for k in cdm:
    print(f"  {k:14s} tau {band(H1[k], 0.1, 0.5):.3e} {band(H1[k], 0.5, 1):.3e}  delay {median_delay(H2[k]):6.2f} d")
print("SIDM at fixed alpha (outer halo without response), relative to its X -> 0 reference:")
for a in alphas_sidm:
    ref = f"ref_a{a:.2f}"
    print(f"  alpha {a:.2f}: reference tau {band(H1[ref], 0.1, 0.5):.3e} {band(H1[ref], 0.5, 1):.3e}, "
          f"delay {median_delay(H2[ref]):.2f} d")
    for s in sigma_m:
        k = f"sidm_a{a:.2f}_s{s:g}"
        print(f"    sigma/m {s:4g}: tau ratio {band(H1[k], 0.1, 0.5) / band(H1[ref], 0.1, 0.5):5.2f} "
              f"{band(H1[k], 0.5, 1) / band(H1[ref], 0.5, 1):5.2f}  delay ratio "
              f"{median_delay(H2[k]) / median_delay(H2[ref]):5.2f}  beyond fold {FLAG[k] / H1[ref].sum():.2f} of tau")

fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
dlog = np.diff(np.log10(SEP_BINS))
lo_env = np.min([H1[k] for k in cdm], axis=0)
hi_env = np.max([H1[k] for k in cdm], axis=0)
ax[0].fill_between(c1, lo_env / dlog, hi_env / dlog, step="mid", color="0.8", label="CDM on the SLACS valley")
a0 = alphas_sidm[0]
ax[0].step(c1, H1[f"ref_a{a0:.2f}"] / dlog, where="mid", color="k", lw=1.8, ls="--",
           label=rf"reference: $\alpha$={a0:.2f}, no response")
for s, col in zip(sigma_m, plt.cm.plasma(np.linspace(0, 0.85, len(sigma_m)))):
    k = f"sidm_a{a0:.2f}_s{s:g}"
    if valid[k]:
        ax[0].step(c1, H1[k] / dlog, where="mid", color=col, lw=1.4, label=rf"SIDM $\sigma/m$={s:g} cm$^2$/g")
ax[0].set(xscale="log", yscale="log", xlim=(0.08, 5), ylim=(1e-6, 2e-3), xlabel="image separation [arcsec]",
          ylabel=r"$d\tau/d\log_{10}\Delta\theta$ (flux ratio $\geq0.1$)")
ax[0].legend(fontsize=7)

cols = ["C3", "C1", "C0"]
for (lo, hi), ls in [((0.1, 0.5), "-"), ((0.5, 1), "--")]:
    ax[1].axhspan(min(ratio_cdm[(lo, hi)]), max(ratio_cdm[(lo, hi)]), color="0.85" if ls == "-" else "0.93", zorder=0)
for a, col in zip(alphas_sidm, cols):
    ref = f"ref_a{a:.2f}"
    for (lo, hi), ls in [((0.1, 0.5), "-"), ((0.5, 1), "--")]:
        r = [band(H1[f"sidm_a{a:.2f}_s{s:g}"], lo, hi) / band(H1[ref], lo, hi)
             if valid[f"sidm_a{a:.2f}_s{s:g}"] else np.nan for s in sigma_m]
        ax[1].plot(sigma_m, r, color=col, ls=ls, marker="o", ms=3,
                   label=rf'$\alpha$={a:.2f}, {lo}-{hi}"' if ls == "-" or a == alphas_sidm[0] else None)
    ax[1].plot(sigma_m, [FLAG[f"sidm_a{a:.2f}_s{s:g}"] / H1[ref].sum() for s in sigma_m], color=col, ls=":", lw=1)
ax[1].axhline(1, color="k", lw=0.6)
ax[1].set(xscale="log", ylim=(0, 2.5), xlabel=r"$\sigma/m$ [cm$^2$ g$^{-1}$]",
          ylabel=r"$\tau$ SIDM / reference", title="grey: CDM spread along the SLACS valley; dotted: beyond fold")
ax[1].legend(fontsize=7)

ax[2].axhspan(np.nanmin(delay_cdm), np.nanmax(delay_cdm), color="0.85", zorder=0)
for a, col in zip(alphas_sidm, cols):
    ref = f"ref_a{a:.2f}"
    ax[2].plot(sigma_m, [median_delay(H2[f"sidm_a{a:.2f}_s{s:g}"]) / median_delay(H2[ref])
                         if valid[f"sidm_a{a:.2f}_s{s:g}"] else np.nan for s in sigma_m],
               color=col, marker="o", ms=3, label=rf"$\alpha$={a:.2f}")
ax[2].axhline(1, color="k", lw=0.6)
ax[2].set(xscale="log", ylim=(0.5, 1.6), xlabel=r"$\sigma/m$ [cm$^2$ g$^{-1}$]",
          ylabel='median delay of 0.1-1" lenses, SIDM / reference')
ax[2].legend(fontsize=7)

zg, lg, smax = f["zg"], f["lg"], f["sigma_m_fold"]
for j, (a, col) in enumerate(zip(alphas_sidm, cols)):
    for iz, ls in zip([0, len(zg) // 2, len(zg) - 1], [":", "-", "--"]):
        ax[3].plot(lg, smax[j, iz], color=col, ls=ls, label=rf"$\alpha$={a:.2f}, $z_l$={zg[iz]:.2f}")
ax[3].set(yscale="log", xlabel=r"log M$_{200}$ [M$_\odot$]", ylabel=r"$\sigma/m$ at the fold [cm$^2$ g$^{-1}$]",
          title="above: no expansion-phase solution")
ax[3].legend(fontsize=6, ncol=2)
fig.tight_layout()
out = paths.PLOTS / ("sidm_forecast_quick.png" if args.quick else "sidm_forecast.png")
fig.savefig(out, dpi=110)
print(f"wrote {out}")
