"""Figure and numbers for the SIDM forecast (sidm/forecast.py): plots/sidm_forecast.png

All models fit the SLACS Einstein radii (each with its own alpha), and all ratios are taken relative to one
baseline, the CDM-valley model nearest nu = 0, so the question is whether SIDM leaves the band of
predictions CDM already allows (grey: the CDM valley, nu = -0.5 ... 1).
(a) separation distribution (flux ratio >= 0.1); (b) optical depth at 0.1-0.5" (solid) and 0.5-1" (dashed)
against sigma/m, for the isothermal-Jeans model (expansion phase; drawn only where < FLAG_MAX of the optical
depth is in haloes beyond its fold, dotted: that share) and the parametric gravothermal model (expansion and
collapse); the CDM limits of the two models (jref, pref) at sigma/m = 0; (c) median delay of 0.1-1" lenses;
(d) sigma/m at the fold of the Jeans model against halo mass.
(The earlier fixed-alpha version of this figure and its numbers are in commit 939bd7b.)

Result (2026-10-24; 19 z_l x 28 M200, every model SLACS-matched; ratios to the CDM valley at nu ~ 0):
  CDM valley: tau(0.1-0.5") x0.65-1.43, tau(0.5-1") x0.69-1.52, median delay x0.86-1.18; the shape
  S = tau(0.1-0.5")/tau(0.5-1") stays at 0.94-1.03.
  Jeans (sigma/m <= 0.2, the range it covers for SLACS): tau x0.97-1.03, delay x0.94-0.98 -- inside.
  Parametric: CDM limit (contracted) tau 0.63 / 0.70, delay 0.97 (edge of the band); SIDM sigma/m = 0.1-5:
  tau(0.1-0.5") 0.55 -> 0.39, tau(0.5-1") 0.61-0.67, delay 0.91-1.02; S = 0.87 (0.1) -> 0.78 (1) -> 0.59 (5)
  against 0.92 for its CDM limit. Sub-0.5" lenses fall below the CDM band; delays stay inside.

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

FLAG_MAX = 0.1
ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
f = np.load(paths.ROOT / "data" / "sidm" / ("forecast_quick.npz" if args.quick else "forecast.npz"))
keys, kinds, alphas, pars = list(f["keys"]), list(f["kinds"]), f["alphas"], f["pars"]
H1, H2, FLAG = dict(zip(keys, f["h1"])), dict(zip(keys, f["h2"])), dict(zip(keys, f["flag_tau"]))
A, PAR = dict(zip(keys, alphas)), dict(zip(keys, pars))
c1, c2 = np.sqrt(SEP_BINS[1:] * SEP_BINS[:-1]), np.sqrt(SEP_E[1:] * SEP_E[:-1])
dc = np.sqrt(DT_E[1:] * DT_E[:-1])
BANDS = [(0.1, 0.5), (0.5, 1.0)]


def band(k, lo, hi):
    return H1[k][(c1 >= lo) & (c1 < hi)].sum()


def median_delay(k, lo=0.1, hi=1.0):
    p = H2[k][(c2 >= lo) & (c2 < hi)].sum(axis=0)
    return np.exp(np.interp(0.5, np.cumsum(p) / p.sum(), np.log(dc))) if p.sum() > 0 else np.nan


cdm = [k for k, kd in zip(keys, kinds) if kd == "cdm"]
base = cdm[int(np.argmin(np.abs([PAR[k] for k in cdm])))]
jeans = [k for k, kd in zip(keys, kinds) if kd == "jeans"]
param = [k for k, kd in zip(keys, kinds) if kd == "param"]
valid = {k: FLAG[k] < FLAG_MAX * (H1[k].sum() + FLAG[k]) for k in jeans}
rel = lambda k, lo, hi: band(k, lo, hi) / band(base, lo, hi)
reld = lambda k: median_delay(k) / median_delay(base)

print(f"baseline: {base} (alpha {A[base]:.3f}); all models matched to SLACS")
print(f"{'model':12s} {'alpha':>6s}  tau(0.1-0.5\")  tau(0.5-1\")  median delay 0.1-1\"  [ratios to baseline]  flagged")
for k in cdm + ["jref"] + jeans + ["pref"] + param:
    if k not in H1:
        continue
    fl = FLAG[k] / (H1[k].sum() + FLAG[k]) if k in FLAG else 0.0
    print(f"{k:12s} {A[k]:6.3f}  {band(k, 0.1, 0.5):.3e} ({rel(k, 0.1, 0.5):4.2f})  {band(k, 0.5, 1):.3e} "
          f"({rel(k, 0.5, 1):4.2f})  {median_delay(k):6.2f} d ({reld(k):4.2f})   {fl:.2f}")

fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
dlog = np.diff(np.log10(SEP_BINS))
ax[0].fill_between(c1, np.min([H1[k] for k in cdm], axis=0) / dlog, np.max([H1[k] for k in cdm], axis=0) / dlog,
                   step="mid", color="0.8", label="CDM on the SLACS valley")
ax[0].step(c1, H1[base] / dlog, where="mid", color="k", lw=1.6, label=f"CDM, {base}")
for k, col in zip(param, plt.cm.plasma(np.linspace(0, 0.85, len(param)))):
    if PAR[k] in (0.5, 2.0, 5.0):
        ax[0].step(c1, H1[k] / dlog, where="mid", color=col, lw=1.3, label=rf"parametric SIDM $\sigma/m$={PAR[k]:g}")
ax[0].set(xscale="log", yscale="log", xlim=(0.08, 5), ylim=(1e-5, 2e-3), xlabel="image separation [arcsec]",
          ylabel=r"$d\tau/d\log_{10}\Delta\theta$ (flux ratio $\geq0.1$)")
ax[0].legend(fontsize=7)

for (lo, hi), ls, shade in zip(BANDS, ["-", "--"], ["0.82", "0.92"]):
    r = [rel(k, lo, hi) for k in cdm]
    ax[1].axhspan(min(r), max(r), color=shade, zorder=0)
    for model, ref, col in [(jeans, "jref", "C0"), (param, "pref", "C3")]:
        sm = [PAR[k] for k in model]
        y = [rel(k, lo, hi) if (k not in valid or valid[k]) else np.nan for k in model]
        ax[1].plot([0.03] + sm, [rel(ref, lo, hi)] + y, color=col, ls=ls, marker="o", ms=3,
                   label=f"{'Jeans' if ref == 'jref' else 'parametric'}, {lo}-{hi}\"")
ax[1].plot([PAR[k] for k in jeans], [FLAG[k] / (H1[k].sum() + FLAG[k]) for k in jeans], "C0:", lw=1,
           label="Jeans: share beyond fold")
ax[1].axhline(1, color="k", lw=0.6)
ax[1].set(xscale="log", ylim=(0, 3), xlabel=r"$\sigma/m$ [cm$^2$ g$^{-1}$] (CDM limit plotted at 0.03)",
          ylabel=r"$\tau$ / CDM ($\nu\approx0$)", title="grey: CDM range allowed by SLACS")
ax[1].legend(fontsize=7)

r = [reld(k) for k in cdm]
ax[2].axhspan(min(r), max(r), color="0.82", zorder=0)
for model, ref, col, lab in [(jeans, "jref", "C0", "Jeans"), (param, "pref", "C3", "parametric")]:
    y = [reld(k) if (k not in valid or valid[k]) else np.nan for k in model]
    ax[2].plot([0.03] + [PAR[k] for k in model], [reld(ref)] + y, color=col, marker="o", ms=3, label=lab)
ax[2].axhline(1, color="k", lw=0.6)
ax[2].set(xscale="log", ylim=(0.5, 2.0), xlabel=r"$\sigma/m$ [cm$^2$ g$^{-1}$]",
          ylabel='median delay of 0.1-1" lenses / CDM')
ax[2].legend(fontsize=7)

zg, lg, smax, ja = f["zg"], f["lg"], f["sigma_m_fold"], f["jeans_alphas"]
for j, a in enumerate(ja):
    for iz, ls in zip([0, len(zg) // 2, len(zg) - 1], [":", "-", "--"]):
        ax[3].plot(lg, smax[j, iz], color=plt.cm.viridis(j / max(1, len(ja) - 1)), ls=ls,
                   label=rf"$\alpha$={a:.2f}, $z_l$={zg[iz]:.1f}" if j in (0, len(ja) - 1) else None)
ax[3].set(yscale="log", xlabel=r"log M$_{200}$ [M$_\odot$]", ylabel=r"$\sigma/m$ at the Jeans fold [cm$^2$ g$^{-1}$]",
          title="above: no expansion-phase solution")
ax[3].legend(fontsize=6, ncol=2)
fig.tight_layout()
out = paths.PLOTS / ("sidm_forecast_quick.png" if args.quick else "sidm_forecast.png")
fig.savefig(out, dpi=110)
print(f"wrote {out}")
