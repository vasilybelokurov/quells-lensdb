"""Disc lenses with SIDM: separation and delay distributions of SPARC galaxies (HIPASS-weighted, as in
compare_separations.py) whose unmeasured centres come from the rotation-curve fits of sparc_sidm.py, for CDM
and for each sigma/m, against the solid-body / flat-centre bracket used so far.

Also: which sigma/m the rotation curves themselves prefer (chi^2 of each fit relative to CDM), by V_flat.
Optical depths with flux ratio >= 0.1 (as the SIDM forecast) and without a flux cut (as the main figure).

Result (2026-10-24, 150 galaxies, all fits converged): disc optical depth (no flux cut) 4.4e-5 for the CDM
fits and 4.5-4.7e-5 for SIDM sigma/m = 0.5-10 (+3-7%), against 4.0e-5 (solid-body centre) and 1.3e-4 (flat
centre) so far; 0.1-0.5" and 0.5-1" bands and median delays (12.5-13.0 d) likewise unchanged. Both fits put
the centres near the solid-body end: the flat-centre bracket is excluded by the rotation curves themselves.
Rotation curves prefer SIDM for V_flat < 150 km/s (median chi2 gain 2-7 for sigma/m = 1-10), not above.

    python sidm/sparc_lensing.py                    -> plots/sidm_sparc.png, data/sidm/sparc_lensing.npz
    python sidm/sparc_lensing.py --model collapse   -> the same for the parametric (collapse) fits of
        sparc_collapse.py: plots/sidm_sparc_collapse.png, data/sidm/sparc_collapse_lensing.npz
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
import sparc_lenses as S  # noqa: E402
from lenscore.population import DT_E, SEP_BINS, SEP_E, optical_depth  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["jeans", "collapse"], default="jeans")
MODEL = ap.parse_args().model
SUFFIX = "" if MODEL == "jeans" else "_collapse"
f = np.load(paths.ROOT / "data" / "sidm" / ("sparc_sidm.npz" if MODEL == "jeans" else "sparc_collapse.npz"))
names, rows, chi2, vel, sig = f["names"], f["rows"], f["chi2"], f["vel"], f["sigma_m"]
ok = f["ok"] if "ok" in f else chi2 < 1e11
tau_fit = f["tau"] if "tau" in f else None
w = S.weights(vel)                                       # HIPASS density per galaxy (velocity bins as before)
brackets = {k: S.sparc_profiles(k) for k in ("solid body", "flat")}
# the bracket rows for the same galaxies (sparc_profiles keeps the same selection and order)
bnames = [ln.split()[0] for ln in open(paths.RAW / "dwarfs" / "SPARC_Lelli2016c.mrt")
          if len(ln.split()) >= 18 and ln.split()[17].isdigit() and ln.split()[1].isdigit()
          and int(ln.split()[17]) <= 2 and float(ln.split()[5]) >= 30 and int(ln.split()[1]) != 0]
idx = [bnames.index(n) for n in names]
c1 = np.sqrt(SEP_BINS[1:] * SEP_BINS[:-1])
c2, dc = np.sqrt(SEP_E[1:] * SEP_E[:-1]), np.sqrt(DT_E[1:] * DT_E[:-1])


def tau(rows_, wts, qmin=None):
    good = np.all(np.isfinite(rows_), axis=1)
    kw = dict(nbeta=24, qmin=0.1) if qmin else {}
    return optical_depth(rows_[good], wts[good], **kw), good


out = {}
for name, rw in [("solid body", brackets["solid body"][0][idx]), ("flat", brackets["flat"][0][idx])] + \
        [(f"sigma/m={s:g}", rows[:, j]) for j, s in enumerate(sig)]:
    h, good = tau(rw, w)
    hq, _ = tau(rw, w, qmin=True)
    h2 = optical_depth(rw[good], w[good], nbeta=24, qmin=0.1, bins=SEP_E, dt_bins=DT_E)
    out[name] = (h, hq, h2, good.sum())
band = lambda h, lo, hi: h[(c1 >= lo) & (c1 < hi)].sum()
print(f"{len(names)} galaxies; tau without / with flux cut, and in bands (no cut): 0.1-0.5\", 0.5-1\", 1-3\"; "
      f"median delay of 0.1-1\" lenses (q >= 0.1)")
for k, (h, hq, h2, n) in out.items():
    p = h2[(c2 >= 0.1) & (c2 < 1)].sum(axis=0)
    med = np.exp(np.interp(0.5, np.cumsum(p) / p.sum(), np.log(dc))) if p.sum() > 0 else np.nan
    print(f"  {k:14s} N={n:3d}  tau {h.sum():.2e} / {hq.sum():.2e}   {band(h, 0.1, 0.5):.2e} {band(h, 0.5, 1):.2e} "
          f"{band(h, 1, 3):.2e}   median delay {med:5.1f} d")
dchi = chi2 - chi2[:, :1]
best = np.array(sig)[np.argmin(np.where(ok, chi2, np.inf), axis=1)]
for lo, hi in [(0, 80), (80, 150), (150, 400)]:
    sel = (vel >= lo) & (vel < hi)
    print(f"  V_flat {lo}-{hi} km/s (N={sel.sum()}): median chi2(SIDM) - chi2(CDM) = "
          + " ".join(f"{s:g}: {np.median(dchi[sel, j]):+.1f}" for j, s in enumerate(sig) if s > 0)
          + f"; best sigma/m histogram {dict(zip(*np.unique(best[sel], return_counts=True)))}")
if tau_fit is not None:
    for lo, hi in [(0, 80), (80, 150), (150, 400)]:
        sel = (vel >= lo) & (vel < hi)
        print(f"  V_flat {lo}-{hi}: collapsed (tau > 1) in the refit: "
              + " ".join(f"{s:g}: {np.sum(tau_fit[sel, j] > 1)}/{sel.sum()}" for j, s in enumerate(sig) if s > 0))
np.savez(paths.ROOT / "data" / "sidm" / f"sparc{SUFFIX}_lensing.npz", keys=np.array(list(out)),
         h=np.array([v[0] for v in out.values()]), hq=np.array([v[1] for v in out.values()]),
         h2=np.array([v[2] for v in out.values()]))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
dlog = np.diff(np.log10(SEP_BINS))
ax[0].fill_between(c1, out["solid body"][0] / dlog, out["flat"][0] / dlog, step="mid", color="C2", alpha=0.2,
                   label="solid-body to flat centre (so far)")
cols = plt.cm.viridis(np.linspace(0, 0.9, len(sig)))
for j, (s, col) in enumerate(zip(sig, cols)):
    h = out[f"sigma/m={s:g}"][0]
    ax[0].step(c1, h / dlog, where="mid", color=col, lw=2 if s == 0 else 1.4,
               label=("CDM fit (NFW)" if MODEL == "jeans" else "CDM limit (contracted)") if s == 0 else rf"SIDM fit $\sigma/m$={s:g} cm$^2$/g")
ax[0].set(xscale="log", yscale="log", xlim=(0.08, 5), ylim=(1e-6, 1e-3), xlabel="image separation [arcsec]",
          ylabel=r"$d\tau/d\log_{10}\Delta\theta$ (no flux cut)", title=f"HI-rich discs (SPARC + HIPASS), {'Jeans' if MODEL == 'jeans' else 'parametric (collapse)'} fits")
ax[0].legend(fontsize=7)
dlt = np.diff(np.log10(DT_E))
for j, (s, col) in enumerate(zip(sig, cols)):
    p = out[f"sigma/m={s:g}"][2][(c2 >= 0.1) & (c2 < 1)].sum(axis=0)
    ax[1].step(dc, p / dlt, where="mid", color=col, lw=2 if s == 0 else 1.4)
ax[1].set(xscale="log", yscale="log", xlim=(0.1, 300), ylim=(1e-8, 1e-4), xlabel="time delay [days]",
          ylabel=r'$d\tau/d\log_{10}\Delta t$, 0.1-1" lenses ($q\geq0.1$)')
for j, (s, col) in enumerate(zip(sig, cols)):
    if s > 0:
        ax[2].scatter(vel[ok[:, j]], dchi[ok[:, j], j], s=8, color=col, label=rf"$\sigma/m$={s:g}")
        if tau_fit is not None:
            c = ok[:, j] & (tau_fit[:, j] > 1)
            ax[2].scatter(vel[c], dchi[c, j], s=40, facecolor="none", edgecolor=col, lw=0.8)
ax[2].axhline(0, color="k", lw=0.6)
ax[2].set(xscale="log", yscale="symlog", xlabel=r"$V_{\rm flat}$ [km/s]",
          ylabel=r"$\chi^2_{\rm SIDM}-\chi^2_{\rm CDM}$ (rotation curve)", title="below 0: SPARC prefers SIDM" + ("; circled: collapsed" if tau_fit is not None else ""))
ax[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(paths.PLOTS / f"sidm_sparc{SUFFIX}.png", dpi=110)
print(f"wrote {paths.PLOTS / f'sidm_sparc{SUFFIX}.png'}")
