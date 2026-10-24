"""Predicted image-separation and time-delay distributions, CDM and SIDM, with and without re-fitting SLACS:
plots/sidm_separations.png and plots/sidm_delays.png

Left: every model matched to the SLACS Einstein radii with its own stellar-mass normalisation alpha
(data/sidm/forecast.npz). Right: self-interactions switched on at the alpha of each model's own CDM limit,
without re-fitting SLACS (data/sidm/forecast_fixed.npz). Grey: CDM on the SLACS valley (the same in both).
Jeans SIDM is drawn only where < 10% of its optical depth is in haloes beyond the fold of the model.
Flux ratio >= 0.1, z_s = 2. Delays: lenses with 0.1-1" separations; top row dtau/dlog10(dt), bottom row
the same normalised to unit area (shape only), with the medians marked.

Result (2026-10-24):
  Separations. CDM valley: S = tau(0.1-0.5")/tau(0.5-1") = 0.96-1.03. Jeans SIDM (sigma/m <= 0.2): unchanged.
  Parametric SIDM, matched: S 0.89 -> 0.61 (sigma/m 0.1 -> 5), all bands lower (fewer lenses); not matched: S
  0.87 -> 0.54, total up (x1.8 at sigma/m = 5), 1-3" lenses x2.5 -- the distribution moves to larger separations.
  Delays of 0.1-1" lenses. Medians: CDM valley 7.8-10.7 d; SIDM matched 8.3-9.2 d; not matched 8.4-9.6 d up to
  sigma/m = 2 and 11.5 d at sigma/m = 5. Normalised shapes coincide with CDM within the band.

    python sidm/plot_separations.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
from lenscore.population import DT_E, SEP_BINS, SEP_E  # noqa: E402

FLAG_MAX = 0.1
c = np.sqrt(SEP_BINS[1:] * SEP_BINS[:-1])
dlog = np.diff(np.log10(SEP_BINS))
PARAM_S = {0.5: "#c51b7d", 2.0: "#e66101", 5.0: "#fdb863"}
JEANS_S = {0.1: "#4393c3", 0.2: "#2166ac", 0.5: "#053061"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, name, title in zip(axes, ["forecast.npz", "forecast_fixed.npz"],
                           ["matched to SLACS (alpha re-fitted for every model)",
                            "not matched: SIDM at the alpha of its CDM limit"]):
    f = np.load(paths.ROOT / "data" / "sidm" / name)
    H = dict(zip(f["keys"], f["h1"]))
    A = dict(zip(f["keys"], f["alphas"]))
    F = dict(zip(f["keys"], f["flag_tau"]))
    cdm = [k for k, kd in zip(f["keys"], f["kinds"]) if kd == "cdm"]
    ax.fill_between(c, np.min([H[k] for k in cdm], axis=0) / dlog, np.max([H[k] for k in cdm], axis=0) / dlog,
                    step="mid", color="0.85", label="CDM on the SLACS valley")
    ax.step(c, H["jref"] / dlog, where="mid", color="#2166ac", lw=2, ls="--",
            label=rf"Jeans CDM limit ($\alpha$={A['jref']:.2f})")
    ax.step(c, H["pref"] / dlog, where="mid", color="#b2182b", lw=2, ls="--",
            label=rf"parametric CDM limit, contracted ($\alpha$={A['pref']:.2f})")
    for s, col in JEANS_S.items():
        k = f"jeans_s{s:g}"
        if k in H and F[k] < FLAG_MAX * (H[k].sum() + F[k]):
            ax.step(c, H[k] / dlog, where="mid", color=col, lw=1.3, label=rf"Jeans $\sigma/m$={s:g} ($\alpha$={A[k]:.2f})")
    for s, col in PARAM_S.items():
        k = f"param_s{s:g}"
        ax.step(c, H[k] / dlog, where="mid", color=col, lw=1.6, label=rf"parametric $\sigma/m$={s:g} ($\alpha$={A[k]:.2f})")
    ax.set(xscale="log", yscale="log", xlim=(0.08, 5), ylim=(1e-5, 3e-3), xlabel="image separation [arcsec]",
           title=title)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower center")
axes[0].set_ylabel(r"$d\tau/d\log_{10}\Delta\theta$ (flux ratio $\geq0.1$, $z_s=2$)")
fig.tight_layout()
fig.savefig(paths.PLOTS / "sidm_separations.png", dpi=110)
print(f"wrote {paths.PLOTS / 'sidm_separations.png'}")


# ---------- time delays of 0.1-1" lenses ----------
c2, dc = np.sqrt(SEP_E[1:] * SEP_E[:-1]), np.sqrt(DT_E[1:] * DT_E[:-1])
dlt = np.diff(np.log10(DT_E))
sub = (c2 >= 0.1) & (c2 < 1.0)
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
for j, (name, title) in enumerate(zip(["forecast.npz", "forecast_fixed.npz"],
                                      ["matched to SLACS (alpha re-fitted for every model)",
                                       "not matched: SIDM at the alpha of its CDM limit"])):
    f = np.load(paths.ROOT / "data" / "sidm" / name)
    D = {k: h[sub].sum(axis=0) for k, h in zip(f["keys"], f["h2"])}
    A = dict(zip(f["keys"], f["alphas"]))
    F = dict(zip(f["keys"], f["flag_tau"]))
    H = dict(zip(f["keys"], f["h1"]))
    cdm = [k for k, kd in zip(f["keys"], f["kinds"]) if kd == "cdm"]
    curves = [("jref", "#2166ac", "--", 2, rf"Jeans CDM limit ($\alpha$={A['jref']:.2f})"),
              ("pref", "#b2182b", "--", 2, rf"parametric CDM limit ($\alpha$={A['pref']:.2f})")]
    curves += [(f"jeans_s{s:g}", col, "-", 1.3, rf"Jeans $\sigma/m$={s:g}") for s, col in JEANS_S.items()
               if f"jeans_s{s:g}" in D and F[f"jeans_s{s:g}"] < FLAG_MAX * (H[f"jeans_s{s:g}"].sum() + F[f"jeans_s{s:g}"])]
    curves += [(f"param_s{s:g}", col, "-", 1.6, rf"parametric $\sigma/m$={s:g}") for s, col in PARAM_S.items()]
    for i, norm in enumerate([False, True]):
        ax = axes[i, j]
        scale = (lambda k: D[k].sum()) if norm else (lambda k: 1.0)
        env = np.array([D[k] / scale(k) for k in cdm])
        ax.fill_between(dc, env.min(axis=0) / dlt, env.max(axis=0) / dlt, step="mid", color="0.85",
                        label="CDM on the SLACS valley")
        for k, col, ls, lw, lab in curves:
            ax.step(dc, D[k] / scale(k) / dlt, where="mid", color=col, ls=ls, lw=lw, label=lab)
            if norm:
                med = np.exp(np.interp(0.5, np.cumsum(D[k]) / D[k].sum(), np.log(dc)))
                ax.axvline(med, color=col, ls=ls, lw=0.8)
        ax.set(xscale="log", yscale="log", xlim=(0.1, 300),
               ylim=(1e-7, 3e-4) if not norm else (3e-3, 2),
               ylabel=r'$d\tau/d\log_{10}\Delta t$, 0.1-1" lenses' if not norm else "normalised to unit area")
        if i == 0:
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=7, loc="lower center")
        else:
            ax.set_xlabel("time delay [days]")
fig.tight_layout()
fig.savefig(paths.PLOTS / "sidm_delays.png", dpi=110)
print(f"wrote {paths.PLOTS / 'sidm_delays.png'}")

# ---------- numbers behind both figures ----------
for name in ["forecast.npz", "forecast_fixed.npz"]:
    f = np.load(paths.ROOT / "data" / "sidm" / name)
    print(f"{name}: tau(0.1-0.5\"), tau(0.5-1\"), tau(1-3\"), S = tau(0.1-0.5)/tau(0.5-1), median delay of 0.1-1\" "
          f"lenses, share of those delays > 10 d")
    for k, kd, a, h1, h2, fl in zip(f["keys"], f["kinds"], f["alphas"], f["h1"], f["h2"], f["flag_tau"]):
        if kd == "cdm" and k not in ("cdm_nu-0.50", "cdm_nu-0.05", "cdm_nu+1.00"):
            continue
        if fl > FLAG_MAX * (h1.sum() + fl):
            continue
        b = [h1[(c >= lo) & (c < hi)].sum() for lo, hi in [(0.1, 0.5), (0.5, 1), (1, 3)]]
        d = h2[sub].sum(axis=0)
        med = np.exp(np.interp(0.5, np.cumsum(d) / d.sum(), np.log(dc)))
        print(f"  {k:12s} alpha {a:.2f}: {b[0]:.2e} {b[1]:.2e} {b[2]:.2e}  S {b[0] / b[1]:.2f}  "
              f"median {med:5.2f} d  >10 d {d[dc > 10].sum() / d.sum():.2f}")
