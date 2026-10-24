"""Three-panel aside combining plots/separations_compare.png and plots/delays.png:
(a) separation distribution with known lenses, (b) separation-delay plane,
(c) delays of sub-arcsecond lenses. Same calculations as compare_separations.py and
delays.py.

plots/separations_delays_combined.png
"""
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import compare_separations as CS
import delays as D
import paths
import sparc_lenses as S
from lenscore import cosmology as cosmo
from lenscore import profiles as prof
from lenscore.population import DT_E, SEP_BINS, SEP_E, optical_depth

# ---------- separation distribution (as compare_separations.py) ----------
massive = CS.sis_population(CS.CHOI, srange=(CS.SIG_SLACS, np.inf))
inter = CS.sis_population(CS.CHOI, srange=(CS.SIG_COUNTS, CS.SIG_SLACS))
lo, hi = CS.sis_population(CS.CHOI, 0.95, srange=(CS.SIG_SLACS, np.inf)), \
    CS.sis_population(CS.CHOI, 1.05, srange=(CS.SIG_SLACS, np.inf))
rows, vel = S.sparc_profiles("solid body")
disc = optical_depth(rows, S.weights(vel))
rows_f, _ = S.sparc_profiles("flat")
disc_flat = optical_depth(rows_f, S.weights(vel))
sig_i = np.logspace(np.log10(CS.SIG_COUNTS), np.log10(CS.SIG_SLACS), 60)
w_i = S.schechter_dlnv(sig_i, *CS.CHOI) * np.gradient(np.log(sig_i))
inter_shallow = optical_depth(prof.power_law(sig_i, 1.6, 2.0), w_i)
known = np.array([float(r["max_sep_arcsec"]) for r in
                  csv.DictReader(open(paths.KNOWN)) if r["max_sep_arcsec"]])

# ---------- delays (as delays.py) ----------
sig_m = np.logspace(np.log10(CS.SIG_SLACS), np.log10(500), 60)
sig_d = np.logspace(np.log10(CS.SIG_COUNTS), np.log10(CS.SIG_SLACS), 50)
wgt = lambda s: S.schechter_dlnv(s, *CS.CHOI) * np.gradient(np.log(s))
pops = {"m": D.population(prof.sis(sig_m), wgt(sig_m)),
        "i": D.population(prof.sis(sig_d), wgt(sig_d)),
        "i16": D.population(prof.power_law(sig_d, 1.6, 2.0), wgt(sig_d)),
        "d": D.population(rows, S.weights(vel))}
kd = [(float(r["max_sep_arcsec"]), float(r["max_abs_delay_d"]), r["type"])
      for r in csv.DictReader(open(paths.KNOWN))
      if r["max_sep_arcsec"] and r["max_abs_delay_d"]]
ks, kdt, kt = map(np.array, zip(*kd))

fig, (a, b, cc) = plt.subplots(1, 3, figsize=(18, 5.4))

# (a)
c = np.sqrt(SEP_BINS[1:] * SEP_BINS[:-1])
dlog = np.diff(np.log10(SEP_BINS))
a.fill_between(c, lo / dlog, hi / dlog, step="mid", color="C3", alpha=0.18, lw=0)
a.step(c, massive / dlog, where="mid", color="C3", lw=2.2)
a.fill_between(c, disc / dlog, disc_flat / dlog, step="mid", color="C2", alpha=0.18, lw=0)
a.step(c, disc / dlog, where="mid", color="C2", lw=2.2)
a.step(c, disc_flat / dlog, where="mid", color="C2", lw=1.2, ls="--")
a.fill_between(c, inter_shallow / dlog, inter / dlog, step="mid", facecolor="none", edgecolor="C1",
               hatch="////", lw=0, alpha=0.5)
a.step(c, inter / dlog, where="mid", color="C1", lw=1.6, ls="--")
a.set_xscale("log")
a.set_yscale("log")
a.set_xlim(0.08, 10)
a.set_ylim(1e-5, 1.5e-3)
a.set_xlabel("image separation [arcsec]")
a.set_ylabel(r"$d\tau/d\log_{10}\Delta\theta$  (source at $z_s=2$)")
a2 = a.twinx()
kb = np.arange(-1.3, 1.01, 0.1)
kn, _ = np.histogram(np.log10(known), kb)
sel = (c >= 2) & (c < 6)
scale = np.sum((known >= 2) & (known < 6)) / massive[sel].sum()
a2.stairs(kn / 0.1, 10**kb, color="k", lw=1.8)
a2.set_yscale("log")
a2.set_ylim(np.array(a.get_ylim()) * scale)
a2.set_ylabel(r"known lenses per dex (matched to solid red at 2-6$''$)")
a.legend(handles=[Line2D([], [], color="C3", lw=2.2, label=r"E/S0, $\sigma>160$ km/s"),
                  Line2D([], [], color="C1", lw=1.6, ls="--", label=r"E/S0, $\sigma=70$-$160$ km/s"),
                  Line2D([], [], color="C2", lw=2.2, label="HI-rich discs (SPARC)"),
                  Line2D([], [], color="k", lw=1.8, label="known lenses (right axis)")],
         frameon=False, fontsize=7.5, loc="upper left", handlelength=1.8, labelspacing=0.3)
a.set_title("(a) image separations", fontsize=10)

# (b)
sc = np.sqrt(SEP_E[1:] * SEP_E[:-1])
dc = np.sqrt(DT_E[1:] * DT_E[:-1])
for key, col, lab in [("m", "C3", r"E/S0, $\sigma>160$ km/s"), ("i", "C1", r"E/S0, $\sigma=70$-$160$ km/s"),
                      ("d", "C2", "HI-rich discs (SPARC)")]:
    h = pops[key]
    z = np.log10(np.where(h > 0, h, np.nan) / h.max())
    b.contour(sc, dc, z.T, levels=[-2, -1, -0.3], colors=col, linewidths=[0.8, 1.3, 2.0],
              negative_linestyles="solid")
    b.plot([], [], color=col, lw=1.6, label=lab)
dd = np.logspace(-1.3, 1, 50)
for qq, ls in [(0.1, "--"), (0.5, ":")]:
    b.plot(dd, 3e3 * cosmo.MPC_M / (2 * cosmo.C_KMS * 1e3) * (dd * cosmo.ARCSEC) ** 2 * (1 - qq) / (1 + qq) / cosmo.DAY,
           color="0.4", lw=1, ls=ls, label=rf"SIS, $D_{{\Delta t}}=3$ Gpc, $q={qq}$")
dbl = kt == "Double"
b.scatter(ks[dbl], kdt[dbl], s=16, color="k", zorder=5, label="known doubles")
b.scatter(ks[~dbl], kdt[~dbl], s=16, facecolors="none", edgecolors="k", zorder=5,
          label="known quads/triples")
b.set_xscale("log")
b.set_yscale("log")
b.set_xlim(0.08, 8)
b.set_ylim(0.1, 2000)
b.set_xlabel("image separation [arcsec]")
b.set_ylabel("time delay [days]")
b.legend(frameon=False, fontsize=7.5, loc="upper left")
b.set_title(r"(b) separation and delay (contours 1, 10, 50% of peak; $q\geq0.1$)", fontsize=10)

# (c)
selc = (sc >= 0.1) & (sc < 1.0)
dl_ = np.diff(np.log10(DT_E))
for key, sty, lab in [("m", dict(color="C3", lw=2), r"E/S0, $\sigma>160$ km/s"),
                      ("i", dict(color="C1", lw=2), r"E/S0, $\sigma=70$-$160$ km/s (SIS)"),
                      ("i16", dict(color="C1", lw=1.4, ls="--"), r"E/S0, $\sigma=70$-$160$ km/s ($\gamma=1.6$)"),
                      ("d", dict(color="C2", lw=2), "HI-rich discs (SPARC)")]:
    cc.step(dc, pops[key][selc].sum(axis=0) / dl_, where="mid", label=lab, **sty)
cc.set_xscale("log")
cc.set_yscale("log")
cc.set_xlim(0.1, 300)
cc.set_ylim(1e-7, 2e-4)
cc.set_xlabel("time delay [days]")
cc.set_ylabel(r"$d\tau/d\log_{10}\Delta t$, lenses with $0.1''<\Delta\theta<1''$")
cc.legend(frameon=False, fontsize=7.5, loc="upper left")
cc.set_title("(c) delays of sub-arcsecond lenses", fontsize=10)

fig.tight_layout()
fig.savefig(paths.PLOTS / "separations_delays_combined.png", dpi=150)
