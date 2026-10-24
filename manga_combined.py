"""Three-panel MaNGA figure: (a) image-separation distribution (as manga_lenses.py),
(b) image separation vs time delay, (c) delay distribution of sub-arcsecond lenses.

Panel (a) has no flux cut. Panels (b) and (c) use the Fermat-potential delays of
delays.py with the flux-ratio cut q >= 0.1 (QUELLS calibration range). The 2D
histograms per group are cached in data/raw/manga/manga_delay_hist.npz.

Run manga_population.py and manga_lenses.py first.
plots/manga_separations_delays.png
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

import compare_separations as CS
import delays as D
import paths
import sparc_lenses as S
from lenscore import cosmology as cosmo
from lenscore.population import DT_E, SEP_BINS, SEP_E, optical_depth

RAW = paths.RAW / "manga"
CACHE = RAW / "manga_delay_hist.npz"

d = np.load(RAW / "manga_profiles.npz", allow_pickle=True)
good = np.isfinite(d["sigma"])
rows, n, sig, sn = d["mt"][good], d["n"][good], d["sigma"][good], d["sersic_n"][good]
GROUPS = {
    "s160": (sig > 160, "C3", r"MaNGA, $\sigma_e>160$ km/s"),
    "s70": ((sig >= 70) & (sig <= 160), "C1", r"MaNGA, $\sigma_e=70$-$160$ km/s"),
    "slow": (sig < 70, "C0", r"MaNGA, $\sigma_e<70$ km/s"),
    "disc": (sn < 2.5, "C2", r"MaNGA, Sersic $n<2.5$ (disc-dominated)"),
}

h2 = D.cached_population(CACHE, {k: (rows[sel], n[sel]) for k, (sel, _, _) in GROUPS.items()})

# ---------- panel (a) ingredients (as manga_lenses.py) ----------
H = np.load(RAW / "manga_sep_hist.npy")
tot = H.sum(axis=0)
c = np.sqrt(SEP_BINS[1:] * SEP_BINS[:-1])
dlog = np.diff(np.log10(SEP_BINS))
etg = CS.sis_population(CS.CHOI, srange=(CS.SIG_SLACS, np.inf)) + \
    CS.sis_population(CS.CHOI, srange=(CS.SIG_COUNTS, CS.SIG_SLACS))
srows, vel = S.sparc_profiles("solid body")
disc_sparc = optical_depth(srows, S.weights(vel))
known = np.array([float(r["max_sep_arcsec"]) for r in
                  csv.DictReader(open(paths.KNOWN)) if r["max_sep_arcsec"]])
kd = [(float(r["max_sep_arcsec"]), float(r["max_abs_delay_d"]), r["type"])
      for r in csv.DictReader(open(paths.KNOWN))
      if r["max_sep_arcsec"] and r["max_abs_delay_d"]]
ks, kdt, kt = map(np.array, zip(*kd))

fig, (a, b, cc) = plt.subplots(1, 3, figsize=(19, 5.6))

# (a)
a.step(c, tot / dlog, where="mid", color="0.55", lw=2.4, label="MaNGA DynPop, all galaxies")
for k in ("s160", "s70", "slow"):
    sel, col, lab = GROUPS[k]
    a.step(c, H[sel].sum(axis=0) / dlog, where="mid", color=col, lw=1.6, label=lab)
sel, col, lab = GROUPS["disc"]
a.step(c, H[sel].sum(axis=0) / dlog, where="mid", color=col, lw=1.6, ls="--", label=lab)
a.step(c, etg / dlog, where="mid", color="C3", lw=1, ls=":", label=r"Choi E/S0 as SIS (all $\sigma$)")
a.step(c, disc_sparc / dlog, where="mid", color="C2", lw=1, ls=":", label="SPARC discs (solid-body centre)")
a.set_xscale("log")
a.set_yscale("log")
a.set_xlim(0.08, 10)
a.set_ylim(1e-6, 6e-3)
a.set_xlabel("image separation [arcsec]")
a.set_ylabel(r"$d\tau/d\log_{10}\Delta\theta$  (source at $z_s=2$)")
a2 = a.twinx()
kb = np.arange(-1.3, 1.01, 0.1)
kn, _ = np.histogram(np.log10(known), kb)
selk = (c >= 2) & (c < 6)
scale = np.sum((known >= 2) & (known < 6)) / tot[selk].sum()
a2.stairs(kn / 0.1, 10**kb, color="k", lw=2.6)
a2.set_yscale("log")
a2.set_ylim(np.array(a.get_ylim()) * scale)
a2.set_ylabel(r"known lenses per dex (matched to MaNGA total at 2-6$''$)")
a.plot([], [], color="k", lw=2.6, label="known lenses (right axis)")
a.legend(frameon=False, fontsize=7.5, loc="upper left")
a.set_title("(a) image separations", fontsize=10)

# (b)
sc = np.sqrt(SEP_E[1:] * SEP_E[:-1])
dc = np.sqrt(DT_E[1:] * DT_E[:-1])
for k in ("s160", "s70", "disc"):
    _, col, lab = GROUPS[k]
    h = h2[k]
    z = np.log10(np.where(h > 0, h, np.nan) / h.max())
    b.contour(sc, dc, z.T, levels=[-2, -1, -0.3], colors=col, linewidths=[0.8, 1.3, 2.0],
              linestyles="--" if k == "disc" else "-", negative_linestyles="--" if k == "disc" else "solid")
    b.plot([], [], color=col, lw=1.6, ls="--" if k == "disc" else "-", label=lab)
dd = np.logspace(-1.3, 1, 50)
for qq, ls in [(0.1, "--"), (0.5, ":")]:
    b.plot(dd, 3e3 * cosmo.MPC_M / (2 * cosmo.C_KMS * 1e3) * (dd * cosmo.ARCSEC) ** 2 * (1 - qq) / (1 + qq) / cosmo.DAY,
           color="0.4", lw=1, ls=ls, label=rf"SIS, $D_{{\Delta t}}=3$ Gpc, $q={qq}$")
dbl = kt == "Double"
b.scatter(ks[dbl], kdt[dbl], s=16, color="k", zorder=5, label="known doubles")
b.scatter(ks[~dbl], kdt[~dbl], s=16, facecolors="none", edgecolors="k", zorder=5, label="known quads/triples")
b.set_xscale("log")
b.set_yscale("log")
b.set_xlim(0.08, 8)
b.set_ylim(0.1, 2000)
b.set_xlabel("image separation [arcsec]")
b.set_ylabel("time delay [days]")
b.legend(frameon=False, fontsize=7.5, loc="upper left")
b.set_title(r"(b) separation and delay (contours 1, 10, 50% of peak; $q\geq0.1$)", fontsize=10)

# (c)
sub = (sc >= 0.1) & (sc < 1.0)
dl_ = np.diff(np.log10(DT_E))
allh = h2["s160"] + h2["s70"] + h2["slow"]
cc.step(dc, allh[sub].sum(axis=0) / dl_, where="mid", color="0.55", lw=2.4, label="MaNGA, all galaxies")
for k in ("s160", "s70", "slow", "disc"):
    _, col, lab = GROUPS[k]
    cc.step(dc, h2[k][sub].sum(axis=0) / dl_, where="mid", color=col, lw=1.8,
            ls="--" if k == "disc" else "-", label=lab)
cc.set_xscale("log")
cc.set_yscale("log")
cc.set_xlim(0.1, 300)
cc.set_ylim(1e-7, 1.5e-3)
cc.set_xlabel("time delay [days]")
cc.set_ylabel(r"$d\tau/d\log_{10}\Delta t$, lenses with $0.1''<\Delta\theta<1''$ ($q\geq0.1$)")
cc.legend(frameon=False, fontsize=7.5, loc="upper left")
cc.set_title("(c) delays of sub-arcsecond lenses", fontsize=10)

fig.tight_layout()
fig.savefig(paths.PLOTS / "manga_separations_delays.png", dpi=150)

print("median delay [d] of sub-arcsecond lenses (0.1-1\"), and by range:")
for k, (sel, _, lab) in list(GROUPS.items()) + [("all", (None, None, "all"))]:
    h = allh if k == "all" else h2[k]
    for lo, hi in [(0.1, 0.5), (0.5, 1.0)]:
        s = (sc >= lo) & (sc < hi)
        p = h[s].sum(axis=0)
        if p.sum() == 0:
            continue
        med = np.interp(0.5, np.cumsum(p) / p.sum(), dc)
        print(f"  {k:5s} {lo}-{hi}\": median {med:6.1f} d, share > 10 d {p[dc > 10].sum() / p.sum():.2f}")
