"""Figures from data/lensed_quasars.csv: separation distribution, its growth with
discovery epoch, and published delay against separation."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "plots"
FIG.mkdir(exist_ok=True)

rows = list(csv.DictReader(open(ROOT / "data" / "lensed_quasars.csv")))


def col(key):
    return np.array([float(r[key]) if r[key] else np.nan for r in rows])


sep = col("max_sep_arcsec")
year = col("discovery_year")
delay = col("max_abs_delay_d")
ddt = col("d_dt_gpc")
quad = np.array([r["type"] != "Double" for r in rows])
ok = np.isfinite(sep)

# Resolution markers: ZTF median image quality, Rubin median seeing, Gaia resolution.
MARKS = [(0.18, "Gaia"), (0.7, "Rubin"), (2.0, "ZTF")]

# 1. Separation distribution, doubles and quads stacked.
fig, ax = plt.subplots(figsize=(5, 3.5))
bins = np.arange(0, 6.01, 0.25)
ax.hist([sep[ok & ~quad], sep[ok & quad]], bins=bins, stacked=True,
        label=["doubles", "quads/triples"], color=["0.35", "0.7"])
for x, lab in MARKS:
    ax.axvline(x, ls="--", lw=0.8, color="C3")
    ax.text(x, ax.get_ylim()[1] * 0.97, f" {lab}", color="C3", va="top", fontsize=8)
ax.set_xlabel(r"maximum image separation [arcsec]")
ax.set_ylabel("systems")
ax.set_title(f"Known lensed quasars (GraL X), N={ok.sum()} with separation", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig(FIG / "separation_hist.png", dpi=150)

# 2. Cumulative separation distribution by discovery epoch.
fig, ax = plt.subplots(figsize=(5, 3.5))
grid = np.linspace(0, 4, 400)
for lo, hi in [(1979, 2010), (2010, 2018), (2018, 2022), (2022, 2027)]:
    s = np.sort(sep[ok & (year >= lo) & (year < hi)])
    ax.plot(grid, np.searchsorted(s, grid) / s.size,
            label=f"{lo}-{hi - 1} (N={s.size})")
for x, lab in MARKS:
    ax.axvline(x, ls="--", lw=0.8, color="0.6")
ax.set_xlabel(r"maximum image separation [arcsec]")
ax.set_ylabel("cumulative fraction")
ax.legend(frameon=False, fontsize=8, title="discovery year", title_fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "separation_by_epoch.png", dpi=150)

# 3. Published delay against separation, with the SIS scale for d_dt = 3 Gpc
#    (Eq. sis-delay of the paper: 42 d (theta/1")^2 for q -> 0).
fig, ax = plt.subplots(figsize=(7, 3.8))
m = ok & np.isfinite(delay) & (sep < 10)
ax.scatter(sep[m & ~quad], delay[m & ~quad], s=14, c="k", label="doubles")
ax.scatter(sep[m & quad], delay[m & quad], s=14, facecolors="none", edgecolors="k",
           label="quads/triples")
th = np.linspace(0.1, 10, 200)
ax.plot(th, 42 * th**2, color="C0", lw=1, label=r"SIS, $q\to0$, $d_{\Delta t}=3$ Gpc")
ax.axhspan(20, 120, color="C1", alpha=0.15, lw=0, label="QUELLS ZTF range (20-120 d)")
ax.axvspan(0, 2.0, color="C2", alpha=0.08, lw=0, label=r"blended in ZTF ($<2''$)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.15, 10)
ax.set_ylim(0.3, 3000)
ax.set_xlabel(r"maximum image separation [arcsec]")
ax.set_ylabel(r"largest published $|\Delta t|$ [d]")
ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout()
fig.savefig(FIG / "delay_vs_separation.png", dpi=150)

# Numbers for the text.
print(f"systems: {len(rows)}, with separation: {ok.sum()}")
for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 2.5), (2.5, np.inf)]:
    print(f'  {lo}-{hi}": {(ok & (sep >= lo) & (sep < hi)).sum()}')
print(f"median separation: {np.nanmedian(sep):.2f}\"")
for lo, hi in [(1979, 2010), (2010, 2018), (2018, 2022), (2022, 2027)]:
    s = sep[ok & (year >= lo) & (year < hi)]
    print(f"  {lo}-{hi - 1}: N={s.size}, median {np.median(s):.2f}\", <1\": {(s < 1).sum()}")
md = m & (sep < 2)
print(f"with delays: {m.sum()}; of these <2\": {md.sum()}, "
      f"<2\" and 20-120 d: {(md & (delay >= 20) & (delay <= 120)).sum()}")
