"""Fractional contribution of galaxy types to the MaNGA lens optical depth at fixed image
separation, from the per-galaxy histograms written by manga_lenses.py.

Types: Sersic n >= 2.5 (spheroid-dominated) or < 2.5 (disc-dominated), crossed with the
measured sigma_e (> 160, 70-160 km/s); sigma_e < 70 km/s as one class.

plots/manga_type_fractions.png
"""
import matplotlib.pyplot as plt
import numpy as np

import paths
from lenscore.population import SEP_BINS

RAW = paths.RAW / "manga"
d = np.load(RAW / "manga_profiles.npz", allow_pickle=True)
good = np.isfinite(d["sigma"])
sig, sn = d["sigma"][good], d["sersic_n"][good]
H = np.load(RAW / "manga_sep_hist.npy")
assert H.shape[0] == good.sum()

# rebin the 60 separation bins (0.0555 dex) into ~0.15-dex bins
edges = SEP_BINS
c = np.sqrt(edges[1:] * edges[:-1])
new = np.logspace(-1, np.log10(8), 14)
idx = np.clip(np.digitize(c, new) - 1, -1, new.size - 2)
keep = (c >= new[0]) & (c < new[-1])
Hc = np.zeros((H.shape[0], new.size - 1))
for k in range(new.size - 1):
    Hc[:, k] = H[:, keep & (idx == k)].sum(axis=1)
cn = np.sqrt(new[1:] * new[:-1])

KMS = r"\,\mathrm{km\,s^{-1}}"
classes = [
    (rf"spheroid-dominated, $\sigma_e>160{KMS}$", (sn >= 2.5) & (sig > 160), "#b2182b"),
    (rf"spheroid-dominated, $\sigma_e=70$–$160{KMS}$", (sn >= 2.5) & (sig >= 70) & (sig <= 160), "#f4a582"),
    (rf"disc-dominated, $\sigma_e>160{KMS}$", (sn < 2.5) & (sig > 160), "#2166ac"),
    (rf"disc-dominated, $\sigma_e=70$–$160{KMS}$", (sn < 2.5) & (sig >= 70) & (sig <= 160), "#92c5de"),
    (rf"all types, $\sigma_e<70{KMS}$", sig < 70, "#bababa"),
]
tot = Hc.sum(axis=0)
fr = np.array([Hc[s].sum(axis=0) / tot for _, s, _ in classes])
assert np.allclose(fr.sum(axis=0), 1)

print("separation [arcsec]: " + " ".join(f"{x:5.2f}" for x in cn))
for (lab, s, _), f in zip(classes, fr):
    print(f"{lab:40s} N={s.sum():5d}: " + " ".join(f"{x:5.2f}" for x in f))

plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(8, 5.6))
base = np.zeros(new.size - 1)
for (lab, _, col), f in zip(classes, fr):
    ax.stairs(base + f, new, baseline=base, fill=True, color=col, label=lab)
    base = base + f
ax.set_xscale("log")
ax.set_xlim(new[0], new[-1])
ax.set_ylim(0, 1)
ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5], labels=["0.1", "0.2", "0.5", "1", "2", "5"])
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("image separation [arcsec]")
ax.set_ylabel("fraction of lensing optical depth")
h, l = ax.get_legend_handles_labels()
order = [0, 1, 4, 2, 3]            # left column: spheroids (+ low sigma); right column: discs
ax.legend([h[k] for k in order], [l[k] for k in order], frameon=False, fontsize=9.5, ncol=2,
          loc="lower center", bbox_to_anchor=(0.5, 1.01), columnspacing=1.2, handlelength=1.4)
ax.text(0.99, 0.02, r"MaNGA DynPop, $z_s=2$; spheroid/disc split at Sérsic $n=2.5$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="white")
fig.tight_layout()
fig.savefig(paths.PLOTS / "manga_type_fractions.png", dpi=150)
