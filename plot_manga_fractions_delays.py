"""Two-panel version of plot_manga_fractions.py: fractional contribution of galaxy types
to the MaNGA lens optical depth (a) at fixed image separation (no flux cut, as
manga_lenses.py) and (b) at fixed time delay for lenses with 0.1" < separation < 1"
(q >= 0.1, delays.py). Same classes and colours as plot_manga_fractions.py.

The 2D separation-delay histograms per class are cached in
data/raw/manga/manga_delay_hist_types.npz.

plots/manga_type_fractions_delays.png
"""
import matplotlib.pyplot as plt
import numpy as np

import delays as D
import lens_population as L

RAW = L.ROOT / "data" / "raw" / "manga"
CACHE = RAW / "manga_delay_hist_types.npz"
d = np.load(RAW / "manga_profiles.npz", allow_pickle=True)
good = np.isfinite(d["sigma"])
rows, n, sig, sn = d["mt"][good], d["n"][good], d["sigma"][good], d["sersic_n"][good]
H = np.load(RAW / "manga_sep_hist.npy")
assert H.shape[0] == good.sum()

KMS = r"\,\mathrm{km\,s^{-1}}"
classes = [
    (rf"spheroid-dominated, $\sigma_e>160{KMS}$", (sn >= 2.5) & (sig > 160), "#b2182b"),
    (rf"spheroid-dominated, $\sigma_e=70$–$160{KMS}$", (sn >= 2.5) & (sig >= 70) & (sig <= 160), "#f4a582"),
    (rf"disc-dominated, $\sigma_e>160{KMS}$", (sn < 2.5) & (sig > 160), "#2166ac"),
    (rf"disc-dominated, $\sigma_e=70$–$160{KMS}$", (sn < 2.5) & (sig >= 70) & (sig <= 160), "#92c5de"),
    (rf"all types, $\sigma_e<70{KMS}$", sig < 70, "#bababa"),
]


def rebin(h, centres, new):
    """Sum the last axis of h (bin centres `centres`) into the bins `new`."""
    idx = np.digitize(centres, new) - 1
    keep = (centres >= new[0]) & (centres < new[-1])
    return np.stack([h[..., keep & (idx == k)].sum(axis=-1) for k in range(new.size - 1)], axis=-1)


def fractions(hs):
    tot = sum(hs)
    fr = np.array([h / tot for h in hs])
    assert np.allclose(fr.sum(axis=0), 1)
    return fr


# (a) separations, as plot_manga_fractions.py
c = np.sqrt(L.SEP_BINS[1:] * L.SEP_BINS[:-1])
new_s = np.logspace(-1, np.log10(8), 14)
Hc = rebin(H, c, new_s)
fr_s = fractions([Hc[s].sum(axis=0) for _, s, _ in classes])

# (b) delays of 0.1-1" lenses
if CACHE.exists():
    h2 = list(np.load(CACHE)["h2"])
else:
    h2 = [D.population(rows[s], n[s]) for _, s, _ in classes]
    np.savez_compressed(CACHE, h2=np.array(h2))
sc = np.sqrt(D.SEP_E[1:] * D.SEP_E[:-1])
dc = np.sqrt(D.DT_E[1:] * D.DT_E[:-1])
sub = (sc >= 0.1) & (sc < 1.0)
new_t = np.logspace(-0.5, 2.5, 16)
fr_t = fractions([rebin(h[sub].sum(axis=0), dc, new_t) for h in h2])

ct = np.sqrt(new_t[1:] * new_t[:-1])
print("delay [d]: " + " ".join(f"{x:6.1f}" for x in ct))
for (lab, s, _), f in zip(classes, fr_t):
    print(f"{lab:40s} N={s.sum():5d}: " + " ".join(f"{x:6.2f}" for x in f))

plt.rcParams.update({"font.size": 11})
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.6))
for ax, fr, edges in [(a, fr_s, new_s), (b, fr_t, new_t)]:
    base = np.zeros(edges.size - 1)
    for (lab, _, col), f in zip(classes, fr):
        ax.stairs(base + f, edges, baseline=base, fill=True, color=col, label=lab)
        base = base + f
    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, 1)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylabel("fraction of lensing optical depth")
a.set_xticks([0.1, 0.2, 0.5, 1, 2, 5], labels=["0.1", "0.2", "0.5", "1", "2", "5"])
a.set_xlabel("image separation [arcsec]")
a.text(0.99, 0.02, r"MaNGA DynPop, $z_s=2$; spheroid/disc split at Sérsic $n=2.5$",
       transform=a.transAxes, ha="right", va="bottom", fontsize=8.5, color="white")
b.set_xticks([1, 3, 10, 30, 100], labels=["1", "3", "10", "30", "100"])
b.set_xlabel("time delay [days]")
b.text(0.99, 0.02, r"lenses with $0.1''<\Delta\theta<1''$, flux ratio $q\geq0.1$",
       transform=b.transAxes, ha="right", va="bottom", fontsize=8.5, color="white")
h, l = a.get_legend_handles_labels()
order = [0, 1, 2, 3, 4]            # columns: spheroids, discs, low sigma
fig.legend([h[k] for k in order], [l[k] for k in order], frameon=False, fontsize=9.5, ncol=3,
           loc="upper center", columnspacing=1.5, handlelength=1.4)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(L.PLOTS / "manga_type_fractions_delays.png", dpi=150)
