"""Image-separation distribution from the MaNGA DynPop population (manga_population.py):
every galaxy a spherical lens with its JAM total mass profile (stars from the MGE x
stellar M/L, gNFW dark halo), weighted by its volume-corrected number density.
Outer haloes beyond the kinematic data are abundance-matched (see manga_population.py).
Same lensing code and assumptions as the other populations (source at z_s = 2,
non-evolving, no magnification bias, no flux cut).

Split by measured sigma_e (> 160, 70-160, < 70 km/s) and by Sersic index (n >= 2.5
spheroid-dominated, n < 2.5 disc-dominated). Compared with the E/S0 SIS model (Choi+07)
and the SPARC discs, and with the known lenses (GraL X).

Run manga_population.py first (it writes data/raw/manga/manga_profiles.npz).
plots/separations_manga.png
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

import compare_separations as CS
import paths
import sparc_lenses as S
from lenscore.population import SEP_BINS, optical_depth

RAW = paths.RAW / "manga"


def per_galaxy_hist(rows, w):
    """Separation histogram for each galaxy (rows x bins), summed over lens redshift."""
    return optical_depth(rows, w, per_row=True)


if __name__ == "__main__":
    d = np.load(RAW / "manga_profiles.npz", allow_pickle=True)
    rows, n, sig, sn = d["mt"], d["n"], d["sigma"], d["sersic_n"]
    good = np.isfinite(sig)
    rows, n, sig, sn = rows[good], n[good], sig[good], sn[good]
    H = per_galaxy_hist(rows, n)
    tot = H.sum(axis=0)
    raw = per_galaxy_hist(d["mt_raw"][good], n).sum(axis=0)
    cc = np.sqrt(SEP_BINS[1:] * SEP_BINS[:-1])
    print(f"outer halo: abundance-matched tau = {tot.sum():.2e}; raw gNFW extrapolation tau = {raw.sum():.2e} "
          f"(ratio {raw.sum() / tot.sum():.2f}); share beyond 4\": am {tot[cc > 4].sum() / tot.sum():.3f}, "
          f"raw {raw[cc > 4].sum() / raw.sum():.3f}")
    c = np.sqrt(SEP_BINS[1:] * SEP_BINS[:-1])
    dlog = np.diff(np.log10(SEP_BINS))

    groups = {
        r"$\sigma_e>160$ km/s": sig > 160,
        r"$\sigma_e=70$-$160$ km/s": (sig >= 70) & (sig <= 160),
        r"$\sigma_e<70$ km/s": sig < 70,
        r"Sersic $n\geq2.5$": sn >= 2.5,
        r"Sersic $n<2.5$": sn < 2.5,
    }
    def fr(h):
        cdf = np.cumsum(h) / h.sum()
        return [np.interp(np.log10(x), np.log10(SEP_BINS[1:]), cdf) for x in (0.5, 1, 2)]
    print(f"MaNGA galaxies used: {len(n)}; lensing (any z): {np.sum(H.sum(axis=1) > 0)}")
    f = fr(tot)
    print(f"MaNGA total: tau = {tot.sum():.2e}; below 0.5/1/2\": {f[0]:.2f} {f[1]:.2f} {f[2]:.2f}")
    for k, s in groups.items():
        h = H[s].sum(axis=0)
        print(f"  {k:28s} N={s.sum():5d}  tau = {h.sum():.2e}  share of total {h.sum() / tot.sum():.2f}; "
              + "share at 0.1-0.5/0.5-1/1-2\": "
              + " ".join(f"{h[(c >= a) & (c < b)].sum() / max(tot[(c >= a) & (c < b)].sum(), 1e-30):.2f}"
                         for a, b in [(0.1, 0.5), (0.5, 1), (1, 2)]))
    # by sigma x Sersic in the sub-arcsecond range
    sub = (c >= 0.1) & (c < 1)
    print("sub-arcsecond (0.1-1\") optical depth shares by sigma_e and Sersic n:")
    for sl, ss in [(">160", sig > 160), ("70-160", (sig >= 70) & (sig <= 160))]:
        for nl, ns in [("n>=2.5", sn >= 2.5), ("n<2.5", sn < 2.5)]:
            s = ss & ns
            print(f"  sigma {sl:7s} {nl:7s}: N={s.sum():5d}, share {H[s][:, sub].sum() / tot[sub].sum():.2f}")

    etg = CS.sis_population(CS.CHOI, srange=(CS.SIG_SLACS, np.inf))
    etg_i = CS.sis_population(CS.CHOI, srange=(CS.SIG_COUNTS, CS.SIG_SLACS))
    srows, vel = S.sparc_profiles("solid body")
    disc = optical_depth(srows, S.weights(vel))
    print(f"for comparison: Choi E/S0 SIS tau = {(etg + etg_i).sum():.2e} "
          f"(sigma>160 {etg.sum():.2e}); SPARC discs {disc.sum():.2e}; "
          f"MaNGA / (Choi + SPARC) = {tot.sum() / (etg + etg_i + disc).sum():.2f}")

    known = np.array([float(r["max_sep_arcsec"]) for r in
                      csv.DictReader(open(paths.KNOWN)) if r["max_sep_arcsec"]])

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.step(c, tot / dlog, where="mid", color="k", lw=2.6, label="MaNGA DynPop, all galaxies")
    for (k, s), col, ls in zip(list(groups.items())[:3], ["C3", "C1", "C0"], ["-", "-", "-"]):
        ax.step(c, H[s].sum(axis=0) / dlog, where="mid", color=col, lw=1.6, ls=ls, label=f"MaNGA, {k}")
    ax.step(c, H[sn < 2.5].sum(axis=0) / dlog, where="mid", color="C2", lw=1.6, ls="--",
            label=r"MaNGA, Sersic $n<2.5$ (disc-dominated)")
    ax.step(c, (etg + etg_i) / dlog, where="mid", color="C3", lw=1, ls=":", label="Choi E/S0 as SIS (all $\\sigma$)")
    ax.step(c, disc / dlog, where="mid", color="C2", lw=1, ls=":", label="SPARC discs (solid-body centre)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.08, 10)
    ax.set_ylim(1e-6, 2e-3)
    ax.set_xlabel("image separation [arcsec]")
    ax.set_ylabel(r"$d\tau/d\log_{10}\Delta\theta$  (source at $z_s=2$)")
    ax2 = ax.twinx()
    kb = np.arange(-1.3, 1.01, 0.1)
    kn, _ = np.histogram(np.log10(known), kb)
    sel = (c >= 2) & (c < 6)
    scale = np.sum((known >= 2) & (known < 6)) / tot[sel].sum()
    ax2.stairs(kn / 0.1, 10**kb, color="0.4", lw=1.4)
    ax2.set_yscale("log")
    ax2.set_ylim(np.array(ax.get_ylim()) * scale)
    ax2.set_ylabel(r"known lenses per dex (matched to MaNGA total at 2-6$''$)")
    ax.plot([], [], color="0.4", lw=1.4, label="known lenses (right axis)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(paths.PLOTS / "separations_manga.png", dpi=150)
    np.save(RAW / "manga_sep_hist.npy", H)
    for lo, hi in [(0.1, 0.5), (0.5, 1), (1, 2), (2, 6)]:
        s = (c >= lo) & (c < hi)
        print(f"  known/predicted (MaNGA, matched at 2-6\") {lo}-{hi}\": "
              f"{np.sum((known >= lo) & (known < hi)) / (tot[s].sum() * scale):.2f}")
