"""Do central black holes turn subcritical discs into lenses?

SPARC discs (solid-body centre, HIPASS weights) with a central point mass from the
Reines & Volonteri 2015 relation for local AGN hosts (arXiv:1508.06274, their eq. 4-5):
    log(M_BH / Msun) = 7.45 + 1.05 log(M_* / 1e11 Msun),   rms 0.55 dex,
with M_* = 0.5 L[3.6]; and M_BH x 10 as an upper bracket (early-type normalisation).
Flux ratio of the two outer images >= 0.1, source at z_s = 2.

plots/bh_test.png
"""
import matplotlib.pyplot as plt
import numpy as np

import delays as D
import lens_population as L
import obs_lenses as O
import sparc_lenses as S


def sparc_mstar():
    """M_* = 0.5 L[3.6] for the templates of sparc_lenses.sparc_profiles, same order."""
    ms = []
    for line in open(O.RAW / "dwarfs" / "SPARC_Lelli2016c.mrt"):
        w = line.split()
        if len(w) < 18 or not w[17].isdigit() or not w[1].isdigit():
            continue
        if int(w[17]) > 2 or float(w[5]) < 30 or int(w[1]) == 0:
            continue
        ms.append(0.5 * float(w[7]) * 1e9)
    return np.array(ms)


def mbh_rv15(mstar):
    return 10 ** (7.45 + 1.05 * np.log10(mstar / 1e11))


def sep_hist(rows, w, edges):
    h = np.zeros(edges.size - 1)
    dz = L.ZL[1] - L.ZL[0]
    for zl in L.ZL:
        scrit, dl = L.sigma_crit(zl)
        dv = L.C_KMS / L.COSMO.Hz(zl) * L.dist_mpc(zl) ** 2 * dz
        for wi, im in zip(w, D.images(rows, dl, scrit)):
            if im is None or im[1].size == 0:
                continue
            da, sep, dphi, q = im
            h += np.histogram(sep, edges, weights=np.full(sep.size, wi * dv * da))[0]
    return h


if __name__ == "__main__":
    rows, vel = S.sparc_profiles("solid body")
    w = S.weights(vel)
    ms = sparc_mstar()
    assert ms.size == vel.size
    mbh = mbh_rv15(ms)
    print(f"SPARC templates {vel.size}: log M_* {np.log10(ms).min():.1f}-{np.log10(ms).max():.1f}, "
          f"log M_BH (RV15) {np.log10(mbh).min():.1f}-{np.log10(mbh).max():.1f}")

    edges = np.logspace(-3, 1, 81)
    cases = {"no black hole": rows,
             "M_BH (Reines & Volonteri 2015)": rows + mbh[:, None],
             "10 x M_BH": rows + 10 * mbh[:, None]}
    zl = 0.6
    scrit, dl = L.sigma_crit(zl)
    hists = {}
    for name, r in cases.items():
        ims = D.images(r, dl, scrit)
        nlens = sum(1 for im in ims if im is not None and im[1].size > 0)
        newly = [i for i, im in enumerate(ims) if im is not None and im[1].size > 0]
        seps = np.concatenate([ims[i][1] for i in newly]) if newly else np.array([])
        hists[name] = sep_hist(r, w, edges)
        c = np.sqrt(edges[1:] * edges[:-1])
        h = hists[name]
        print(f"{name:32s}: lensing templates at z_l=0.6: {nlens:3d}; tau = {h.sum():.2e}; "
              f"tau(>0.1\") = {h[c > 0.1].sum():.2e}; tau(<0.1\") = {h[c <= 0.1].sum():.2e}")

    c = np.sqrt(edges[1:] * edges[:-1])
    dlog = np.diff(np.log10(edges))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for (name, h), sty in zip(hists.items(), [dict(color="C2", lw=2.2), dict(color="k", lw=1.6),
                                              dict(color="k", lw=1.2, ls="--")]):
        ax.step(c, h / dlog, where="mid", label=name, **sty)
    ax.axvline(0.1, color="0.7", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 10)
    ax.set_ylim(1e-8, 3e-4)
    ax.set_xlabel("image separation [arcsec]")
    ax.set_ylabel(r"$d\tau/d\log_{10}\Delta\theta$  ($z_s=2$, flux ratio $\geq0.1$)")
    ax.set_title("HI-rich discs (SPARC) with and without a central black hole", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(L.PLOTS / "bh_test.png", dpi=150)
