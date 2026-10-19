"""Absolute image-separation distributions for a source at z_s = 2, all populations
non-evolving and through the same spherical lensing code.

  E/S0            Choi+07 sigma function + SIS, sigma_SIS = f sigma_* with f = 1
                  (Bolton+08: sigma_e8 = 1.019 +- 0.008 sigma_SIE); band f = 0.95-1.05
  HI-rich Sa+     SPARC spherical-equivalent mass profiles (S0s excluded) weighted by
                  the HIPASS velocity function; solid-body vs flat centre inside the first
                  measured point shown as inner-profile sensitivity (not a bound)
  sum             E/S0 + HI-rich Sa+, illustrative
  Bernardi        all-type stellar sigma function forced to SIS: printed only, not
                  plotted (not a bound; it falls below Choi near 2-3 arcsec)

The halo model and pure NFW haloes are left out until a calibrated galaxy population
exists. No magnification bias. plots/separations_compare.png
"""
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

import lens_population as L
import sparc_lenses as S

CHOI = (8.0e-3 * L.H**3, 161.0, 2.32, 2.67)
BERNARDI = (2.099e-2, 113.78, 0.94, 1.85)
WIDE_BINS = np.logspace(-4, 3, 141)           # overflow check


SIG_COUNTS = 70.0       # Choi+07 counts complete down to here
SIG_SLACS = 160.0       # smallest SLACS sigma: SIS mass calibration tested above here


def sis_population(params, f=1.0, bins=None, srange=(0, np.inf)):
    sig = np.logspace(np.log10(10), np.log10(500), 160)
    rows = 2 * (f * sig[:, None]) ** 2 * L.R3[None, :] / L.G
    w = S.schechter_dlnv(sig, *params) * np.gradient(np.log(sig))
    w = np.where((sig >= srange[0]) & (sig < srange[1]), w, 0.0)
    if bins is None:
        return S.optical_depth(rows, w)
    keep = L.SEP_BINS
    L.SEP_BINS = bins
    try:
        return S.optical_depth(rows, w)
    finally:
        L.SEP_BINS = keep


def tau_sis_analytic(params):
    phistar, sstar, a, b = params
    return (16 * np.pi**3 / 30 * phistar * (sstar / L.C_KMS) ** 4
            * gamma((a + 4) / b) / gamma(a / b) * L.dist_mpc(L.ZS) ** 3)


def fractions(h):
    cdf = np.cumsum(h) / h.sum()
    return [np.interp(np.log10(x), np.log10(L.SEP_BINS[1:]), cdf) for x in (0.5, 1, 2)]


if __name__ == "__main__":
    etg = sis_population(CHOI)
    tested = (SIG_SLACS, np.inf)
    etg_lo = sis_population(CHOI, 0.95, srange=tested)
    etg_hi = sis_population(CHOI, 1.05, srange=tested)
    bern = sis_population(BERNARDI)
    rows, vel = S.sparc_profiles("solid body")
    disc = S.optical_depth(rows, S.weights(vel))
    rows, vel = S.sparc_profiles("flat")
    disc_flat = S.optical_depth(rows, S.weights(vel))
    etg_parts = [sis_population(CHOI, srange=r) for r in
                 [(SIG_SLACS, np.inf), (SIG_COUNTS, SIG_SLACS), (0, SIG_COUNTS)]]
    # Spread for the untested intermediate early types: inner slope gamma = 1.6 inside
    # R_e = 2 kpc (mass inside R_e fixed to the SIS value), as in slope_test.py. Shallow
    # profiles need no flux cut; the steep side (gamma > 2, at most +45% with a flux cut)
    # is not shown.
    sig_i = np.logspace(np.log10(SIG_COUNTS), np.log10(SIG_SLACS), 60)
    w_i = S.schechter_dlnv(sig_i, *CHOI) * np.gradient(np.log(sig_i))
    re_i = 2.0
    m_re = 2 * sig_i**2 * re_i / L.G
    r3 = L.R3[None, :]
    rows_i = np.where(r3 < re_i, m_re[:, None] * (r3 / re_i) ** (3 - 1.6), m_re[:, None] * (r3 / re_i))
    etg_shallow = S.optical_depth(rows_i, w_i)

    # checks
    ta = tau_sis_analytic(CHOI)
    assert abs(etg.sum() / ta - 1) < 2e-3, (etg.sum(), ta)
    wide = sis_population(CHOI, bins=WIDE_BINS)
    outside = 1 - etg.sum() / wide.sum()
    assert outside < 1e-3, outside
    print(f"checks: Choi tau numerical/analytic = {etg.sum() / ta:.4f}; "
          f"share outside the plotted bins = {outside:.1e}")

    res = {
        "E/S0: Choi VDF + SIS": etg,
        "HI-rich Sa+: SPARC + HIPASS (solid-body centre)": disc,
        "HI-rich Sa+: SPARC + HIPASS (flat centre)": disc_flat,
        "all-type stellar VDF forced to SIS (Bernardi)": bern,
    }
    print("optical depth, z_s = 2, fraction below 0.5/1/2 arcsec:")
    for k, h in res.items():
        f = fractions(h)
        print(f"  {k:50s} tau = {h.sum():.2e}   {f[0]:.2f} {f[1]:.2f} {f[2]:.2f}")
    print(f"  E/S0 by sigma: >{SIG_SLACS:.0f} km/s tau = {etg_parts[0].sum():.2e}, "
          f"{SIG_COUNTS:.0f}-{SIG_SLACS:.0f} {etg_parts[1].sum():.2e}, <{SIG_COUNTS:.0f} {etg_parts[2].sum():.2e}")
    print(f"  E/S0 sigma 70-160 with gamma = 1.6 (R_e = 2 kpc): tau = {etg_shallow.sum():.2e} "
          f"(SIS {etg_parts[1].sum():.2e}, ratio {etg_shallow.sum() / etg_parts[1].sum():.2f})")
    print(f"  E/S0 sigma > {SIG_SLACS:.0f}, calibration f = 0.95 / 1.05: tau = {etg_lo.sum():.2e} / {etg_hi.sum():.2e}")

    c = np.sqrt(L.SEP_BINS[1:] * L.SEP_BINS[:-1])
    dlog = np.diff(np.log10(L.SEP_BINS))
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.fill_between(c, etg_lo / dlog, etg_hi / dlog, step="mid", color="C3", alpha=0.18, lw=0,
                    label="_nolegend_")          # caption: sigma_SIS/sigma_* = 0.95-1.05
    ax.step(c, etg_parts[0] / dlog, where="mid", color="C3", lw=2.2,
            label=rf"E/S0, $\sigma>{SIG_SLACS:.0f}$ km/s: counts (Choi) and SIS mass (SLACS) measured"
                  rf"  ($\tau={etg_parts[0].sum():.1e}$)")
    ax.fill_between(c, disc / dlog, disc_flat / dlog, step="mid", color="C2", alpha=0.18, lw=0,
                    label="_nolegend_")          # caption: inner-profile sensitivity
    ax.step(c, disc / dlog, where="mid", color="C2", lw=2.2,
            label=rf"HI-rich Sa+: SPARC + HIPASS, solid-body centre  ($\tau={disc.sum():.1e}$)")
    ax.step(c, disc_flat / dlog, where="mid", color="C2", lw=1.2, ls="--",
            label="_nolegend_")                  # caption: flat centre, tau = 1.3e-4
    ax.fill_between(c, etg_shallow / dlog, etg_parts[1] / dlog, step="mid", facecolor="none",
                    edgecolor="C3", hatch="////", lw=0, alpha=0.5,
                    label="_nolegend_")                # caption: gamma = 1.6-2 inside R_e = 2 kpc
    for part, sty, lab in zip(etg_parts,
                              [dict(color="C3", lw=1.0, ls="-", alpha=0.0),
                               dict(color="C3", lw=1.3, ls="--"), dict(color="C3", lw=1.3, ls=":")],
                              [None, rf"E/S0, $\sigma={SIG_COUNTS:.0f}$-${SIG_SLACS:.0f}$ km/s: counted, SIS mass untested"
                                     rf"  ($\tau={etg_parts[1].sum():.1e}$)",
                               None]):                 # sigma < 70 km/s: 0.2% of E/S0 tau, not plotted
        if lab:
            ax.step(c, part / dlog, where="mid", label=lab, **sty)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.08, 10)
    ax.set_ylim(1e-5, 1.5e-3)
    ax.set_xlabel("image separation [arcsec]")
    ax.set_ylabel(r"$d\tau/d\log_{10}\Delta\theta$  (source at $z_s=2$)")
    # no in-plot footnote: non-evolving, spherical, no magnification bias go in the caption
    # Known lenses (GraL X), on a right-hand axis in lenses per dex. The axis is scaled
    # so that data and the measured E/S0 model hold the same number between 2 and 6
    # arcsec, where known samples should be most complete.
    known = np.array([float(r["max_sep_arcsec"]) for r in
                      csv.DictReader(open(L.ROOT / "data" / "lensed_quasars.csv"))
                      if r["max_sep_arcsec"]])
    kb = np.arange(-1.3, 1.01, 0.1)
    kn, _ = np.histogram(np.log10(known), kb)
    n_data = np.sum((known >= 2) & (known < 6))
    sel = (c >= 2) & (c < 6)
    scale = n_data / etg_parts[0][sel].sum()            # lenses per unit optical depth
    ax2 = ax.twinx()
    ax2.stairs(kn / 0.1, 10**kb, color="k", lw=1.8,
               label=f"known lensed quasars, GraL X ({known.size}; right axis)")
    ax2.set_yscale("log")
    ax2.set_ylim(np.array(ax.get_ylim()) * scale)
    ax2.set_ylabel(r"known lenses per dex (matched to solid red at 2-6$''$)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # compact legend for the empty top-left corner; full descriptions in the caption
    from matplotlib.lines import Line2D
    short = [Line2D([], [], color="C3", lw=2.2, label=r"E/S0, $\sigma>160$ km/s"),
             Line2D([], [], color="C3", lw=1.3, ls="--", label=r"E/S0, $\sigma=70$-$160$ km/s"),
             Line2D([], [], color="C2", lw=2.2, label="HI-rich discs (SPARC)"),
             Line2D([], [], color="k", lw=1.8, label="known lenses (right axis)")]
    ax.legend(handles=short, frameon=False, fontsize=7.5, loc="upper left",
              handlelength=1.8, labelspacing=0.3, borderaxespad=0.4)
    print(f"known lenses: {known.size}; scale {scale:.3g} lenses per unit tau (2-6\" match, N={n_data})")
    for lo, hi in [(0.1, 0.5), (0.5, 1), (1, 2), (2, 6)]:
        sm = (c >= lo) & (c < hi)
        nk = np.sum((known >= lo) & (known < hi))
        print(f"  {lo}-{hi}\": known {nk:4d}, measured-E/S0 model x scale {etg_parts[0][sm].sum() * scale:7.1f}, "
              f"all E/S0 (incl. untested) x scale {(etg_parts[0][sm] + etg_parts[1][sm]).sum() * scale:7.1f}, "
              f"discs x scale {disc[sm].sum() * scale:6.1f}")
    fig.tight_layout()
    fig.savefig(L.PLOTS / "separations_compare.png", dpi=150, bbox_inches="tight")
