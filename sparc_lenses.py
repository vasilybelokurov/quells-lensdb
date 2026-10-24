"""Image-separation distribution from empirical disc lenses: SPARC rotation curves.

Each SPARC galaxy (Q = 1, 2; i > 30 deg) is a spherical lens with the measured
enclosed mass M(<r) = V_obs(r)^2 r / G. Inside the first measured radius r_0 we
bracket the unknown centre:
    solid body  V = V_0 r / r_0     (cored, minimal central mass)
    flat        V = V_0             (isothermal, maximal central mass)
Beyond the last point V stays at its last value out to r_200 = V / (10 H_0).

Abundance: HIPASS late-type rotational velocity function (Zwaan et al. 2010,
arXiv:0912.1754, Sec. 3): dn/dlnV = phi* (V/V*)^a exp[-(V/V*)^b] b / Gamma(a/b),
phi* = 0.061 Mpc^-3, log V* = 2.06, a = 0.66, b = 2.10, with V = V_flat (or the
maximum V_obs when SPARC has no V_flat). The density in each 0.1-dex velocity bin is
shared equally among the SPARC galaxies in that bin.

Early types for comparison, through the same lensing code: SIS with the Choi et al.
2007 velocity dispersion function (phi* = 8.0e-3 h^3 Mpc^-3, sigma* = 161 km/s,
a = 2.32, b = 2.67).

No evolution, spherical lenses (disc inclination ignored), source at z_s = 2,
no magnification bias. plots/separations_sparc.png
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

import paths
from lenscore import cosmology as cosmo
from lenscore import lensing
from lenscore import profiles as prof
from lenscore.population import SEP_BINS, optical_depth

H0_KPC = 70.0 / 1e3                       # km/s/kpc
VBINS = np.arange(1.2, 2.61, 0.1)         # log10 V [km/s]


def schechter_dlnv(v, phistar, vstar, a, b):
    x = v / vstar
    return phistar * x**a * np.exp(-x**b) * b / gamma(a / b)


def sparc_profiles(inner):
    """Enclosed-mass rows on lensing.R3 and the velocity used for the abundance weight."""
    rows, vel = [], []
    for line in open(paths.RAW / "dwarfs" / "SPARC_Lelli2016c.mrt"):
        w = line.split()
        if len(w) < 18 or not w[17].isdigit() or not w[1].isdigit():
            continue
        name, htype, fdist = w[0], int(w[1]), int(w[4])
        inc, vflat, q = float(w[5]), float(w[15]), int(w[17])
        # HIPASS counts Sa and later types, so S0 (T = 0) templates are excluded.
        if q > 2 or inc < 30 or htype == 0:
            continue
        rc = np.loadtxt(paths.RAW / "dwarfs" / "rotmod" / f"{name}_rotmod.dat")
        # Hubble-flow distances (f_D = 1) assume H0 = 73; rescale radii to H0 = 70.
        r, v = rc[:, 0] * (73.0 / 70.0 if fdist == 1 else 1.0), rc[:, 1]
        vtag = vflat if vflat > 0 else v.max()
        vr = np.interp(lensing.R3, r, v)
        if inner == "solid body":
            vr = np.where(lensing.R3 < r[0], v[0] * lensing.R3 / r[0], vr)
        r200 = v[-1] / (10 * H0_KPC)
        m = vr**2 * np.minimum(lensing.R3, r200) / cosmo.G
        rows.append(np.maximum.accumulate(m))
        vel.append(vtag)
    return np.array(rows), np.array(vel)


def weights(vel):
    """Comoving number density [Mpc^-3] assigned to each galaxy."""
    lv = np.log10(vel)
    w = np.zeros(vel.size)
    vg = np.logspace(VBINS[0], VBINS[-1], 2001)
    phi = schechter_dlnv(vg, 0.061, 10**2.06, 0.66, 2.10)
    for lo, hi in zip(VBINS[:-1], VBINS[1:]):
        sel = (lv >= lo) & (lv < hi)
        g = (vg >= 10**lo) & (vg < 10**hi)
        n_bin = np.trapezoid(phi[g], np.log(vg[g]))
        if sel.any():
            w[sel] = n_bin / sel.sum()
    return w


def sis_rows():
    sig = np.logspace(np.log10(30), np.log10(450), 120)
    m = prof.sis(sig)
    w = schechter_dlnv(sig, 8.0e-3 * cosmo.H**3, 161.0, 2.32, 2.67) * np.gradient(np.log(sig))
    return m, w


def summary(name, h):
    cdf = np.cumsum(h) / h.sum()
    f = [np.interp(np.log10(x), np.log10(SEP_BINS[1:]), cdf) for x in (0.5, 1, 2)]
    print(f"  {name:34s} tau = {h.sum():.2e}   below 0.5/1/2\": {f[0]:.2f} {f[1]:.2f} {f[2]:.2f}")


if __name__ == "__main__":
    res = {}
    for inner in ("solid body", "flat"):
        rows, vel = sparc_profiles(inner)
        w = weights(vel)
        res[f"SPARC discs, inner {inner}"] = optical_depth(rows, w)
    print(f"SPARC lenses: {len(vel)} galaxies, V = {vel.min():.0f}-{vel.max():.0f} km/s; "
          f"disc density in SPARC-covered bins {w.sum():.3e} Mpc^-3")
    m, w = sis_rows()
    res["early types, SIS + Choi+07"] = optical_depth(m, w)
    res["early types + discs (solid body)"] = (res["early types, SIS + Choi+07"]
                                                + res["SPARC discs, inner solid body"])
    print("optical depth for a source at z_s = 2:")
    for k, h in res.items():
        summary(k, h)

    centres = np.sqrt(SEP_BINS[1:] * SEP_BINS[:-1])
    dlog = np.diff(np.log10(SEP_BINS))
    sty = {"SPARC discs, inner solid body": dict(color="C2", lw=2),
           "SPARC discs, inner flat": dict(color="C2", lw=1.4, ls="--"),
           "early types, SIS + Choi+07": dict(color="C3", lw=2),
           "early types + discs (solid body)": dict(color="k", lw=2.4)}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for k, h in res.items():
        ax.step(centres, h / dlog, where="mid", label=k, **sty[k])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.03, 10)
    ax.set_ylim(1e-7, 3e-3)
    ax.set_xlabel("image separation [arcsec]")
    ax.set_ylabel(r"$d\tau/d\log_{10}\Delta\theta$  (source at $z_s=2$)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(paths.PLOTS / "separations_sparc.png", dpi=150)
