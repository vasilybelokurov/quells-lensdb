"""Joint image-separation / time-delay distribution of the populations in the note.

Populations (source at z_s = 2, non-evolving, spherical):
  massive E/S0       SIS, sigma > 160 km/s (Choi+07 counts)
  intermediate E/S0  sigma = 70-160 km/s, SIS and rho ∝ r^-1.6 inside R_e = 2 kpc
  HI-rich discs      SPARC rotation curves (solid-body centre) + HIPASS weights
Delay between the two outer images from the Fermat potential,
    dt = D_dt / c [phi(theta_B) - phi(theta_A)],  phi = (theta - beta)^2 / 2 - psi,
    psi(theta) = int alpha dtheta,  D_dt = (1 + z_l) D_l D_s / D_ls.
Only sources whose fainter outer image has >= 10% of the brighter one's flux count
(the QUELLS calibration range). Check: SIS delays against (D_dt / 2c) dtheta^2 (1-q)/(1+q).

plots/delays.png
"""
import csv
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import compare_separations as CS
import lenscore
import paths
import sparc_lenses as S
from lenscore import cosmology as cosmo
from lenscore import lensing
from lenscore import profiles as prof
from lenscore.cosmology import DAY, MPC_M, dt_distance_m
from lenscore.population import DT_E, SEP_E, ZL, optical_depth

QMIN, NBETA = 0.1, 24


def population(rows, w):
    """Optical depth in (separation, delay) bins, flux ratio >= QMIN."""
    return optical_depth(rows, w, nbeta=NBETA, qmin=QMIN, bins=SEP_E, dt_bins=DT_E)


def cached_population(path, groups):
    """population() for each name -> (rows, w), stored in an npz together with a key over
    the inputs, the settings and the lenscore source; recomputed whenever the key differs."""
    h = hashlib.sha1()
    for f in ("cosmology.py", "lensing.py", "population.py"):
        h.update((Path(lenscore.__file__).parent / f).read_bytes())
    h.update(repr((QMIN, NBETA, SEP_E.tolist(), DT_E.tolist(), ZL.tolist())).encode())
    for k, (rows, w) in groups.items():
        h.update(k.encode())
        h.update(np.ascontiguousarray(rows).tobytes())
        h.update(np.ascontiguousarray(w).tobytes())
    key = h.hexdigest()
    if path.exists():
        d = np.load(path)
        if "key" in d and str(d["key"]) == key:
            return {k: d[k] for k in groups}
    out = {}
    for k, (rows, w) in groups.items():
        out[k] = population(rows, w)
        print(f"delays computed for {k}: N = {len(rows)}")
    np.savez_compressed(path, key=key, **out)
    return out


def check_sis():
    zl, s = 0.6, 200.0
    scrit, dl = cosmo.sigma_crit(zl)
    _, sep, dphi, q = lensing.images(prof.sis([s]), dl, scrit, NBETA, QMIN)
    ok = np.isfinite(sep[0])
    sep, dphi, q = sep[0, ok], dphi[0, ok], q[0, ok]
    dt = dt_distance_m(zl) / (cosmo.C_KMS * 1e3) * dphi / DAY
    expect = dt_distance_m(zl) / (2 * cosmo.C_KMS * 1e3) * (sep * cosmo.ARCSEC) ** 2 * (1 - q) / (1 + q) / DAY
    err = np.max(np.abs(dt / expect - 1))
    assert err < 0.02, err
    print(f"check: SIS delays vs (D_dt/2c) sep^2 (1-q)/(1+q): max deviation {err:.3%}; "
          f"D_dt(z_l=0.6, z_s=2) = {dt_distance_m(zl) / MPC_M / 1e3:.2f} Gpc")


def median2(h, axis_edges, sel):
    """Weighted median of the delay for separation bins sel."""
    p = h[sel].sum(axis=0)
    cdf = np.cumsum(p) / p.sum()
    centres = np.sqrt(axis_edges[1:] * axis_edges[:-1])
    return np.interp(0.5, cdf, centres)


if __name__ == "__main__":
    check_sis()
    sig_m = np.logspace(np.log10(CS.SIG_SLACS), np.log10(500), 60)
    sig_i = np.logspace(np.log10(CS.SIG_COUNTS), np.log10(CS.SIG_SLACS), 50)
    wgt = lambda s: S.schechter_dlnv(s, *CS.CHOI) * np.gradient(np.log(s))
    pops = {
        "massive E/S0 (SIS)": population(prof.sis(sig_m), wgt(sig_m)),
        "intermediate E/S0 (SIS)": population(prof.sis(sig_i), wgt(sig_i)),
        r"intermediate E/S0 ($\gamma=1.6$)": population(prof.power_law(sig_i, 1.6, 2.0), wgt(sig_i)),
    }
    rows, vel = S.sparc_profiles("solid body")
    pops["HI-rich discs (SPARC)"] = population(rows, S.weights(vel))

    sc = np.sqrt(SEP_E[1:] * SEP_E[:-1])
    dc = np.sqrt(DT_E[1:] * DT_E[:-1])
    print("median delay [d] (q >= 0.1) and share with < 120 d (reachable) / > 10 d (recovery >~ 50%):")
    for name, h in pops.items():
        for lo, hi in [(0.1, 0.5), (0.5, 1.0), (1.0, 2.0)]:
            sel = (sc >= lo) & (sc < hi)
            p = h[sel].sum(axis=0)
            if p.sum() == 0:
                continue
            f_win = p[dc <= 120].sum() / p.sum()
            f10 = p[(dc > 10) & (dc <= 120)].sum() / p.sum()
            print(f"  {name:32s} {lo}-{hi}\": median {median2(h, DT_E, sel):6.1f} d, "
                  f"<120 d {f_win:.2f}, 10-120 d {f10:.2f}")
    tot = sum(pops[k] for k in ("massive E/S0 (SIS)", "intermediate E/S0 (SIS)", "HI-rich discs (SPARC)"))
    for lo, hi in [(0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 6.0)]:
        sel = (sc >= lo) & (sc < hi)
        p = tot[sel].sum(axis=0)
        print(f"  all (SIS E/S0 + discs) {lo}-{hi}\": median {median2(tot, DT_E, sel):6.1f} d, "
              f"<120 d {p[dc <= 120].sum() / p.sum():.2f}, 10-120 d {p[(dc > 10) & (dc <= 120)].sum() / p.sum():.2f}")

    known = [(float(r["max_sep_arcsec"]), float(r["max_abs_delay_d"]), r["type"])
             for r in csv.DictReader(open(paths.KNOWN))
             if r["max_sep_arcsec"] and r["max_abs_delay_d"]]
    ks, kd, kt = map(np.array, zip(*known))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4))
    for name, col in [("massive E/S0 (SIS)", "C3"), ("intermediate E/S0 (SIS)", "C1"),
                      ("HI-rich discs (SPARC)", "C2")]:
        h = pops[name]
        z = np.log10(np.where(h > 0, h, np.nan) / h.max())
        a1.contour(sc, dc, z.T, levels=[-2, -1, -0.3], colors=col, linewidths=[0.8, 1.3, 2.0],
                   negative_linestyles="solid")
        a1.plot([], [], color=col, lw=1.6, label=name)
    dd = np.logspace(-1.3, 1, 50)
    for qq, ls in [(0.1, "--"), (0.5, ":")]:
        a1.plot(dd, 3e3 * MPC_M / (2 * cosmo.C_KMS * 1e3) * (dd * cosmo.ARCSEC) ** 2 * (1 - qq) / (1 + qq) / DAY,
                color="0.4", lw=1, ls=ls, label=rf"SIS, $D_{{\Delta t}}=3$ Gpc, $q={qq}$")
    doubles = kt == "Double"
    a1.scatter(ks[doubles], kd[doubles], s=18, color="k", zorder=5, label="known doubles (GraL X)")
    a1.scatter(ks[~doubles], kd[~doubles], s=18, facecolors="none", edgecolors="k", zorder=5,
               label="known quads/triples (largest delay)")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlim(0.08, 8)
    a1.set_ylim(0.1, 2000)
    a1.set_xlabel("image separation [arcsec]")
    a1.set_ylabel("time delay [days]")
    a1.set_title(r"contours: 1%, 10%, 50% of peak (flux ratio $\geq0.1$)", fontsize=10)
    a1.legend(frameon=False, fontsize=7.5, loc="upper left")

    sel = (sc >= 0.1) & (sc < 1.0)
    dlog = np.diff(np.log10(DT_E))
    for name, sty in [("massive E/S0 (SIS)", dict(color="C3", lw=2)),
                      ("intermediate E/S0 (SIS)", dict(color="C1", lw=2)),
                      (r"intermediate E/S0 ($\gamma=1.6$)", dict(color="C1", lw=1.4, ls="--")),
                      ("HI-rich discs (SPARC)", dict(color="C2", lw=2))]:
        a2.step(dc, pops[name][sel].sum(axis=0) / dlog, where="mid", label=name, **sty)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlim(0.1, 300)
    a2.set_ylim(1e-7, 2e-4)
    a2.set_xlabel("time delay [days]")
    a2.set_ylabel(r"$d\tau/d\log_{10}\Delta t$, lenses with $0.1''<\Delta\theta<1''$")
    a2.set_title("delays of sub-arcsecond lenses", fontsize=10)
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(paths.PLOTS / "delays.png", dpi=150)
