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

import matplotlib.pyplot as plt
import numpy as np

import compare_separations as CS
import lens_population as L
import sparc_lenses as S

QMIN, NBETA = 0.1, 24
DAY = 86400.0
MPC_M = 3.0857e22
SEP_E = np.logspace(-1.3, 1.0, 47)
DT_E = np.logspace(-1.5, 3.5, 51)


def dt_distance_m(zl, zs=L.ZS):
    dcl, dcs = L.dist_mpc(zl), L.dist_mpc(zs)
    return (1 + zl) * (dcl / (1 + zl)) * (dcs / (1 + zs)) / ((dcs - dcl) / (1 + zs)) * MPC_M


def images(rows, dl, scrit):
    """Per row: equal-area source samples passing the flux cut -> (sep", dt-factor rad^2, q)."""
    dm = np.diff(rows, axis=1)
    dm[:, 0] += rows[:, 0]
    m2d = dm @ L.PROJ.T
    kbar = m2d / (np.pi * L.R2**2 * scrit)
    kap = 0.5 * kbar * np.gradient(np.log(np.maximum(m2d, 1e-30)), np.log(L.R2), axis=1)
    theta = L.R2 / dl
    alpha = theta[None, :] * kbar
    psi = np.concatenate([np.zeros((len(rows), 1)),
                          np.cumsum(0.5 * (alpha[:, 1:] + alpha[:, :-1]) * np.diff(theta), axis=1)], axis=1)
    g = theta[None, :] - alpha
    inv_mu = (1 - kbar) * (1 - 2 * kap + kbar)
    pb = (np.arange(NBETA) + 0.5) / NBETA
    out = []
    for i in range(len(rows)):
        br = -g[i].min()
        if br <= 0:
            out.append(None)
            continue
        te = L.outermost_root(g[i], theta, np.array([0.0]))[0]
        if np.isfinite(te):
            br = min(br, 2 * te)
        beta = br * np.sqrt(pb)
        ta = L.outermost_root(g[i], theta, beta)
        tb = L.outermost_root(g[i], theta, -beta)
        mua = 1 / np.abs(np.interp(ta, theta, inv_mu[i]))
        mub = 1 / np.abs(np.interp(tb, theta, inv_mu[i]))
        q = np.minimum(mua, mub) / np.maximum(mua, mub)
        ok = np.isfinite(ta) & np.isfinite(tb) & (q >= QMIN)
        phia = 0.5 * (ta - beta) ** 2 - np.interp(ta, theta, psi[i])
        phib = 0.5 * (tb + beta) ** 2 - np.interp(tb, theta, psi[i])
        out.append((np.pi * br**2 / NBETA, ((ta + tb) / L.ARCSEC)[ok], (phib - phia)[ok], q[ok]))
    return out


def population(rows, w):
    """2D optical-depth histogram in (log sep, log dt) and per-sample arrays."""
    h2 = np.zeros((SEP_E.size - 1, DT_E.size - 1))
    dz = L.ZL[1] - L.ZL[0]
    for zl in L.ZL:
        scrit, dl = L.sigma_crit(zl)
        ddt = dt_distance_m(zl)
        dv = L.C_KMS / L.COSMO.Hz(zl) * L.dist_mpc(zl) ** 2 * dz
        for wi, im in zip(w, images(rows, dl, scrit)):
            if im is None or im[1].size == 0:
                continue
            da, sep, dphi, q = im
            dt = ddt / (L.C_KMS * 1e3) * dphi / DAY
            h2 += np.histogram2d(sep, dt, [SEP_E, DT_E], weights=np.full(sep.size, wi * dv * da))[0]
    return h2


def sis_rows(sig):
    return 2 * sig[:, None] ** 2 * L.R3[None, :] / L.G


def slope_rows(sig, gamma, re=2.0):
    m_re = 2 * sig**2 * re / L.G
    r = L.R3[None, :]
    return np.where(r < re, m_re[:, None] * (r / re) ** (3 - gamma), m_re[:, None] * (r / re))


def check_sis():
    zl, s = 0.6, 200.0
    scrit, dl = L.sigma_crit(zl)
    da, sep, dphi, q = images(sis_rows(np.array([s])), dl, scrit)[0]
    dt = dt_distance_m(zl) / (L.C_KMS * 1e3) * dphi / DAY
    expect = dt_distance_m(zl) / (2 * L.C_KMS * 1e3) * (sep * L.ARCSEC) ** 2 * (1 - q) / (1 + q) / DAY
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
        "massive E/S0 (SIS)": population(sis_rows(sig_m), wgt(sig_m)),
        "intermediate E/S0 (SIS)": population(sis_rows(sig_i), wgt(sig_i)),
        r"intermediate E/S0 ($\gamma=1.6$)": population(slope_rows(sig_i, 1.6), wgt(sig_i)),
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
             for r in csv.DictReader(open(L.ROOT / "data" / "lensed_quasars.csv"))
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
        a1.plot(dd, 3e3 * MPC_M / (2 * L.C_KMS * 1e3) * (dd * L.ARCSEC) ** 2 * (1 - qq) / (1 + qq) / DAY,
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
    fig.savefig(L.PLOTS / "delays.png", dpi=150)
