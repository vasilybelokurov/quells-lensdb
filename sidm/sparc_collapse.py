"""Core collapse in discs: the parametric gravothermal SIDM model (parametric.py, Yang 2024) for the SPARC
galaxies of sparc_sidm.py, with their measured baryons.

Baryons: M_b(r) of sparc_sidm.baryons (Y_d = 0.5, Y_b = 0.7) drives the adiabatic contraction of the
progenitor; the model's baryon terms in t_c,b and f_in need a Hernquist (M, r_H), fitted to M_b(r) at the
measured radii (least squares in log M). Age T = 10 Gyr (as sparc_sidm.py), so tau = (sigma/m) T / t_c,b(1).

Two stages:
  forward  the CDM (NFW) fit of sparc_sidm.py taken as the initial halo; tau, F_t and the rotation-curve chi^2
           of the SIDM halo it would become, without refitting.
  fit      (log M200, log c) of the initial halo refitted to V_obs for each sigma/m, with the same prior on c
           (0.11 dex around Diemer+19); lensing rows as sparc_sidm.lensing_rows (measured mass outside r0, model
           inside, joined at r0). sigma/m = 0 is the contracted-CDM limit of the model.

Result (2026-10-24, 150 galaxies; lensing by sparc_lensing.py --model collapse):
  forward: the CDM-fit haloes of massive discs (V_flat > 150 km/s) have t_c(1 cm^2/g) ~ 11 Gyr (F_t ~ 0.03,
  baryon-dominated centres, often high fitted c), so half of them would be collapsed at sigma/m = 1, and their
  rotation curves would then be fitted much worse (median dchi2 +600). Below 150 km/s t_c ~ 300-400 Gyr.
  refit: the halo adjusts; collapsed (tau > 1) massive discs 6/52, 10/52, 15/52, 24/52, 37/52 for sigma/m = 1, 3,
  10, 30, 100 with no chi2 penalty up to 30 (median dchi2 within +-1) and +12 at 100; discs below 150 km/s
  almost never collapse and prefer SIDM cores (median dchi2 -7 to -13). Disc optical depth 4.48e-5 (contracted
  CDM limit) -> 4.51-4.56e-5, median delay 12.2 -> 11.9-12.0 d: collapse leaves disc lensing unchanged, because
  the rotation curve fixes the mass at the Einstein radius.

    python sidm/sparc_collapse.py [--forward] [--test]   -> data/sidm/sparc_collapse.npz
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import parametric as P
import sparc_sidm as SS
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
from lenscore import cosmology as cosmo  # noqa: E402
from lenscore import lensing  # noqa: E402
from lenscore import profiles as prof  # noqa: E402

SIGMA_M = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0]
T_GYR = SS.T_GYR


def hernquist_fit(g, m_b):
    """(M, r_H, rms) of the Hernquist profile with the measured baryon mass M = M_b(r_last) and the same
    half-mass radius, r_half = (1 + sqrt 2) r_H; rms of log M_H / M_b at the measured radii. (A free fit of
    both runs away for gas-rich galaxies whose M_b still rises at the last point.)"""
    r = g["r"]
    rr = np.geomspace(1e-3, r[-1], 2000)
    mb = m_b(rr)
    m = mb[-1]
    rh = np.interp(0.5 * m, mb, rr) / (1 + np.sqrt(2))
    good = m_b(r) > 0
    rms = np.sqrt(np.mean(np.log(m * r[good] ** 2 / (r[good] + rh) ** 2 / m_b(r)[good]) ** 2))
    return m, rh, rms


def halo(p, mh, rh, m_b):
    """parametric.Halo for the initial NFW (log M200, log c) and the disc baryons."""
    return P.Halo(10 ** p[0], 0.0, mh, 1.8153 * rh, c=10 ** p[1], m_b=m_b)


def vcirc(h, tau, r):
    return np.sqrt(cosmo.G * (h.m_dm(tau, r) + h.m_b(r)) / r)


def chi2_of(g, h, tau):
    vm = vcirc(h, tau, g["r"])
    return np.sum(((vm - g["v"]) / g["ev"]) ** 2)


def forward(args):
    g, p_cdm = args
    try:
        m_b, mbt = SS.baryons(g)
        mh, rh, hrms = hernquist_fit(g, m_b)
        h = halo(p_cdm, mh, rh, m_b)
        taus = [h.tau(s, T_GYR) for s in SIGMA_M]
        return g["name"], (mh, rh, hrms, h.Ft, h.tc1, h.a), taus, [chi2_of(g, h, t) for t in taus]
    except Exception as e:
        return g["name"], repr(e), None, None


def fit(g):
    m_b, mbt = SS.baryons(g)
    mh, rh, hrms = hernquist_fit(g, m_b)

    def chi2(p, s):
        if not (8 < p[0] < 15 and -0.5 < p[1] < 2):
            return 1e12
        try:
            h = halo(p, mh, rh, m_b)
        except Exception:
            return 1e12
        prior = ((p[1] - np.log10(prof.concentration(10 ** p[0], 0.0))) / SS.C_SCATTER) ** 2
        return chi2_of(g, h, h.tau(s, T_GYR)) + prior

    out = []
    for s in SIGMA_M:
        grid = [(lm, lc) for lm in np.arange(9.5, 13.6, 0.5) for lc in np.arange(0.4, 1.6, 0.2)]
        vals = [chi2(q, s) for q in grid]
        res = minimize(chi2, np.array(grid[int(np.argmin(vals))]), args=(s,), method="Nelder-Mead",
                       options=dict(xatol=1e-3, fatol=1e-2, maxiter=300))
        h = halo(res.x, mh, rh, m_b)
        out.append((res.x, res.fun, res.fun < 1e11, h.tau(s, T_GYR), h.Ft))
    return out, (mh, rh, hrms)


def lensing_rows(g, fits, hern):
    m_b, _ = SS.baryons(g)
    r, v = g["r"], g["v"]
    vr = np.interp(lensing.R3, r, v)
    r200 = v[-1] / (10 * 70.0 / 1e3)
    m_meas = np.maximum.accumulate(vr**2 * np.minimum(lensing.R3, r200) / cosmo.G)
    m0_meas = v[0] ** 2 * r[0] / cosmo.G
    rows = []
    for s, f in zip(SIGMA_M, fits):
        h = halo(f[0], hern[0], hern[1], m_b)
        tau = h.tau(s, T_GYR)
        model = h.m_dm(tau) + m_b(lensing.R3)
        m0 = h.m_dm(tau, np.array([r[0]]))[0] + m_b(r[0])
        rows.append(np.where(lensing.R3 < r[0], model * m0_meas / m0, m_meas))
    return np.array(rows)


def task(g):
    try:
        fits, hern = fit(g)
        return g["name"], fits, hern, lensing_rows(g, fits, hern)
    except Exception as e:
        return g["name"], repr(e), None, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", action="store_true", help="forward stage only (no refits)")
    ap.add_argument("--test", action="store_true", help="three galaxies only")
    args = ap.parse_args()
    gals = SS.galaxies()
    if args.test:
        gals = [g for g in gals if g["name"] in ("DDO154", "NGC3198", "NGC2841")]
    print(f"{len(gals)} SPARC galaxies; sigma/m = {SIGMA_M} cm^2/g, T = {T_GYR} Gyr", flush=True)

    if args.forward:
        f = np.load(paths.ROOT / "data" / "sidm" / "sparc_sidm.npz")
        pcdm = {n: np.log10(p[0, 2:4]) for n, p in zip(f["names"], f["params"])}
        gals = [g for g in gals if g["name"] in pcdm]
        with ProcessPoolExecutor(13) as ex:
            res = list(ex.map(forward, [(g, pcdm[g["name"]]) for g in gals], chunksize=2))
        vel = np.array([g["vtag"] for g in gals])
        ok = [r[2] is not None for r in res]
        for r in res:
            if r[2] is None:
                print(f"  {r[0]}: failed {r[1]}")
        info = np.array([r[1] for r in res if r[2] is not None])
        tau = np.array([r[2] for r in res if r[2] is not None])
        chi = np.array([r[3] for r in res if r[2] is not None])
        vel = vel[ok]
        print("Hernquist fit rms (dex): median %.3f, max %.3f" % (np.median(info[:, 2]) / np.log(10),
                                                                 info[:, 2].max() / np.log(10)))
        print("tau at T = 10 Gyr (median and max per V_flat bin), F_t, and median chi2 change without refit")
        for lo, hi in [(0, 80), (80, 150), (150, 400)]:
            sel = vel >= lo
            sel &= vel < hi
            print(f"  V_flat {lo}-{hi} (N={sel.sum()}): F_t median {np.median(info[sel, 3]):.2f}  t_c(1 cm2/g) "
                  f"median {np.median(info[sel, 4]):.0f} Gyr, min {info[sel, 4].min():.0f} Gyr")
            for j, s in enumerate(SIGMA_M[1:], 1):
                print(f"     s/m {s:5g}: tau median {np.median(tau[sel, j]):.3f} max {tau[sel, j].max():.3f}  "
                      f"collapsed (tau>1) {np.sum(tau[sel, j] > 1):3d}   dchi2 median "
                      f"{np.median(chi[sel, j] - chi[sel, 0]):+8.1f}")
        np.savez(paths.ROOT / "data" / "sidm" / "sparc_collapse_forward.npz", names=np.array([g["name"] for g, o in zip(gals, ok) if o]),
                 info=info, tau=tau, chi2=chi, vel=vel, sigma_m=np.array(SIGMA_M))
        sys.exit()

    with ProcessPoolExecutor(13) as ex:
        res = list(ex.map(task, gals, chunksize=1))
    names, rows, params, chi, taus, vel, herns = [], [], [], [], [], [], []
    for (name, fits, hern, rw), g in zip(res, gals):
        if rw is None:
            print(f"  {name}: failed ({fits})")
            continue
        names.append(name)
        rows.append(rw)
        params.append([f[0] for f in fits])
        chi.append([f[1] for f in fits])
        taus.append([f[3] for f in fits])
        herns.append(hern)
        vel.append(g["vtag"])
        print(f"  {name:12s} V {g['vtag']:5.0f}  " + "  ".join(
            f"s{s:g}: chi2 {f[1]:6.1f} tau {f[3]:5.2f}" for s, f in zip(SIGMA_M, fits)), flush=True)
    np.savez(paths.ROOT / "data" / "sidm" / ("sparc_collapse_test.npz" if args.test else "sparc_collapse.npz"),
             names=np.array(names), rows=np.array(rows), params=np.array(params), chi2=np.array(chi),
             tau=np.array(taus), hern=np.array(herns), vel=np.array(vel), sigma_m=np.array(SIGMA_M), t_gyr=T_GYR)
    print("done")
