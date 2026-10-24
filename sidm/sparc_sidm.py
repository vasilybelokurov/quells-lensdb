"""SIDM fits to SPARC rotation curves, and the inner mass profiles they predict for lensing.

Galaxies: as sparc_lenses.sparc_profiles (quality 1-2, inclination >= 30 deg, no S0s; Hubble-flow distances
rescaled to H0 = 70). Baryons (spherical, as elsewhere in this project):
    M_b(r) = r [V_gas |V_gas| + Y_d V_disk^2 + Y_b V_bul^2] / G,  Y_d = 0.5, Y_b = 0.7 ([3.6]; Lelli+16),
inside the first measured radius r0 the shape of a spherical exponential disc with the catalogued R_disk,
beyond the last point constant.
Dark matter: NFW (M200, c) times (1 - M_b/M200), or isothermal-Jeans SIDM (jeans.py) with these baryons and
X = (sigma/m) x T, T = 10 Gyr (as Kaplinghat et al. 2019 for SPARC). CDM: (log M200, log c) fitted to V_obs; SIDM: the core
(log rho0, log sigma0) fitted (jeans.forward: isothermal core, r1 from the rate condition, NFW matched at r1),
both with a log-normal prior on c around Diemer+19 (0.11 dex) for the implied NFW halo, as Kaplinghat et al.
(2019) did. params per sigma/m: (p0, p1, M200, c, r1).
Lensing mass: the measured M(<r) = V_obs^2 r / G outside r0 (as sparc_lenses), the fitted model inside r0,
scaled to join the data at r0. SIDM fits that do not exist (no solution, e.g. beyond the fold) are flagged.

    python sidm/sparc_sidm.py [--test]      -> data/sidm/sparc_sidm.npz
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jeans as J
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
from lenscore import cosmology as cosmo  # noqa: E402
from lenscore import lensing  # noqa: E402
from lenscore import profiles as prof  # noqa: E402
from lenscore.cosmology import rho_crit  # noqa: E402

OUT = paths.ROOT / "data" / "sidm"
Y_D, Y_B, T_GYR = 0.5, 0.7, 10.0
SIGMA_M = [0.0, 0.5, 1.0, 3.0, 10.0]
C_SCATTER = 0.11


def galaxies():
    """SPARC galaxies used in sparc_lenses.sparc_profiles, with their rotation-curve data."""
    out = []
    for line in open(paths.RAW / "dwarfs" / "SPARC_Lelli2016c.mrt"):
        w = line.split()
        if len(w) < 18 or not w[17].isdigit() or not w[1].isdigit():
            continue
        name, htype, fdist, inc, vflat, q = w[0], int(w[1]), int(w[4]), float(w[5]), float(w[15]), int(w[17])
        if q > 2 or inc < 30 or htype == 0:
            continue
        rc = np.loadtxt(paths.RAW / "dwarfs" / "rotmod" / f"{name}_rotmod.dat")
        f = 73.0 / 70.0 if fdist == 1 else 1.0                 # radii and baryon V^2 scale with distance
        out.append(dict(name=name, r=rc[:, 0] * f, v=rc[:, 1], ev=np.maximum(rc[:, 2], 1.0),
                        vb2=f * (rc[:, 3] * np.abs(rc[:, 3]) + Y_D * rc[:, 4] ** 2 + Y_B * rc[:, 5] ** 2),
                        rd=float(w[11]) * f, vtag=vflat if vflat > 0 else rc[:, 1].max()))
    return out


def baryons(g):
    """M_b(r) callable and total baryon mass."""
    r, mb = g["r"], np.maximum.accumulate(np.maximum(g["vb2"] * g["r"] / cosmo.G, 0))
    rd = g["rd"] if g["rd"] > 0 else r[0]
    shape = lambda x: 1 - (1 + x / rd) * np.exp(-x / rd)

    def m_b(x):
        x = np.asarray(x, float)
        inner = mb[0] * shape(x) / shape(r[0])
        return np.where(x < r[0], inner, np.interp(x, r, mb, right=mb[-1]))
    return m_b, mb[-1]


def nfw_m200_c(rho_s, rs, z=0.0):
    """M200 [Msun] and c of an NFW halo (rho_s, r_s)."""
    mu = lambda x: np.log1p(x) - x / (1 + x)
    rc200 = 200 * rho_crit(z)
    lo, hi = 1e-2, 1e3
    for _ in range(60):                         # mean density 3 rho_s mu(x) / x^3 falls with x
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if 3 * rho_s * mu(mid) / mid**3 > rc200 else (lo, mid)
    c = np.sqrt(lo * hi)
    return 4 * np.pi * rho_s * rs**3 * mu(c), c


def dm_profile(p, X, m_b, mbt):
    """M_DM(<r) callable and (M200, c): NFW (p = log M200, log c) for X = 0, else SIDM from its core
    (p = log rho0, log sigma0) through jeans.forward. (None, None) if no SIDM halo exists for p."""
    if X == 0:
        m_cdm, _, _ = J.halo_from(10 ** p[0], 10 ** p[1], 0.0, m_b, mbt)
        return m_cdm, (10 ** p[0], 10 ** p[1])
    m_dm, nfw = J.forward(10 ** p[0], 10 ** p[1], X, m_b)
    if m_dm is None:
        return None, None
    return m_dm, nfw_m200_c(nfw[1], nfw[2])


def fit(g):
    """Best parameters for each sigma/m: (p, chi2, ok, M200, c, r1)."""
    m_b, mbt = baryons(g)
    r, v, ev, vb2 = g["r"], g["v"], g["ev"], g["vb2"]

    def chi2(p, X):
        mdm, mc = dm_profile(p, X, m_b, mbt)
        if mdm is None or not (1e8 < mc[0] < 1e15 and 0.3 < mc[1] < 100):
            return 1e12
        vm = np.sqrt(np.maximum(vb2 + cosmo.G * mdm(r) / r, 0))
        cdj = np.log10(prof.concentration(mc[0], 0.0))
        return np.sum(((vm - v) / ev) ** 2) + ((np.log10(mc[1]) - cdj) / C_SCATTER) ** 2

    out = []
    for s in SIGMA_M:
        X = s * T_GYR
        if X == 0:
            grid = [(lm, lc) for lm in np.arange(9.5, 13.6, 0.25) for lc in np.arange(0.3, 1.6, 0.15)]
        else:
            grid = [(lr, ls) for lr in np.arange(6.0, 10.6, 0.25) for ls in np.log10(np.geomspace(8, 350, 30))]
        vals = [chi2(q, X) for q in grid]
        p0 = np.array(grid[int(np.argmin(vals))])
        res = minimize(chi2, p0, args=(X,), method="Nelder-Mead",
                       options=dict(xatol=1e-4, fatol=1e-3, maxiter=600))
        ok = res.fun < 1e11
        mdm, mc = dm_profile(res.x, X, m_b, mbt) if ok else (None, (np.nan, np.nan))
        r1 = J.forward(10 ** res.x[0], 10 ** res.x[1], X, m_b)[1][0] if ok and X > 0 else 0.0
        out.append((res.x, res.fun, ok, mc[0], mc[1], r1))
    return out


def lensing_rows(g, fits):
    """Enclosed-mass rows on lensing.R3 per sigma/m: measured mass outside r0, fitted model inside (scaled to
    the measured mass at r0); NaN row if the fit failed."""
    m_b, mbt = baryons(g)
    r, v = g["r"], g["v"]
    vr = np.interp(lensing.R3, r, v)
    r200 = v[-1] / (10 * 70.0 / 1e3)
    m_meas = np.maximum.accumulate(vr**2 * np.minimum(lensing.R3, r200) / cosmo.G)
    m0_meas = v[0] ** 2 * r[0] / cosmo.G
    rows = []
    for s, f in zip(SIGMA_M, fits):
        mdm = dm_profile(f[0], s * T_GYR, m_b, mbt)[0] if f[2] else None
        if mdm is None:
            rows.append(np.full(lensing.R3.size, np.nan))
            continue
        model = m_b(lensing.R3) + mdm(lensing.R3)
        inner = model * m0_meas / (m_b(r[0]) + mdm(r[0]))
        rows.append(np.where(lensing.R3 < r[0], inner, m_meas))
    return np.array(rows)


def task(g):
    try:
        fits = fit(g)
        return g["name"], fits, lensing_rows(g, fits)
    except Exception as e:                            # report and keep going
        return g["name"], repr(e), None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="three galaxies only")
    args = ap.parse_args()
    gals = galaxies()
    if args.test:
        gals = [g for g in gals if g["name"] in ("DDO154", "NGC3198", "NGC2841")]
    print(f"{len(gals)} SPARC galaxies; sigma/m = {SIGMA_M} cm^2/g, T = {T_GYR} Gyr", flush=True)
    with ProcessPoolExecutor(13) as ex:
        res = list(ex.map(task, gals, chunksize=1))
    names, rows, params, chi, oks, vel = [], [], [], [], [], []
    for (name, fits, rw), g in zip(res, gals):
        if rw is None:
            print(f"  {name}: failed ({fits})")
            continue
        names.append(name)
        rows.append(rw)
        params.append([[f[0][0], f[0][1], f[3], f[4], f[5]] for f in fits])
        chi.append([f[1] for f in fits])
        oks.append([f[2] for f in fits])
        vel.append(g["vtag"])
        print(f"  {name:12s} V {g['vtag']:5.0f}  n {len(g['r']):3d}  " + "  ".join(
            f"s{s:g}: chi2 {f[1]:6.1f} M {np.log10(f[3]):5.2f} c {f[4]:5.1f} r1 {f[5]:5.1f}" if f[2] else f"s{s:g}: none"
            for s, f in zip(SIGMA_M, fits)), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / ("sparc_sidm_test.npz" if args.test else "sparc_sidm.npz"), names=np.array(names),
             rows=np.array(rows), params=np.array(params), chi2=np.array(chi), ok=np.array(oks),
             vel=np.array(vel), sigma_m=np.array(SIGMA_M), t_gyr=T_GYR)
    print("done")
