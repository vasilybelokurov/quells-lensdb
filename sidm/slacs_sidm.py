"""SLACS Einstein radii with SIDM haloes: the stellar-mass normalisation alpha that SLACS requires for each
sigma/m, so that SIDM is compared with CDM at equal fit to SLACS (not at fixed alpha).

Same lenses, stars and M200 as slacs_valley.py; dark matter: isothermal-Jeans SIDM (jeans.py) with the outer
halo without response, X = (sigma/m) x cosmic age at each lens's z_l. sigma/m = 0 is the same halo without
self-interactions (jeans.halo), the check against the CDM valley at nu = 0. Lenses whose halo is beyond the
fold at that X are left out of the median; their number is stored.

With --model param the dark matter is the parametric gravothermal model (parametric.py; core expansion and
collapse, no fold), whose sigma/m = 0 limit is the NFW halo contracted by the stars.

    python sidm/slacs_sidm.py                 -> data/sidm/slacs_valley_sidm.npz, alpha(sigma/m) printed
    python sidm/slacs_sidm.py --model param   -> data/sidm/slacs_valley_param.npz
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jeans as J
import numpy as np
import parametric as P
import slacs_valley as SV

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lenscore import cosmology as cosmo  # noqa: E402
from lenscore import lensing  # noqa: E402

SIGMA_M = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
ALPHAS = SV.ALPHAS
S = SV.sample()


def task(args):
    """log10(theta_E model/obs) for one lens and one alpha, for every sigma/m (NaN beyond the fold)."""
    i, a = args
    zl, zs, m200 = S["zl"][i], S["zs"][i], S["m200"][i]
    ms, re = a * 10 ** S["logms"][i], S["re_v_kpc"][i]
    b = J.branch(m200, zl, ms, re, np.geomspace(0.02, J.r200(m200, zl), 100))
    m_cdm, _, m_b = J.halo(m200, zl, ms, re)
    out = []
    for s in SIGMA_M:
        m3 = (lambda r: m_cdm(r) + m_b(r)) if s == 0 else J.m3d_at_x(m200, zl, ms, re, s * cosmo.COSMO.age(zl), b)
        if m3 is None:
            out.append(np.nan)
            continue
        out.append(np.log10(SV.theta_e(m3(lensing.R3)[None, :], zl, zs)[0] / S["theta_obs"][i]))
    return i, a, out


def task_param(args):
    """As task, with the parametric model (no fold: every lens is modelled)."""
    i, a = args
    zl, zs = S["zl"][i], S["zs"][i]
    h = P.Halo(S["m200"][i], zl, a * 10 ** S["logms"][i], S["re_v_kpc"][i])
    t = cosmo.COSMO.age(zl)
    return i, a, [np.log10(SV.theta_e(h.m_tot(h.tau(s, t))[None, :], zl, zs)[0] / S["theta_obs"][i])
                  for s in SIGMA_M]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["jeans", "param"], default="jeans")
    model = ap.parse_args().model
    n = len(S["zl"])
    q = np.full((len(SIGMA_M), len(ALPHAS), n), np.nan)
    with ProcessPoolExecutor(13) as ex:
        for i, a, out in ex.map(task if model == "jeans" else task_param,
                                [(i, a) for i in range(n) for a in ALPHAS], chunksize=2):
            q[:, list(ALPHAS).index(a), i] = out
    med = np.nanmedian(q, axis=2)
    nfold = np.isnan(q).sum(axis=2)
    alpha_match = np.array([SV.valley(row, ALPHAS) for row in med])
    name = "slacs_valley_sidm.npz" if model == "jeans" else "slacs_valley_param.npz"
    np.savez(SV.OUT / name, sigma_m=SIGMA_M, alphas=ALPHAS, med=med, nfold=nfold,
             alpha_match=alpha_match, q=q)
    cdm0 = SV.valley(SV.cdm_median(S, ALPHAS, 0.0), ALPHAS)
    print(f"CDM, nu = 0 (lenscore galaxy_halo_m3d): alpha = {cdm0:.3f}")
    for s, a, nf in zip(SIGMA_M, alpha_match, nfold):
        k = np.argmin(np.abs(ALPHAS - (a if np.isfinite(a) else ALPHAS[0])))
        print(f"sigma/m = {s:4g}: SLACS-matched alpha = {a:.3f}  (lenses beyond the fold at that alpha: {nf[k]}/{n})")
