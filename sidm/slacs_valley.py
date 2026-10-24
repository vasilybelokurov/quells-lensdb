"""SLACS Einstein radii as a constraint on the galaxy + halo model: the alpha_SPS - nu degeneracy valley.

Per lens (Auger et al. 2009, 70 lenses with V-band R_e, M_* and R_Ein): z_l, z_s, M_*(Chabrier), R_e(V),
theta_E. Model: Hernquist stars with M_* = alpha x M_*(Chabrier), NFW halo with M200 from HSC weak lensing
with SLACS selection (Sonnenfeld et al. 2018, MNRAS 481, 164, as used by Shajib et al. 2021):
    log M200 = 13.03 + 1.42 (log M_*,Chab - 11.3),
Diemer+19 concentration, and either the Dutton+07 response nu (CDM) or isothermal-Jeans SIDM (outer halo
without response). Statistic: median over lenses of log10(theta_E model / theta_E observed).
The CDM valley is where the median is 0 (same calculation as lens_small_separation fig03).

    python sidm/slacs_valley.py     -> data/sidm/slacs_valley_cdm.npz, valley alpha(nu) printed
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import obs_lenses as O  # noqa: E402
import paths  # noqa: E402
from lenscore import cosmology as cosmo  # noqa: E402
from lenscore import lensing  # noqa: E402
from lenscore import profiles as prof  # noqa: E402

OUT = paths.ROOT / "data" / "sidm"
ALPHAS = np.linspace(0.8, 3.0, 12)
NUS = np.linspace(-0.5, 1.0, 11)


def sample():
    s = O.slacs()
    ok = np.isfinite(s["re_v_kpc"]) & np.isfinite(s["logms"]) & np.isfinite(s["rein_kpc"]) & np.isfinite(s["zs"])
    out = {k: v[ok] for k, v in s.items()}
    out["m200"] = 10 ** (13.03 + 1.42 * (out["logms"] - 11.3))
    out["theta_obs"] = np.array([rk / cosmo.sigma_crit(zl, zs)[1] for rk, zl, zs in
                                 zip(out["rein_kpc"], out["zl"], out["zs"])])          # rad
    return out


def theta_e(rows, zl, zs):
    scr, dl = cosmo.sigma_crit(zl, zs)
    return lensing.einstein_radius(rows, dl, scr)


def cdm_median(s, alphas, nu):
    """Median log10(theta_E model / obs) for each alpha (CDM with response nu)."""
    q = np.zeros((len(s["zl"]), len(alphas)))
    for i in range(len(s["zl"])):
        zl, m200 = s["zl"][i], s["m200"][i]
        c = prof.concentration(m200, zl)
        rows = []
        for a in alphas:
            m3, rmax = prof.galaxy_halo_m3d(m200, c, zl, a * 10 ** s["logms"][i], s["re_v_kpc"][i], nu)
            rows.append(m3(np.minimum(lensing.R3, rmax)))
        q[i] = np.log10(theta_e(np.array(rows), zl, s["zs"][i]) / s["theta_obs"][i])
    return np.median(q, axis=0)


def valley(med, alphas):
    """alpha where the median log ratio crosses 0 (NaN if it does not)."""
    if not (med.min() < 0 < med.max()):
        return np.nan
    return np.interp(0, med, alphas)


if __name__ == "__main__":
    s = sample()
    print(f"SLACS lenses used: {len(s['zl'])}")
    med = np.array([cdm_median(s, ALPHAS, nu) for nu in NUS])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "slacs_valley_cdm.npz", alphas=ALPHAS, nus=NUS, med=med)
    for nu, row in zip(NUS, med):
        print(f"  nu = {nu:+.2f}: valley alpha = {valley(row, ALPHAS):.3f}")
    ref = Path.home() / "Work/Code/lens_small_separation/data/derived/fig03_slacs_alpha_nu.npz"
    if ref.exists():                      # cross-check against lens_small_separation (local, private)
        r = np.load(ref)
        if np.allclose(r["alphas"], ALPHAS) and np.allclose(r["nus"], NUS):
            print(f"cross-check vs lens_small_separation fig03: max |difference| of the median = "
                  f"{np.abs(r['med'] - med).max():.4f} dex")
