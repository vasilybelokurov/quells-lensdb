"""Sub-arcsecond lens statistics with SIDM haloes against the CDM range allowed by SLACS.

Population (as lens_small_separation): Tinker+08 haloes (M200c 10^11-10^13.7), each with a Hernquist
early-type galaxy, M_* = alpha x Moster+13, R_e from Shen+03 at the Chabrier mass, Diemer+19 NFW halo.
Source at z_s = 2, lens redshifts 0.1-1.9, no magnification bias. Images through lenscore: two outer images
with flux ratio >= 0.1 (QUELLS), 24 equal-area source positions per lens.

Every model uses the alpha that fits the SLACS Einstein radii with the same dark matter (slacs_valley.py,
slacs_sidm.py), so models are compared at equal fit to SLACS:
  cdm     Dutton+07 response nu, alpha on the CDM valley: the range of CDM predictions allowed by SLACS.
  jref    CDM without response built like the Jeans haloes (jeans.halo), at its own SLACS alpha.
  jeans   isothermal-Jeans SIDM (jeans.py; expansion phase only), X = (sigma/m) x cosmic age at z_l. Haloes
          beyond the fold are not lensed; the optical depth the same halo would have without self-interactions
          is booked as 'flagged'.
  pref    the contracted CDM progenitor of the parametric model (parametric.py, tau = 0), at its SLACS alpha.
  param   parametric gravothermal SIDM (Yang 2024; expansion and collapse, frozen at tau = 1.1).
Output: data/sidm/forecast.npz with (separation, delay) histograms of the optical depth per model.

    python sidm/forecast.py [--quick] [--fixed-alpha]     (--fixed-alpha -> data/sidm/forecast_fixed.npz)
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jeans as J  # noqa: E402
import parametric as P  # noqa: E402

import paths  # noqa: E402
from lenscore import cosmology as cosmo  # noqa: E402
from lenscore import lensing  # noqa: E402
from lenscore import profiles as prof  # noqa: E402
from lenscore.population import DT_E, SEP_BINS, SEP_E  # noqa: E402

OUT = paths.ROOT / "data" / "sidm"
QMIN, NBETA, N_R1 = 0.1, 24, 100
MAX_SLACS_FOLD = 0.1          # largest share of SLACS lenses beyond the Jeans fold for a usable alpha match


def models(fixed_alpha=False):
    """(key, kind, alpha, parameter) with SLACS-matched alphas; with fixed_alpha the SIDM models keep the alpha
    of their own CDM limit (self-interactions switched on without re-fitting SLACS), for all sigma/m."""
    import slacs_valley as SV
    v = np.load(OUT / "slacs_valley_cdm.npz")
    out = [(f"cdm_nu{nu:+.2f}", "cdm", SV.valley(row, v["alphas"]), nu) for nu, row in zip(v["nus"], v["med"])]
    for name, kind0, kind in [("slacs_valley_sidm.npz", "jref", "jeans"), ("slacs_valley_param.npz", "pref", "param")]:
        s = np.load(OUT / name)
        if fixed_alpha:
            a0 = round(float(s["alpha_match"][s["sigma_m"] == 0][0]), 3)
            out += [(f"{kind0}" if sm == 0 else f"{kind}_s{sm:g}", kind0 if sm == 0 else kind, a0, sm)
                    for sm in s["sigma_m"]]
            continue
        for sm, a, nf in zip(s["sigma_m"], s["alpha_match"], s["nfold"]):
            # the SLACS match is meaningful only if (almost) all SLACS haloes are modelled at that sigma/m
            nf_at = nf[np.argmin(np.abs(s["alphas"] - a))] if np.isfinite(a) else nf.max()
            if np.isfinite(a) and nf_at <= MAX_SLACS_FOLD * s["q"].shape[2]:
                out.append((f"{kind0}" if sm == 0 else f"{kind}_s{sm:g}", kind0 if sm == 0 else kind, round(float(a), 3), sm))
    return out


def halo_task(args):
    """All models for one (z_l, log M200) cell: per model the counted samples (sep, delay, weight), the flagged
    optical depth, and per Jeans alpha the sigma/m at the fold."""
    zl, lm, dlnm, dz, mods = args
    from colossus.lss import mass_function
    m200 = 10**lm
    n = mass_function.massFunction(m200 * cosmo.H, zl, mdef="200c", model="tinker08", q_out="dndlnM") * cosmo.H**3 * dlnm
    vol = cosmo.dvdz(zl) * dz
    scr, dl = cosmo.sigma_crit(zl)
    tdl = cosmo.dt_distance_m(zl) / (cosmo.C_KMS * 1e3) / cosmo.DAY
    age = cosmo.COSMO.age(zl)
    ms0 = prof.mstar_moster13(m200, zl)
    re = prof.re_shen03(ms0)
    c = prof.concentration(m200, zl)
    rows, keys, branches, flagged = [], [], {}, {}
    for key, kind, a, par in mods:
        ms = a * ms0
        if kind == "cdm":
            m3, rmax = prof.galaxy_halo_m3d(m200, c, zl, ms, re, par)
            rows.append(m3(np.minimum(lensing.R3, rmax)))
        elif kind in ("jref", "jeans"):
            m_cdm, _, m_b = J.halo(m200, zl, ms, re)
            if kind == "jeans":
                if a not in branches:
                    branches[a] = J.branch(m200, zl, ms, re, np.geomspace(0.02, J.r200(m200, zl), N_R1))
                m3 = J.m3d_at_x(m200, zl, ms, re, par * age, branches[a])
                if m3 is None:                      # beyond the fold: book the halo without self-interactions
                    flagged[key] = (m_cdm(lensing.R3) + m_b(lensing.R3))
                    continue
                rows.append(m3(lensing.R3))
            else:
                rows.append(m_cdm(lensing.R3) + m_b(lensing.R3))
        else:
            h = P.Halo(m200, zl, ms, re)
            rows.append(h.m_tot(0.0 if kind == "pref" else h.tau(par, age)))
        keys.append(key)
    all_rows = np.array(rows + list(flagged.values()))
    beta_r, sep, dphi, _ = lensing.images(all_rows, dl, scr, NBETA, QMIN)
    out = {}
    for k, br, s, d in zip(keys, beta_r, sep, dphi):
        ok = np.isfinite(s)
        out[k] = (s[ok], tdl * d[ok], np.full(ok.sum(), n * vol * np.pi * br**2 / NBETA))
    flag_tau = {k: n * vol * np.pi * br**2 * np.isfinite(s).mean()
                for k, br, s in zip(flagged, beta_r[len(keys):], sep[len(keys):])}
    xmax = {a: b[-1, 3] / age for a, b in branches.items() if len(b)}
    return zl, lm, out, flag_tau, xmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarse grid, for testing")
    ap.add_argument("--fixed-alpha", action="store_true",
                    help="SIDM at the alpha of its CDM limit (not re-matched to SLACS) -> forecast_fixed.npz")
    args = ap.parse_args()
    mods = models(args.fixed_alpha)
    if args.quick:
        zg, lg, dz, dlog = np.array([0.4, 0.8, 1.2]), np.array([11.5, 12.0, 12.5, 13.0]), 0.6, 0.5
    else:
        zg, lg, dz, dlog = np.arange(0.1, 1.91, 0.1), np.arange(11.0, 13.71, 0.1), 0.1, 0.1
    tasks = [(z, lm, dlog * np.log(10), dz, mods) for z in zg for lm in lg]
    print(f"{len(tasks)} haloes x {len(mods)} models", flush=True)
    for m in mods:
        print(f"  {m[0]:12s} kind {m[1]:6s} alpha {m[2]:.3f} parameter {m[3]:g}", flush=True)
    t0 = time.time()
    keys = [m[0] for m in mods]
    h2 = {k: np.zeros((SEP_E.size - 1, DT_E.size - 1)) for k in keys}
    h1 = {k: np.zeros(SEP_BINS.size - 1) for k in keys}
    flag = {k: 0.0 for k in keys}
    jalphas = sorted({m[2] for m in mods if m[1] == "jeans"})
    smax = np.full((len(jalphas), zg.size, lg.size), np.nan)
    with ProcessPoolExecutor(max_workers=13) as ex:
        for i, (zl, lm, out, flag_tau, xmax) in enumerate(ex.map(halo_task, tasks, chunksize=1)):
            for k, (s, d, w) in out.items():
                h2[k] += np.histogram2d(s, d, [SEP_E, DT_E], weights=w)[0]
                h1[k] += np.histogram(s, SEP_BINS, weights=w)[0]
            for k, v in flag_tau.items():
                flag[k] += v
            for j, a in enumerate(jalphas):
                smax[j, np.argmin(np.abs(zg - zl)), np.argmin(np.abs(lg - lm))] = xmax.get(a, np.nan)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(tasks)} haloes, {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = ("forecast_quick" if args.quick else "forecast") + ("_fixed" if args.fixed_alpha else "") + ".npz"
    np.savez(OUT / name, keys=np.array(keys), kinds=np.array([m[1] for m in mods]),
             alphas=np.array([m[2] for m in mods]), pars=np.array([m[3] for m in mods]),
             h2=np.array([h2[k] for k in keys]), h1=np.array([h1[k] for k in keys]),
             flag_tau=np.array([flag[k] for k in keys]), sigma_m_fold=smax, jeans_alphas=np.array(jalphas),
             zg=zg, lg=lg)
    print(f"wrote {OUT / name} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
