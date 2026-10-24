"""Sub-arcsecond lens statistics with SIDM haloes against the CDM range allowed by SLACS.

Population (as lens_small_separation): Tinker+08 haloes (M200c 10^11-10^13.7), each with a Hernquist
early-type galaxy, M_* = alpha x Moster+13, R_e from Shen+03 at the Chabrier mass, Diemer+19 NFW halo.
Source at z_s = 2, lens redshifts 0.1-1.9 (dz = 0.15), non-evolving populations apart from the
Moster/c(M) redshift dependence, no magnification bias. Images through lenscore, two outer images with
flux ratio >= 0.1 (QUELLS), 24 equal-area source positions per lens.

Models
  cdm_valley   Dutton+07 response nu with alpha on the SLACS valley alpha(nu) (slacs_valley.py): the range
               of CDM predictions consistent with SLACS Einstein radii.
  ref          CDM without response, built like the SIDM haloes (sidm/jeans.halo): the X -> 0 limit.
  sidm         isothermal-Jeans SIDM (sidm/jeans.py), outer halo without response, at fixed alpha;
               X = (sigma/m) x cosmic age at z_l (an upper bound on the time since halo formation).
               Haloes beyond the fold of the model at that X are not lensed and their CDM cross-section
               is booked as 'flagged'.
Output: data/sidm/forecast.npz with (separation, delay) histograms of the optical depth per model.

    python sidm/forecast.py [--quick]
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

import paths  # noqa: E402
from lenscore import cosmology as cosmo  # noqa: E402
from lenscore import lensing  # noqa: E402
from lenscore import profiles as prof  # noqa: E402
from lenscore.population import DT_E, SEP_BINS, SEP_E  # noqa: E402

OUT = paths.ROOT / "data" / "sidm"
QMIN, NBETA = 0.1, 24
SIGMA_M = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])           # cm^2/g
N_R1 = 100                                                   # nominal r1 points, 0.02 kpc to r200


def models(valley, alphas_sidm):
    """List of (key, kind, alpha, parameter)."""
    out = [(f"cdm_nu{nu:+.2f}", "cdm", a, nu) for nu, a in valley if np.isfinite(a)]
    for a in alphas_sidm:
        out.append((f"ref_a{a:.2f}", "ref", a, 0.0))
        out += [(f"sidm_a{a:.2f}_s{s:g}", "sidm", a, s) for s in SIGMA_M]
    return out


def halo_task(args):
    """All models for one (z_l, log M200) cell: per model the counted samples (sep, dt, weight) and the
    flagged optical depth."""
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
    rows, keys, flagged, branches = [], [], {}, {}
    for key, kind, a, par in mods:
        if kind == "cdm":
            m3, rmax = prof.galaxy_halo_m3d(m200, c, zl, a * ms0, re, par)
            rows.append(m3(np.minimum(lensing.R3, rmax)))
        elif kind == "ref":
            m_cdm, _, m_b = J.halo(m200, zl, a * ms0, re)
            rows.append(m_cdm(lensing.R3) + m_b(lensing.R3))
        else:
            if a not in branches:
                branches[a] = J.branch(m200, zl, a * ms0, re, np.geomspace(0.02, J.r200(m200, zl), N_R1))
            m3 = J.m3d_at_x(m200, zl, a * ms0, re, par * age, branches[a])
            if m3 is None:
                flagged[key] = True
                continue
            rows.append(m3(lensing.R3))
        keys.append(key)
    beta_r, sep, dphi, _ = lensing.images(np.array(rows), dl, scr, NBETA, QMIN)
    out = {}
    for k, br, s, d in zip(keys, beta_r, sep, dphi):
        ok = np.isfinite(s)
        out[k] = (s[ok], tdl * d[ok], np.full(ok.sum(), n * vol * np.pi * br**2 / NBETA))
    xmax = {a: (b[-1, 3] / age if len(b) else 0.0) for a, b in branches.items()}   # sigma/m at the fold
    return zl, lm, out, flagged, xmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarse grid, for testing")
    args = ap.parse_args()
    v = np.load(OUT / "slacs_valley_cdm.npz")
    import slacs_valley as SV
    valley = [(nu, SV.valley(row, v["alphas"])) for nu, row in zip(v["nus"], v["med"])]
    a0 = np.interp(0.0, [x[0] for x in valley], [x[1] for x in valley])        # valley alpha at nu = 0
    alphas_sidm = [round(a0, 2), 1.5, 1.0]
    zg = np.arange(0.1, 1.91, 0.15) if not args.quick else np.array([0.4, 0.8, 1.2])
    lg = np.arange(11.0, 13.71, 0.15) if not args.quick else np.array([11.5, 12.0, 12.5, 13.0])
    dz, dlnm = 0.15 if not args.quick else 0.6, (0.15 if not args.quick else 0.5) * np.log(10)
    mods = models(valley, alphas_sidm)
    tasks = [(z, lm, dlnm, dz, mods) for z in zg for lm in lg]
    print(f"{len(tasks)} haloes x {len(mods)} models; SIDM alphas {alphas_sidm}; valley alpha(nu=0) = {a0:.3f}", flush=True)
    t0 = time.time()
    h2 = {k: np.zeros((SEP_E.size - 1, DT_E.size - 1)) for k, *_ in mods}
    h1 = {k: np.zeros(SEP_BINS.size - 1) for k, *_ in mods}
    flag_tau = {k: 0.0 for k, *_ in mods}
    smax = np.full((len(alphas_sidm), zg.size, lg.size), np.nan)
    with ProcessPoolExecutor(max_workers=13) as ex:
        for i, (zl, lm, out, flagged, xmax) in enumerate(ex.map(halo_task, tasks, chunksize=1)):
            for k, (s, d, w) in out.items():
                h2[k] += np.histogram2d(s, d, [SEP_E, DT_E], weights=w)[0]
                h1[k] += np.histogram(s, SEP_BINS, weights=w)[0]
            for k in flagged:                       # charge the reference cross-section of that alpha
                a = k.split("_")[1][1:]
                s, d, w = out[f"ref_a{a}"]
                flag_tau[k] += w.sum()
            for j, a in enumerate(alphas_sidm):
                smax[j, np.argmin(np.abs(zg - zl)), np.argmin(np.abs(lg - lm))] = xmax.get(a, np.nan)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(tasks)} haloes, {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "forecast_quick.npz" if args.quick else "forecast.npz"
    np.savez(OUT / name, keys=np.array([m[0] for m in mods]), kinds=np.array([m[1] for m in mods]),
             alphas=np.array([m[2] for m in mods]), pars=np.array([m[3] for m in mods]),
             h2=np.array([h2[m[0]] for m in mods]), h1=np.array([h1[m[0]] for m in mods]),
             flag_tau=np.array([flag_tau[m[0]] for m in mods]), sigma_m_fold=smax,
             alphas_sidm=np.array(alphas_sidm), zg=zg, lg=lg, sigma_m=SIGMA_M)
    print(f"wrote {OUT / name} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
