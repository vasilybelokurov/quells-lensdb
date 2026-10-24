"""A volume-weighted galaxy population with measured inner mass profiles, from
MaNGA DynPop (successor of ATLAS3D): Jeans (JAM) models of ~10k MaNGA galaxies of
all morphologies (Zhu et al. 2023, MNRAS 522, 6326; catalogue v2 with Zhu et al. 2025,
ApJS 280, 55; zenodo 17518315).

Per galaxy (JAM quality >= 1, MaNGA main sample, one entry per galaxy):
  stars  the r-band MGE (cosmological dimming corrected by (1+z)^3), each Gaussian
         circularised (sigma = sigma' sqrt(q')) and deprojected to a spherical
         Gaussian, times the JAMcyl+gNFW stellar M/L:
             M_*(<r) = (M/L) sum_j L_j [erf(x/sqrt2) - sqrt(2/pi) x exp(-x^2/2)],
             x = r / sigma_j,  L_j = 2 pi Sigma_j sigma'_j^2 q'_j
  DM     the fitted gNFW halo, rho = rho_s (r/r_s)^g (1/2 + r/2r_s)^(-g-3)
         (rho_s as defined in the catalogue), used only inside the radius covered by
         the kinematic data, r_max (median 1.6 R_e). Beyond r_max the gNFW fit is an
         unconstrained extrapolation (implied halo masses scatter by +-1.2 dex about
         abundance matching), so there we add only the shell mass of an NFW halo with
         the Moster+13 halo mass for the galaxy's M_* and the Diemer & Joyce (2019)
         concentration, truncated at its r_200 ("outer='am'"; outer='gnfw' keeps the
         raw extrapolation, truncated at the gNFW r_200).
Abundance
  n_i = C * esweight_i * f_complete(M_*): esweight from drpall DR17 (Wake et al. 2017);
  C fixed by matching the weighted stellar mass function of the whole main sample
  to GAMA (Baldry et al. 2012) at 10^10-10^11 Msun; f_complete corrects for the JAM
  quality cut in 0.25-dex stellar-mass bins.
Stellar masses for weighting: NSA elliptical-Petrosian masses (h = 1) -> h = 0.7
(linear in drpall, log10 in the DynPop catalogue).

Run directly for the checks: plots/manga_checks.png
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.special import erf

import paths
from lenscore import cosmology as cosmo
from lenscore import lensing
from lenscore import profiles as prof

RAW = paths.RAW / "manga"
MGE_DIR = RAW / "mge" / "SDSSDR17_MaNGA_MGE" / "MGE"
ARCSEC = np.pi / 180 / 3600
RHO_CRIT0 = 2.775e11 * 0.7**2 / 1e9            # Msun / kpc^3, H0 = 70
QMIN_JAM = 1
MS_BINS = np.arange(8.5, 12.01, 0.25)


def baldry12_dex(logm):
    """GAMA stellar mass function per dex (Baldry et al. 2012, H0 = 70, Chabrier)."""
    m = 10 ** (logm - 10.66)
    return np.log(10) * np.exp(-m) * (3.96e-3 * m ** (1 - 0.35) + 0.79e-3 * m ** (1 - 1.47))


def load():
    jam = fits.open(RAW / "SDSSDR17_MaNGA_JAM_v2.fits")
    g = jam[1].data
    gn = jam[8].data                                          # JAMcyl + gNFW
    drp = fits.open(RAW / "drpall-v3_1_1.fits")[1].data
    main = drp["esweight"] > 0
    # one entry per galaxy in the main sample
    _, first = np.unique(drp["mangaid"][main], return_index=True)
    dm = drp[main][first]
    w_of = dict(zip(dm["mangaid"], dm["esweight"]))
    logms_all = np.log10(np.maximum(dm["nsa_elpetro_mass"], 1)) - 2 * np.log10(0.7)

    # normalisation: weighted SMF of the whole main sample vs GAMA
    h, e = np.histogram(logms_all, MS_BINS, weights=dm["esweight"])
    c = 0.5 * (e[1:] + e[:-1])
    sel = (c > 10.0) & (c < 11.0)
    norm = np.median(baldry12_dex(c[sel]) / (h[sel] / np.diff(e)[sel]))

    # DynPop galaxies: quality cut, main sample, unique
    ok = (g["Qual"] >= QMIN_JAM) & np.isin(g["mangaid"], dm["mangaid"])
    idx = np.nonzero(ok)[0]
    _, firstj = np.unique(g["mangaid"][idx], return_index=True)
    idx = idx[firstj]
    w = np.array([w_of[m] for m in g["mangaid"][idx]])
    logms = g["nsa_elpetro_mass"][idx] - 2 * np.log10(0.7)   # DynPop stores log10 (h=1); drpall linear

    # completeness of the quality cut per stellar-mass bin
    hall, _ = np.histogram(logms_all, MS_BINS, weights=dm["esweight"])
    hjam, _ = np.histogram(logms, MS_BINS, weights=w)
    fcomp = np.where(hjam > 0, hall / np.maximum(hjam, 1e-30), 0)
    k = np.clip(np.digitize(logms, MS_BINS) - 1, 0, MS_BINS.size - 2)
    n = norm * w * fcomp[k]
    return dict(g=g[idx], gn=gn[idx], n=n, logms=logms, norm=norm, fcomp=fcomp,
                smf_all=(c, h / np.diff(e) * norm), n_all=len(dm))


def read_mge(plateifu, z):
    m = np.loadtxt(MGE_DIR / f"{plateifu}.dat", skiprows=3, ndmin=2)
    m[:, 0] *= (1 + z) ** 3
    return m                                                  # L/pc^2, arcsec, q'


def profiles(pop, outer="am"):
    """Enclosed total, stellar and DM mass on lensing.R3 (kpc) for every galaxy."""
    g, gn = pop["g"], pop["gn"]
    r = lensing.R3
    out_t, out_s, ltot = [], [], []
    rfine = np.logspace(-4, 4, 4000)
    if outer == "am":
        from colossus.halo import concentration, mass_so
        lm_grid = np.linspace(10, 15.5, 600)
        lms_grid = np.log10(prof.mstar_moster13(10**lm_grid, 0.05))
        m200 = 10 ** np.interp(pop["logms"], lms_grid, lm_grid)
        c200 = concentration.concentration(m200 * cosmo.H, "200c", 0.05, model="diemer19")
        r200am = mass_so.M_to_R(m200 * cosmo.H, 0.05, "200c") / cosmo.H
    for i in range(len(g)):
        da_pc = g["DA"][i] * 1e6
        mge = read_mge(g["plateifu"][i], g["z"][i])
        sig_pc = mge[:, 1] * ARCSEC * da_pc
        lum = 2 * np.pi * mge[:, 0] * sig_pc**2 * mge[:, 2]
        sc = sig_pc * np.sqrt(mge[:, 2]) / 1e3                # kpc, circularised
        x = r[:, None] / sc[None, :]
        frac = erf(x / np.sqrt(2)) - np.sqrt(2 / np.pi) * x * np.exp(-x**2 / 2)
        ms = 10 ** gn["log_ML_stellar"][i] * (frac * lum[None, :]).sum(axis=1)
        rho_s, rs, gam = 10 ** gn["log_rho_s"][i], gn["rs"][i], gn["gamma_gNFW"][i]
        rho = rho_s * (rfine / rs) ** gam * (0.5 + 0.5 * rfine / rs) ** (-gam - 3)
        mdm_f = np.concatenate([[0], np.cumsum(0.5 * (4 * np.pi * rfine[1:]**2 * rho[1:]
                                                       + 4 * np.pi * rfine[:-1]**2 * rho[:-1])
                                                * np.diff(rfine))])
        mean = mdm_f / (4 / 3 * np.pi * rfine**3)
        r200 = rfine[np.nonzero(mean > 200 * RHO_CRIT0)[0][-1]] if np.any(mean > 200 * RHO_CRIT0) else rfine[-1]
        mdm = np.interp(np.minimum(r, r200), rfine, mdm_f)
        if outer == "am":
            rmax = g["rmax_arcsec"][i] * ARCSEC * g["DA"][i] * 1e3          # kpc
            mtot = ms + mdm
            mt_rmax = np.interp(rmax, r, mtot)
            rs_am = r200am[i] / c200[i]
            nfw = lambda x: m200[i] * prof.nfw_m(np.minimum(x, r200am[i]), rs_am, c200[i])
            out_t.append(np.where(r <= rmax, mtot, mt_rmax + (ms - np.interp(rmax, r, ms))
                                  + nfw(r) - nfw(rmax)))
            out_s.append(ms)
            ltot.append(lum.sum())
            continue
        out_t.append(ms + mdm)
        out_s.append(ms)
        ltot.append(lum.sum())
    return np.array(out_t), np.array(out_s), np.array(ltot)


if __name__ == "__main__":
    pop = load()
    print(f"main sample: {pop['n_all']} galaxies; DynPop with Qual >= {QMIN_JAM}: {len(pop['g'])}; "
          f"normalisation C = {pop['norm']:.3e} Mpc^-3 per unit esweight")
    mt, ms, lt = profiles(pop)
    g = pop["g"]
    dl = np.log10(lt) - g["Lum_tot_MGE"]
    print(f"check: log sum(MGE L) - Lum_tot_MGE (after (1+z)^3): median {np.median(dl):+.3f}, "
          f"16-84% {np.percentile(dl, 16):+.3f} {np.percentile(dl, 84):+.3f}  "
          f"(expected ~ 3 log(1+z) = {np.median(3 * np.log10(1 + g['z'])):+.3f})")
    re_kpc = g["Re_arcsec_MGE"] * ARCSEC * g["DA"] * 1e3
    m_re = np.array([np.interp(x, lensing.R3, row) for x, row in zip(re_kpc, mt)])
    dm_re = np.log10(m_re) - pop["gn"]["log_Mt_Re"]
    print(f"check: log M_tot(<R_e) ours - catalogue (sphere of R_e): median {np.median(dm_re):+.3f}, "
          f"16-84% {np.percentile(dm_re, 16):+.3f} {np.percentile(dm_re, 84):+.3f}")
    mt_raw, _, _ = profiles(pop, outer="gnfw")
    np.savez_compressed(RAW / "manga_profiles.npz", mt=mt, mt_raw=mt_raw, ms=ms, n=pop["n"], logms=pop["logms"],
                        sigma=g["Sigma_Re"], sersic_n=g["nsa_sersic_n"], re_kpc=re_kpc,
                        lambda_re=g["Lambda_Re"], eps=g["Eps_MGE"], plateifu=g["plateifu"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    c, phi = pop["smf_all"]
    a1.plot(c, phi, "o", color="k", label="MaNGA main sample, esweight x C")
    lm = np.linspace(8.5, 12, 100)
    a1.plot(lm, baldry12_dex(lm), color="C3", label="GAMA (Baldry+12)")
    hj, _ = np.histogram(pop["logms"], MS_BINS, weights=pop["n"])
    a1.step(c, hj / np.diff(MS_BINS), where="mid", color="C0", label=f"DynPop Qual>={QMIN_JAM}, completeness-corrected")
    a1.set_yscale("log")
    a1.set_xlabel(r"$\log M_*$ [M$_\odot$]")
    a1.set_ylabel(r"$dn/d\log M_*$ [Mpc$^{-3}$ dex$^{-1}$]")
    a1.legend(frameon=False, fontsize=8)
    sig = g["Sigma_Re"]
    sb = np.arange(1.4, 2.65, 0.05)
    early = g["nsa_sersic_n"] >= 2.5
    for sel, col, lab in [(np.ones(len(sig), bool), "k", "all"), (early, "C3", r"Sersic $n\geq2.5$")]:
        h, _ = np.histogram(np.log10(sig[sel]), sb, weights=pop["n"][sel])
        a2.step(10 ** (0.5 * (sb[1:] + sb[:-1])), h / np.diff(sb), where="mid", color=col, label=f"MaNGA, {lab}")
    import compare_separations as CS
    import sparc_lenses as S
    ss = np.logspace(1.4, 2.65, 100)
    a2.plot(ss, S.schechter_dlnv(ss, *CS.CHOI) * np.log(10), color="C3", ls="--", label="Choi+07 E/S0")
    a2.plot(ss, S.schechter_dlnv(ss, *CS.BERNARDI) * np.log(10), color="0.4", ls="--", label="Bernardi+10 all")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_ylim(1e-5, 0.1)
    a2.set_xlabel(r"$\sigma_e$ [km s$^{-1}$]")
    a2.set_ylabel(r"$dn/d\log\sigma$ [Mpc$^{-3}$ dex$^{-1}$]")
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.PLOTS / "manga_checks.png", dpi=150)
