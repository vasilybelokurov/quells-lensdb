"""Intrinsic image-separation distribution of lensed quasars for haloes whose dark
matter responds to the central galaxy in different ways.

One galaxy population, alternative dark-matter profiles (stars held fixed):
  NFW        no response
  contracted Cautun et al. 2020 (arXiv:1911.04557) eq. 11
  cored      coreNFW, Read et al. 2016 (arXiv:1508.04143): M_DM(<r) tanh^n(r/r_c),
             r_c = 1.75 R_e, n = 0.5 and 1 (a core-shape experiment at these masses)

Population
  haloes     Tinker+08 mass function, Diemer+19 c(M), M_200c (colossus)
  stars      Moster+13 SHMR (arXiv:1205.5807; M_200c, Chabrier), 0.15 dex scatter
  sizes      van der Wel+14 early-type circularised median R_e (arXiv:1404.2844,
             appendix table), 0.12 dex scatter; Hernquist, a = R_e / 1.8153
             Every central is given an early-type galaxy: absolute numbers at low
             mass are overestimated, scenario ratios are the robust output.
  baryons    total halo mass M_200c = DM + stars + diffuse gas; DM = (1 - f_b) of the
             DMO profile (before any response), gas = f_b M - M_* with the DMO shape.

Lensing (spherical): projected mass on a log grid, radial critical curve
lambda_r = 1 - 2 kappa + kbar = 0, cross-section pi beta_r^2, image separations from
equal-area samples of the source position. Source at z_s = 2, no magnification bias.
Output: optical depth per log separation, absolute (not renormalised).
"""
import matplotlib.pyplot as plt
import numpy as np
from colossus.halo import concentration, mass_so
from colossus.lss import mass_function

import paths
from lenscore import cosmology as cosmo
from lenscore import lensing
from lenscore import profiles as prof
from lenscore.cosmology import FB, H
from lenscore.population import SEP_BINS, ZL

# van der Wel+14 appendix: circularised log10(R_e/kpc) medians for early types.
# Missing low-mass entries filled with the neighbouring mass bin.
VDW_Z = np.array([0.25, 0.75, 1.25, 1.75])
VDW_LOGM = np.array([9.25, 9.75, 10.25, 10.75, 11.25])
VDW_LOGR = np.array([[0.17, 0.17, 0.24, 0.57, 0.72],
                     [0.12, 0.10, 0.13, 0.36, 0.74],
                     [0.06, 0.06, -0.01, 0.16, 0.49],
                     [0.12, 0.12, -0.05, 0.08, 0.37]])
VDW_SLOPE_HI = 0.75                 # power-law slope used above 10^11.25


def re_vdw14(mstar, z):
    lm = np.log10(mstar)
    zc = np.clip(z, VDW_Z[0], VDW_Z[-1])
    row = np.array([np.interp(zc, VDW_Z, VDW_LOGR[:, j]) for j in range(VDW_LOGM.size)])
    lr = np.interp(np.clip(lm, VDW_LOGM[0], VDW_LOGM[-1]), VDW_LOGM, row)
    lr = lr + VDW_SLOPE_HI * np.clip(lm - VDW_LOGM[-1], 0, None)
    return 10 ** lr



def enclosed_profiles(m200, c, r200, mstar, re, model):
    """Total 3D enclosed mass on lensing.R3 (kpc) for each halo row; arrays of shape (N,)."""
    r = lensing.R3[None, :]
    inside = r <= r200[:, None]
    m0 = m200[:, None] * np.where(inside, prof.nfw_m(r, (r200 / c)[:, None], c[:, None]), 1.0)
    mst = mstar[:, None] * prof.hernquist_m(r, (re / 1.8153)[:, None])
    mgas = (FB * m200 - mstar)[:, None] * m0 / m200[:, None]
    if model == "NFW":
        mdm = (1 - FB) * m0
    elif model == "contracted":
        eta = (mst + mgas) / (FB * m0)
        mdm = (1 - FB) * m0 * (0.45 + 0.38 * (eta + 1.16) ** 0.53)
    elif model.startswith("cored"):
        n = float(model.split("n=")[1])
        mdm = (1 - FB) * m0 * np.tanh(r / (1.75 * re)[:, None]) ** n
    else:
        raise ValueError(model)
    return mdm + mst + mgas



# ---------- population integral ----------
MODELS = ["NFW", "contracted", "cored n=0.5", "cored n=1"]
LOGM = np.arange(10.0, 14.01, 0.1)
DLNM = 0.1 * np.log(10)
GH = [(-np.sqrt(3), 1 / 6), (0.0, 2 / 3), (np.sqrt(3), 1 / 6)]   # 3-point Gauss-Hermite
SIG_LOGMS, SIG_LOGRE = 0.15, 0.12


def run():
    dz = ZL[1] - ZL[0]
    hist = {m: np.zeros(SEP_BINS.size - 1) for m in MODELS}
    lens_m = {m: np.zeros(LOGM.size) for m in MODELS}
    for zl in ZL:
        scrit, dl = cosmo.sigma_crit(zl)
        m200 = 10**LOGM
        dndlnm = mass_function.massFunction(m200 * H, zl, mdef="200c", model="tinker08",
                                            q_out="dndlnM") * H**3           # Mpc^-3
        c = concentration.concentration(m200 * H, "200c", zl, model="diemer19")
        r200 = mass_so.M_to_R(m200 * H, zl, "200c") / H                        # kpc
        dvdz = cosmo.dvdz(zl)                                                   # Mpc^3/sr
        ms0 = prof.mstar_moster13(m200, zl)
        rows = []
        for xs, ws in GH:
            ms = ms0 * 10 ** (SIG_LOGMS * xs)
            re_med = re_vdw14(ms, zl)
            for xr, wr in GH:
                rows.append((ms, re_med * 10 ** (SIG_LOGRE * xr), ws * wr))
        for model in MODELS:
            for ms, re, w in rows:
                m3d = enclosed_profiles(m200, c, r200, ms, re, model)
                beta_r, seps, _, _ = lensing.images(m3d, dl, scrit)
                area = np.pi * beta_r**2
                wt = dndlnm * DLNM * dvdz * dz * area * w                       # dtau
                lens_m[model] += wt
                for k in range(seps.shape[1]):
                    ok = np.isfinite(seps[:, k])
                    hist[model] += np.histogram(seps[ok, k], SEP_BINS,
                                                weights=wt[ok] / seps.shape[1])[0]
    return hist, lens_m


# ---------- checks ----------
def check_sis():
    """SIS of sigma = 250 km/s at z_l = 0.5: compare theta_E, area, separation."""
    zl, s = 0.5, 250.0
    scrit, dl = cosmo.sigma_crit(zl)
    m3d = (2 * s**2 * lensing.R3 / cosmo.G)[None, :]
    beta_r, seps, _, _ = lensing.images(m3d, dl, scrit)
    area = np.pi * beta_r**2
    dcl, dcs = cosmo.dist_mpc(zl), cosmo.dist_mpc(cosmo.ZS)
    te = 4 * np.pi * (s / cosmo.C_KMS) ** 2 * (dcs - dcl) / dcs / cosmo.ARCSEC
    print(f"SIS check: theta_E {te:.3f}\"; separations {np.nanmin(seps):.3f}-"
          f"{np.nanmax(seps):.3f}\" (expect {2 * te:.3f}); area/(pi theta_E^2) = "
          f"{area[0] / (np.pi * (te * cosmo.ARCSEC) ** 2):.3f}")


def check_slacs():
    """Median SLACS-like lens: z_l = 0.2, z_s = 0.6, M_* = 10^11.3."""
    zl, zs = 0.2, 0.6
    logm = np.linspace(12, 14.5, 251)
    ms = prof.mstar_moster13(10**logm, zl)
    m200 = np.array([10 ** np.interp(11.3, np.log10(ms), logm)])
    c = concentration.concentration(m200 * H, "200c", zl, model="diemer19")
    r200 = mass_so.M_to_R(m200 * H, zl, "200c") / H
    mst = np.array([10**11.3])
    re = re_vdw14(mst, zl)
    scrit, dl = cosmo.sigma_crit(zl, zs)
    print(f"SLACS-like: log M200 = {np.log10(m200[0]):.2f}, c = {c[0]:.1f}, "
          f"R_e = {re[0]:.1f} kpc ({re[0] / dl / cosmo.ARCSEC:.2f}\")")
    for model in MODELS:
        m3d = enclosed_profiles(m200, c, r200, mst, re, model)
        m2d = lensing.project(m3d)
        kbar = m2d[0] / (np.pi * lensing.R2**2 * scrit)
        te = np.interp(-1.0, -kbar, lensing.R2)                      # kbar decreasing outward
        # 3D total-density slope averaged over 0.5-1 R_e
        lr = np.log(lensing.R3)
        rho_slope = np.gradient(np.log(np.gradient(m3d[0], lensing.R3) / lensing.R3**2), lr)
        sel = (lensing.R3 > 0.5 * re[0]) & (lensing.R3 < re[0])
        print(f"  {model:12s} theta_E = {te / dl / cosmo.ARCSEC:.2f}\" "
              f"({te:.1f} kpc), density slope 0.5-1 R_e = {rho_slope[sel].mean():.2f}")


if __name__ == "__main__":
    check_sis()
    check_slacs()
    hist, lens_m = run()
    centres = np.sqrt(SEP_BINS[1:] * SEP_BINS[:-1])
    dlog = np.diff(np.log10(SEP_BINS))
    tau = {m: h.sum() for m, h in hist.items()}
    print("\noptical depth for a source at z_s = 2, and fraction below 0.5\", 1\", 2\":")
    for m, h in hist.items():
        cdf = np.cumsum(h) / h.sum()
        f = [np.interp(np.log10(x), np.log10(SEP_BINS[1:]), cdf) for x in (0.5, 1, 2)]
        lm = lens_m[m] / lens_m[m].sum()
        med = np.interp(0.5, np.cumsum(lm), LOGM)
        print(f"  {m:12s} tau = {tau[m]:.2e}  ratio to NFW = {tau[m] / tau['NFW']:.2f}  "
              f"fractions {f[0]:.2f} {f[1]:.2f} {f[2]:.2f}  median lens log M200 = {med:.2f}")

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.05))
    cols = {"NFW": "k", "contracted": "C3", "cored n=0.5": "C0", "cored n=1": "C9"}
    for m, h in hist.items():
        ax.step(centres, h / dlog, where="mid", color=cols[m], lw=1.6, label=m)
        with np.errstate(divide="ignore", invalid="ignore"):
            axr.step(centres, h / hist["NFW"], where="mid", color=cols[m], lw=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, None)
    ax.set_ylabel(r"$d\tau/d\log_{10}\Delta\theta$  (source at $z_s=2$)")
    ax.legend(frameon=False, fontsize=9)
    axr.set_yscale("log")
    axr.set_ylim(0.05, 20)
    axr.axhline(1, color="0.6", lw=0.8)
    axr.set_ylabel("ratio to NFW")
    axr.set_xlabel("image separation [arcsec]")
    axr.set_xlim(0.05, 10)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.1)
    fig.savefig(paths.PLOTS / "separations_contraction_cores.png", dpi=150)
