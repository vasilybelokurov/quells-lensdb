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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from colossus.cosmology import cosmology
from colossus.halo import concentration, mass_so
from colossus.lss import mass_function

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

COSMO = cosmology.setCosmology("quells", dict(flat=True, H0=70.0, Om0=0.3, Ob0=0.048,
                                              sigma8=0.81, ns=0.96))
H = 0.7
FB = 0.048 / 0.3
G = 4.30091e-6                      # kpc (km/s)^2 / Msun
C_KMS = 299792.458
ARCSEC = np.pi / 180 / 3600
ZS = 2.0

# ---------- galaxy recipes ----------
MOSTER = dict(M10=11.590, M11=1.195, N10=0.0351, N11=-0.0247,
              b10=1.376, b11=-0.826, g10=0.608, g11=0.329)


def mstar_moster13(m200, z):
    p, a = MOSTER, z / (1 + z)
    m1 = 10 ** (p["M10"] + p["M11"] * a)
    n, b, g = p["N10"] + p["N11"] * a, p["b10"] + p["b11"] * a, p["g10"] + p["g11"] * a
    return 2 * n * m200 / ((m200 / m1) ** -b + (m200 / m1) ** g)


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


# ---------- radial machinery ----------
R3 = np.logspace(-3.5, 3.7, 900)                 # kpc, 3D radii (cell edges)
RMID = np.sqrt(R3[1:] * R3[:-1])
R2 = np.logspace(-3.3, 2.3, 500)                 # kpc, projected radii
# M2D(R) = sum over shells of dM(shell) * f(R, r): 1 inside the cylinder, else
# the fraction of a thin shell of radius r falling inside projected radius R.
_x = np.clip(R2[:, None] / RMID[None, :], 0, 1)
PROJ = np.where(R2[:, None] >= RMID[None, :], 1.0, 1 - np.sqrt(1 - _x**2))


def nfw_m(r, rs, c):
    x = r / rs
    mu = lambda y: np.log1p(y) - y / (1 + y)
    return mu(x) / mu(c)


def hernquist_m(r, a):
    return r**2 / (r + a) ** 2


def enclosed_profiles(m200, c, r200, mstar, re, model):
    """Total 3D enclosed mass on R3 (kpc) for each halo row; arrays of shape (N,)."""
    r = R3[None, :]
    inside = r <= r200[:, None]
    m0 = m200[:, None] * np.where(inside, nfw_m(r, (r200 / c)[:, None], c[:, None]), 1.0)
    mst = mstar[:, None] * hernquist_m(r, (re / 1.8153)[:, None])
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


def outermost_root(g, theta, values):
    """theta of the outermost solution of g(theta) = v for each v (linear in a bracket)."""
    out = np.full(len(values), np.nan)
    for k, v in enumerate(values):
        d = g - v
        idx = np.nonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0]
        if idx.size:
            a = idx[-1]
            out[k] = theta[a] + (theta[a + 1] - theta[a]) * d[a] / (d[a] - d[a + 1])
    return out


def lens_properties(m3d, dl_kpc, scrit, nbeta=8):
    """Cross-section (sr) and image separations (arcsec) for each row of m3d."""
    dm = np.diff(m3d, axis=1)
    dm[:, 0] += m3d[:, 0]
    m2d = dm @ PROJ.T                                       # (N, len(R2))
    kbar = m2d / (np.pi * R2**2 * scrit[:, None])
    dlnm = np.gradient(np.log(np.maximum(m2d, 1e-30)), np.log(R2), axis=1)
    kap = 0.5 * kbar * dlnm                                 # Sigma = dM2D/dR / 2 pi R
    lam_r = 1 - 2 * kap + kbar                              # radial eigenvalue (diagnostic)
    theta = R2[None, :] / dl_kpc[:, None]                   # rad
    g = theta * (1 - kbar)                                  # beta(theta) = theta - alpha

    area = np.zeros(len(m3d))
    seps = np.full((len(m3d), nbeta), np.nan)
    pbeta = (np.arange(nbeta) + 0.5) / nbeta                # equal-area source samples
    for i in range(len(m3d)):
        # Multiply imaged sources have beta < max(alpha - theta): at the radial
        # critical curve for a cored or cuspy lens, at theta -> 0 (a cut) for an SIS.
        j = np.argmin(g[i])
        br = -g[i, j]
        if br <= 0:
            continue
        beta = br * np.sqrt(pbeta)
        # g need not be monotonic beyond its minimum (e.g. SPARC profiles), so take
        # the outermost bracketed root on each side rather than interpolating g.
        seps[i] = (outermost_root(g[i], theta[i], beta)
                   + outermost_root(g[i], theta[i], -beta)) / ARCSEC
        area[i] = np.pi * br**2
    return area, seps


# ---------- geometry ----------
def dist_mpc(z):
    return COSMO.comovingDistance(0.0, z) / H


def sigma_crit(zl, zs=ZS):
    """Msun / kpc^2 and the lens angular-diameter distance in kpc."""
    dcl, dcs = dist_mpc(zl), dist_mpc(zs)
    dl, ds, dls = dcl / (1 + zl), dcs / (1 + zs), (dcs - dcl) / (1 + zs)
    return C_KMS**2 / (4 * np.pi * G) * ds / (dl * dls) / 1e3, dl * 1e3


# ---------- population integral ----------
MODELS = ["NFW", "contracted", "cored n=0.5", "cored n=1"]
LOGM = np.arange(10.0, 14.01, 0.1)
DLNM = 0.1 * np.log(10)
ZL = np.linspace(0.05, 1.95, 39)
GH = [(-np.sqrt(3), 1 / 6), (0.0, 2 / 3), (np.sqrt(3), 1 / 6)]   # 3-point Gauss-Hermite
SIG_LOGMS, SIG_LOGRE = 0.15, 0.12
SEP_BINS = np.logspace(-2, np.log10(20), 61)


def run():
    dz = ZL[1] - ZL[0]
    hist = {m: np.zeros(SEP_BINS.size - 1) for m in MODELS}
    lens_m = {m: np.zeros(LOGM.size) for m in MODELS}
    for zl in ZL:
        scrit, dl = sigma_crit(zl)
        m200 = 10**LOGM
        dndlnm = mass_function.massFunction(m200 * H, zl, mdef="200c", model="tinker08",
                                            q_out="dndlnM") * H**3           # Mpc^-3
        c = concentration.concentration(m200 * H, "200c", zl, model="diemer19")
        r200 = mass_so.M_to_R(m200 * H, zl, "200c") / H                        # kpc
        dvdz = C_KMS / COSMO.Hz(zl) * dist_mpc(zl) ** 2                        # Mpc^3/sr
        ms0, re0 = mstar_moster13(m200, zl), None
        rows = []
        for xs, ws in GH:
            ms = ms0 * 10 ** (SIG_LOGMS * xs)
            re_med = re_vdw14(ms, zl)
            for xr, wr in GH:
                rows.append((ms, re_med * 10 ** (SIG_LOGRE * xr), ws * wr))
        for model in MODELS:
            for ms, re, w in rows:
                m3d = enclosed_profiles(m200, c, r200, ms, re, model)
                area, seps = lens_properties(m3d, np.full(m200.size, dl),
                                             np.full(m200.size, scrit))
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
    scrit, dl = sigma_crit(zl)
    m3d = (2 * s**2 * R3 / G)[None, :]
    area, seps = lens_properties(m3d, np.array([dl]), np.array([scrit]))
    dcl, dcs = dist_mpc(zl), dist_mpc(ZS)
    te = 4 * np.pi * (s / C_KMS) ** 2 * (dcs - dcl) / dcs / ARCSEC
    print(f"SIS check: theta_E {te:.3f}\"; separations {np.nanmin(seps):.3f}-"
          f"{np.nanmax(seps):.3f}\" (expect {2 * te:.3f}); area/(pi theta_E^2) = "
          f"{area[0] / (np.pi * (te * ARCSEC) ** 2):.3f}")


def check_slacs():
    """Median SLACS-like lens: z_l = 0.2, z_s = 0.6, M_* = 10^11.3."""
    zl, zs = 0.2, 0.6
    logm = np.linspace(12, 14.5, 251)
    ms = mstar_moster13(10**logm, zl)
    m200 = np.array([10 ** np.interp(11.3, np.log10(ms), logm)])
    c = concentration.concentration(m200 * H, "200c", zl, model="diemer19")
    r200 = mass_so.M_to_R(m200 * H, zl, "200c") / H
    mst = np.array([10**11.3])
    re = re_vdw14(mst, zl)
    scrit, dl = sigma_crit(zl, zs)
    print(f"SLACS-like: log M200 = {np.log10(m200[0]):.2f}, c = {c[0]:.1f}, "
          f"R_e = {re[0]:.1f} kpc ({re[0] / dl / ARCSEC:.2f}\")")
    for model in MODELS:
        m3d = enclosed_profiles(m200, c, r200, mst, re, model)
        dm = np.diff(m3d, axis=1); dm[:, 0] += m3d[:, 0]
        m2d = dm @ PROJ.T
        kbar = m2d[0] / (np.pi * R2**2 * scrit)
        te = np.interp(-1.0, -kbar, R2)                      # kbar decreasing outward
        # 3D total-density slope averaged over 0.5-1 R_e
        lr = np.log(R3)
        rho_slope = np.gradient(np.log(np.gradient(m3d[0], R3) / R3**2), lr)
        sel = (R3 > 0.5 * re[0]) & (R3 < re[0])
        print(f"  {model:12s} theta_E = {te / dl / ARCSEC:.2f}\" "
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
    fig.savefig(PLOTS / "separations_contraction_cores.png", dpi=150)
