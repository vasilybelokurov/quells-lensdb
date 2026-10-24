"""Profiles and scaling relations of the halo model in lens_population.py:
cuspy (NFW), contracted and cored dark matter around the same galaxy.

plots/halo_model_profiles.png  two haloes: densities, enclosed masses, mean convergence
plots/halo_model_scalings.png  median galaxy per halo mass: M_*, R_e, separation,
                               dark-matter fractions inside theta_E and R_e
"""
import matplotlib.pyplot as plt
import numpy as np
from colossus.halo import concentration, mass_so

import lens_population as L
import obs_lenses as O
import paths
from lenscore import cosmology as cosmo
from lenscore import lensing
from lenscore import profiles as prof

ZL = 0.6
COLS = {"NFW": "k", "contracted": "C3", "cored n=0.5": "C0", "cored n=1": "C9"}
SCRIT, DL = cosmo.sigma_crit(ZL)


def halo(logm, z=ZL):
    m = np.atleast_1d(10.0**np.asarray(logm))
    c = concentration.concentration(m * cosmo.H, "200c", z, model="diemer19")
    r200 = mass_so.M_to_R(m * cosmo.H, z, "200c") / cosmo.H
    ms = prof.mstar_moster13(m, z)
    return m, c, r200, ms, L.re_vdw14(ms, z)


def components(m, c, r200, ms, re, model):
    """Enclosed DM, stellar and gas mass on lensing.R3 for one model."""
    r = lensing.R3[None, :]
    m0 = m[:, None] * np.where(r <= r200[:, None], prof.nfw_m(r, (r200 / c)[:, None], c[:, None]), 1.0)
    mst = ms[:, None] * prof.hernquist_m(r, (re / 1.8153)[:, None])
    mgas = (cosmo.FB * m - ms)[:, None] * m0 / m[:, None]
    mdm = L.enclosed_profiles(m, c, r200, ms, re, model) - mst - mgas
    return mdm, mst, mgas


def project(m3d):
    return lensing.project(m3d)


def density(m3d):
    return np.gradient(m3d, lensing.R3, axis=1) / (4 * np.pi * lensing.R3**2) / 1e9   # Msun / pc^3


def theta_e(m2d, scrit=SCRIT):
    kbar = m2d / (np.pi * lensing.R2**2 * scrit)
    return np.array([np.interp(-1.0, -k, lensing.R2) for k in kbar])              # kpc


# ---------- figure 1: profiles of two haloes ----------
fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
for row, logm in enumerate([12.0, 13.0]):
    m, c, r200, ms, re = halo(logm)
    axd, axm, axk = axes[row]
    for model, col in COLS.items():
        mdm, mst, mgas = components(m, c, r200, ms, re, model)
        axd.loglog(lensing.R3, density(mdm)[0], color=col, lw=1.6, label=f"DM, {model}")
        axm.loglog(lensing.R3, mdm[0], color=col, lw=1.6)
        tot = project(mdm + mst + mgas)
        kbar = tot[0] / (np.pi * lensing.R2**2 * SCRIT)
        axk.loglog(lensing.R2, kbar, color=col, lw=1.6)
        te = theta_e(tot)[0]
        axk.plot(te, 1, "o", color=col, ms=6, zorder=5)
        if model == "NFW":
            axd.loglog(lensing.R3, density(mst)[0], color="C1", lw=2, ls="--", label="stars")
            axm.loglog(lensing.R3, mst[0], color="C1", lw=2, ls="--", label="stars")
            axm.loglog(lensing.R3, mgas[0], color="0.6", lw=1, ls=":", label="diffuse gas")
            st2 = project(mst)[0] / (np.pi * lensing.R2**2 * SCRIT)
            axk.loglog(lensing.R2, st2, color="C1", lw=2, ls="--", label="stars only")
    for ax in (axd, axm, axk):
        ax.axvline(re[0], color="C1", lw=0.8, ls=":")
        ax.axvline(1.75 * re[0], color="C9", lw=0.8, ls=":")
        ax.set_xlim(0.05, 300 if ax is axm else 100)
    axk.axhline(1, color="0.4", lw=0.8)
    axd.set_ylim(1e-5, 1e3)
    axm.set_ylim(1e7, 3 * m[0])
    axk.set_ylim(0.05, 100)
    axd.set_ylabel(r"$\rho$ [M$_\odot$ pc$^{-3}$]")
    axm.set_ylabel(r"$M(<r)$ [M$_\odot$]")
    axk.set_ylabel(r"$\bar\kappa(<R)$, $z_l=0.6$, $z_s=2$")
    info = (f"$\\log M_{{200c}}={logm:.1f}$, $\\log M_*={np.log10(ms[0]):.2f}$\n"
            f"$R_e={re[0]:.1f}$ kpc, $c={c[0]:.1f}$")
    axd.text(0.04, 0.05, info, transform=axd.transAxes, fontsize=9)
    sec = axk.secondary_xaxis("top", functions=(lambda r: r / DL / cosmo.ARCSEC,
                                                lambda t: t * DL * cosmo.ARCSEC))
    sec.set_xlabel("arcsec", fontsize=8)
    axk.text(0.97, 0.94, r"dots: $\theta_E$; dotted: $R_e$, $1.75R_e$",
             transform=axk.transAxes, ha="right", va="top", fontsize=8)
for ax in axes[1]:
    ax.set_xlabel("radius [kpc]")
axes[1, 2].set_xlabel("projected radius R [kpc]")
axes[0, 2].set_xlabel("")
axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
axes[0, 1].legend(frameon=False, fontsize=8, loc="lower right")
axes[0, 2].legend(frameon=False, fontsize=8, loc="lower left")
axes[0, 0].set_title("density", fontsize=10)
axes[0, 1].set_title("enclosed mass (DM by model)", fontsize=10)
axes[0, 2].set_title("mean convergence inside R", fontsize=10)
fig.tight_layout()
fig.savefig(paths.PLOTS / "halo_model_profiles.png", dpi=150)

# ---------- figure 2: scalings, with observations ----------
def model_scalings(z, logm):
    """Median-galaxy quantities per dark-matter model at redshift z."""
    m, c, r200, ms, re = halo(logm, z)
    out = dict(m=m, c=c, ms=ms, re=re, models={})
    scrit, dl = cosmo.sigma_crit(z) if z > 0.05 else (np.nan, np.nan)
    for model in COLS:
        mdm, mst, mgas = components(m, c, r200, ms, re, model)
        tot3, tot2, dm2 = mdm + mst + mgas, project(mdm + mst + mgas), project(mdm)
        at2 = lambda rad, num: np.array([np.interp(x, lensing.R2, a / b) for x, a, b in zip(rad, num, tot2)])
        d = dict(fdm_re=at2(re, dm2), fdm_re2=at2(re / 2, dm2),
                 fdm3d_re=np.array([np.interp(x, lensing.R3, a / b) for x, a, b in zip(re, mdm, tot3)]),
                 rho150=np.array([np.interp(0.15, lensing.R3, row) for row in density(mdm)]))
        if np.isfinite(scrit):
            te = theta_e(tot2, scrit)
            d["sep"] = 2 * te / dl / cosmo.ARCSEC
            d["fdm_te"] = at2(te, dm2)
        out["models"][model] = d
    return out


def auger10_shmr(logmvir, logms0, logmvir0, g1=7.17, g2=0.201, b=0.557):
    """Auger et al. 2010 (ApJ 721, L163) M_*(M_vir), Salpeter SPS masses."""
    x = 10 ** (logmvir - logmvir0)
    return logms0 + g1 * np.log10(x) - (g1 - g2) / b * np.log10(1 + x**b)


logm = np.arange(9.5, 14.01, 0.05)
lens = model_scalings(ZL, logm)
local = model_scalings(0.0, logm)
slacs_z = model_scalings(0.2, logm)
slacs, sl2s, a10 = O.slacs(), O.sl2s(), O.auger2010()
rd, sp = O.read2019(), O.sparc()
dsph, dirr = rd["dsph"], ~rd["dsph"]

fig, axes = plt.subplots(3, 3, figsize=(14, 12.5))
mlab = r"$\log M_{200c}$ [M$_\odot$]"
slab = r"$\log M_*$ (Chabrier) [M$_\odot$]"

# (a) stellar-to-halo mass relation
ax = axes[0, 0]
ax.plot(logm, np.log10(lens["ms"]), color="C1", lw=2, label=f"Moster+13, $z={ZL}$")
ax.plot(logm, np.log10(local["ms"]), color="C1", lw=1.2, ls="--", label="Moster+13, $z=0$")
lv = np.linspace(12.3, 14.0, 50)
for (lm0, lv0, ls_), lab in [((11.29, 10.38, "-"), "NFW"), ((11.36, 10.43, ":"), "Gnedin AC")]:
    ax.plot(lv, auger10_shmr(lv, lm0, lv0) - 0.25, color="0.35", lw=1.4, ls=ls_,
            label=f"SLACS fit, {lab} (Auger+10b)")
ax.errorbar([np.log10(14e12 / 0.7)], [np.log10(2.55e11 / 0.7)],
            xerr=[[np.log10(14 / 9)], [np.log10(20 / 14)]], fmt="*", ms=11, color="k",
            label="SLACS stack, WL+SL (Gavazzi+07)")
ax.scatter(rd["logm200"][dsph], rd["logms"][dsph], s=30, marker="s", color="C4", label="dSph (Read+19)")
ax.scatter(rd["logm200"][dirr], rd["logms"][dirr], s=30, marker="D", facecolors="none",
           edgecolors="C4", label="dIrr (Read+19)")
ax.set_xlabel(mlab)
ax.set_ylabel(r"$\log M_*$ [M$_\odot$]")
ax.set_xlim(8.5, 14.2)
ax.set_ylim(5, 12)
ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.set_title("stellar vs halo mass", fontsize=10)

# (b) size-mass relation
ax = axes[0, 1]
ax.scatter(sp["logms"], sp["re_kpc"], s=9, color="C2", alpha=0.5, lw=0, label="SPARC discs")
ax.scatter(slacs["logms"], slacs["re_kpc"], s=12, color="0.25", lw=0, label=r"SLACS ($z\approx0.2$)")
ax.scatter(sl2s["logms"], sl2s["re_kpc"], s=14, facecolors="none", edgecolors="0.4",
           label=r"SL2S ($z\approx0.5$)")
ax.scatter(rd["logms"][dsph], rd["re_kpc"][dsph], s=30, marker="s", color="C4", label="dSph")
ax.scatter(rd["logms"][dirr], rd["re_kpc"][dirr], s=30, marker="D", facecolors="none",
           edgecolors="C4", label="dIrr")
lms = np.linspace(5, 12, 141)
ax.plot(lms, L.re_vdw14(10**lms, 0.25), color="C1", lw=1.4, ls="--", label="model, $z=0.25$")
ax.plot(lms, L.re_vdw14(10**lms, ZL), color="C1", lw=2, label=f"model, $z={ZL}$")
ax.set_yscale("log")
ax.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20], labels=["0.2", "0.5", "1", "2", "5", "10", "20"])
ax.yaxis.set_minor_locator(plt.NullLocator())
ax.set_xlim(5, 12)
ax.set_xlabel(slab)
ax.set_ylabel(r"$R_e$ [kpc]")
ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
ax.set_title("size vs stellar mass (model: early types)", fontsize=10)

# (c) concentration
ax = axes[0, 2]
ax.plot(logm, lens["c"], color="k", lw=2, label=f"$z={ZL}$")
ax.plot(logm, local["c"], color="k", lw=1.2, ls="--", label="$z=0$")
ax.set_xlabel(mlab)
ax.set_ylabel(r"$c_{200c}$ (Diemer & Joyce 2019)")
ax.legend(frameon=False, fontsize=8)
ax.set_title("halo concentration (model only)", fontsize=10)

# (d)-(f) lensing at z_l = 0.6
for model, col in COLS.items():
    d = lens["models"][model]
    axes[1, 0].plot(logm, d["sep"], color=col, lw=1.8, label=model)
    axes[1, 1].plot(logm, d["fdm_te"], color=col, lw=1.8)
    axes[1, 2].plot(logm, d["fdm_re"], color=col, lw=1.8)
axes[1, 0].set_yscale("log")
axes[1, 0].set_ylim(1e-3, 10)
axes[1, 0].set_ylabel(r"separation $2\theta_E$ [arcsec], $z_l=0.6$, $z_s=2$")
axes[1, 0].legend(frameon=False, fontsize=8)
axes[1, 1].errorbar([np.log10(14e12 / 0.7)], [0.37], yerr=[0.18], fmt="*", ms=11, color="k",
                    label=r"SLACS, inside $R_{\rm Ein}$ (Gavazzi+07; bar = rms)")
axes[1, 1].legend(frameon=False, fontsize=7.5, loc="upper left")
axes[1, 1].set_ylabel(r"DM fraction inside $\theta_E$ (projected)")
axes[1, 2].set_ylabel(r"DM fraction inside $R_e$ (projected)")
for ax in axes[1]:
    ax.set_xlabel(mlab)
    ax.set_xlim(11, 14.1)
for ax in (axes[1, 1], axes[1, 2]):
    ax.set_ylim(0, 1)

# (g) inner dark-matter density of dwarfs
ax = axes[2, 0]
for model, col in COLS.items():
    ax.plot(np.log10(local["ms"]), local["models"][model]["rho150"], color=col, lw=1.8, label=model)
ax.errorbar(rd["logms"][dsph], rd["rho150"][dsph], yerr=rd["e_rho150"][:, dsph], fmt="s", ms=5,
            color="C4", label="dSph (Read+19)")
ax.errorbar(rd["logms"][dirr], rd["rho150"][dirr], yerr=rd["e_rho150"][:, dirr], fmt="D", ms=5,
            mfc="none", color="C4", label="dIrr (Read+19)")
ax.set_yscale("log")
ax.set_xlim(5, 11.5)
ax.set_xlabel(slab)
ax.set_ylabel(r"$\rho_{\rm DM}(150\,{\rm pc})$ [M$_\odot$ pc$^{-3}$], $z=0$")
ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
ax.set_title("inner DM density (dwarfs)", fontsize=10)

# (h) 3D DM fraction inside R_e: SPARC rotation curves
ax = axes[2, 1]
ax.scatter(sp["logms"], np.clip(sp["fdm3d_re"], -0.3, 1), s=10, color="C2", alpha=0.6, lw=0,
           label=r"SPARC, $1-V_{\rm bar}^2/V_{\rm obs}^2$ at $R_{\rm eff}$")
for model, col in COLS.items():
    ax.plot(np.log10(local["ms"]), local["models"][model]["fdm3d_re"], color=col, lw=1.8)
ax.axhline(0, color="0.8", lw=0.8)
ax.set_ylim(-0.3, 1)
ax.set_xlim(6.5, 11.6)
ax.set_xlabel(slab)
ax.set_ylabel(r"DM fraction inside $R_e$ (3D), $z=0$")
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
ax.set_title("discs and dwarfs: rotation curves", fontsize=10)

# (i) projected DM fraction inside R_e/2: SLACS lenses
ax = axes[2, 2]
ax.errorbar(a10["logms_chab"], a10["fdm_chab"], yerr=a10["e_fdm_chab"], fmt="o", ms=3.5,
            color="0.2", ecolor="0.75", elinewidth=0.8, label="SLACS, Chabrier (Auger+10)")
ax.scatter(a10["logms_chab"], a10["fdm_salp"], s=14, facecolors="none", edgecolors="0.45",
           label="SLACS, Salpeter")
for model, col in COLS.items():
    ax.plot(np.log10(slacs_z["ms"]), slacs_z["models"][model]["fdm_re2"], color=col, lw=1.8)
ax.axhline(0, color="0.8", lw=0.8)
ax.set_xlim(10.3, 12)
ax.set_ylim(-0.4, 1)
ax.set_xlabel(slab)
ax.set_ylabel(r"DM fraction inside $R_e/2$ (projected), $z=0.2$")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.set_title("massive lenses", fontsize=10)

fig.suptitle("median galaxy in each halo; lines: model (colours as in panel d), symbols: data",
             fontsize=10)
fig.tight_layout()
fig.savefig(paths.PLOTS / "halo_model_scalings.png", dpi=150)
