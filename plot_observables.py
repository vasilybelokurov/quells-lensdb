"""The two inputs that set the lens statistics, on a common circular-velocity scale.

(a) How many: dn/dlog V_c from the measurements each model starts from
    Choi+07 early-type sigma function and Bernardi+10 all-galaxy sigma function
    (SIS: V_c = sqrt(2) sigma), HIPASS late-type rotation-velocity function
    (Zwaan+10, V_flat), and LCDM haloes (Tinker+08 mass function, z = 0, pure-DM
    NFW V_max with Diemer+19 concentrations).
(b-d) How concentrated: mean projected density inside R, Sigma_bar(<R), at fixed
    V_c, against Sigma_crit. A galaxy lenses only if its curve reaches Sigma_crit;
    the crossing radius is the Einstein radius.
    SIS: Sigma_bar = V_c^2 / (2 G R). Halo model: median galaxy whose peak V_c
    matches. SPARC: measured rotation curves, shown only where measured
    (R >= first point). SLACS: Sigma_bar inside R_Ein (= Sigma_crit of each lens,
    Auger+09) and inside R_e/2 (Auger+10), binned by V_c = sqrt(2) sigma.

plots/lensing_inputs.png
"""
import matplotlib.pyplot as plt
import numpy as np
from colossus.halo import concentration, mass_so
from colossus.lss import mass_function
from scipy.special import gamma

import lens_population as L
import obs_lenses as O
import plot_halo_models as P
import sparc_lenses as S

PC2 = 1e6                                       # pc^2 per kpc^2
VBANDS = [(110, 155), (155, 220), (220, 320)]    # km/s


def vf_sigma(v, phistar, sstar, a, b):
    """dn/dlog10 V_c for a modified-Schechter sigma function with V_c = sqrt(2) sigma."""
    s = v / np.sqrt(2)
    x = s / sstar
    return np.log(10) * phistar * x**a * np.exp(-x**b) * b / gamma(a / b)


def halo_vmax_function(v):
    """LCDM haloes at z = 0: dn/dlog10 V_max for pure-DM NFW haloes."""
    lm = np.linspace(9, 15.5, 400)
    m = 10**lm
    dndlnm = mass_function.massFunction(m * L.H, 0.0, mdef="200c", model="tinker08",
                                        q_out="dndlnM") * L.H**3
    c = concentration.concentration(m * L.H, "200c", 0.0, model="diemer19")
    r200 = mass_so.M_to_R(m * L.H, 0.0, "200c") / L.H
    v200 = np.sqrt(L.G * m / r200)
    mu = np.log1p(c) - c / (1 + c)
    vmax = v200 * np.sqrt(0.2162 * c / mu)
    dlnm_dlogv = np.gradient(np.log(m), np.log10(vmax))
    return np.interp(np.log10(v), np.log10(vmax), dndlnm * dlnm_dlogv)


def sigma_bar(m3d_rows):
    dm = np.diff(m3d_rows, axis=1)
    dm[:, 0] += m3d_rows[:, 0]
    return (dm @ L.PROJ.T) / (np.pi * L.R2**2) / PC2        # Msun / pc^2


def vc_peak(m3d_rows, rmax=60.0):
    sel = L.R3 < rmax
    return np.sqrt(L.G * m3d_rows[:, sel] / L.R3[sel]).max(axis=1)


# ---------- data ----------
sp_rows, sp_v = [], []
for line in open(O.RAW / "dwarfs" / "SPARC_Lelli2016c.mrt"):
    w = line.split()
    if len(w) < 18 or not w[17].isdigit() or not w[1].isdigit():
        continue
    if int(w[17]) > 2 or float(w[5]) < 30:
        continue
    rc = np.loadtxt(O.RAW / "dwarfs" / "rotmod" / f"{w[0]}_rotmod.dat")
    sp_rows.append(rc)
rows_sb, vel_sb = S.sparc_profiles("solid body")
sb_sparc = sigma_bar(rows_sb)
r0 = np.array([rc[0, 0] for rc in sp_rows])

slacs, a10 = O.slacs(), O.auger2010()
sig_auger = np.array([float(l.split("&")[2].replace("$", "").split("\\pm")[0])
                      for l in open(O.RAW / "slacs" / "auger2010_T_lens_data.tex")
                      if l.startswith("SDSSJ")])
mre2 = np.array([float(l.split("&")[5].replace("$", "").split("\\pm")[0])
                 for l in open(O.RAW / "slacs" / "auger2010_T_lens_data.tex")
                 if l.startswith("SDSSJ")])

# halo-model median galaxies at z_l = 0.6
logm = np.arange(10.5, 14.01, 0.02)
m, c, r200, ms, re = P.halo(logm, 0.6)
hm = {k: P.components(m, c, r200, ms, re, k) for k in P.COLS}
hm_tot = {k: v[0] + v[1] + v[2] for k, v in hm.items()}
hm_vc = vc_peak(hm_tot["NFW"])

_sc = [L.sigma_crit(z)[0] / PC2 for z in np.linspace(0.3, 0.8, 26)]
scrit_band = [min(_sc), max(_sc)]
_, DL06 = L.sigma_crit(0.6)

# ---------- figure ----------
fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
ax = axes[0, 0]
v = np.logspace(np.log10(20), np.log10(600), 300)
# Solid where the fit rests on measurements, dashed where it is extrapolated.
# Lower limits of the measured ranges (V_c = sqrt(2) sigma for the sigma functions):
#   Choi+07: complete to sigma ~ 70 km/s; Bernardi+10: SDSS fibre sigma reliable to
#   ~70 km/s (their preferred fit uses sigma > 125); HIPASS: V ~ 30 km/s and above.
hipass = lambda vv: (np.log(10) * 0.061 * (vv / 10**2.06) ** 0.66
                     * np.exp(-(vv / 10**2.06) ** 2.10) * 2.10 / gamma(0.66 / 2.10))
curves = [
    (lambda vv: vf_sigma(vv, 8.0e-3 * L.H**3, 161.0, 2.32, 2.67), np.sqrt(2) * 70, "C3",
     r"early types, SDSS $\sigma$ (Choi+07); $V_c=\sqrt{2}\sigma$"),
    (lambda vv: vf_sigma(vv, 2.099e-2, 113.78, 0.94, 1.85), np.sqrt(2) * 70, "C1",
     r"all galaxies, SDSS fibre $\sigma$ (Bernardi+10); $V_c=\sqrt{2}\sigma$"),
    (hipass, 30.0, "C2", r"gas-rich discs, HI widths (HIPASS, Zwaan+10)"),
]
for fn, vlo, col, lab in curves:
    meas = v >= vlo
    ax.plot(v[meas], fn(v[meas]), color=col, lw=2.2, label=lab)
    ax.plot(v[~meas], fn(v[~meas]), color=col, lw=1.4, ls="--")
ax.plot(v, halo_vmax_function(v), color="0.4", lw=1.4, ls=":",
        label=r"$\Lambda$CDM haloes, theory (Tinker+08 + NFW $V_{\max}$)")
ax.plot([], [], color="0.5", lw=1.4, ls="--", label="fit extrapolated beyond the data")
for lo, hi in VBANDS:
    ax.axvspan(lo, hi, color="0.9", zorder=0)
ax.plot(vel_sb, np.full(vel_sb.size, 2e-5), "|", color="C2", ms=10, alpha=0.6)
ax.text(22, 2.6e-5, "SPARC galaxies", color="C2", fontsize=8)
ax.plot(np.sqrt(2) * slacs["sigma"], np.full(slacs["sigma"].size, 1e-5), "|", color="0.2",
        ms=10, alpha=0.6)
ax.text(22, 1.3e-5, r"SLACS lenses ($\sqrt{2}\sigma$)", color="0.2", fontsize=8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(20, 600)
ax.set_xticks([20, 50, 100, 200, 500], labels=["20", "50", "100", "200", "500"])
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_ylim(5e-6, 3)
ax.set_xlabel(r"circular velocity $V_c$ [km s$^{-1}$]")
ax.set_ylabel(r"$dn/d\log_{10}V_c$ [Mpc$^{-3}$ dex$^{-1}$]")
ax.set_title("(a) how many: abundances, solid where measured", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 0.12))
ax2 = ax.secondary_xaxis("top", functions=(lambda vv: 2 * np.pi * (vv / L.C_KMS) ** 2 * 0.5 / L.ARCSEC * 2,
                                           lambda t: L.C_KMS * np.sqrt(t * L.ARCSEC / (4 * np.pi * 0.5))))
ax2.set_xlabel(r"SIS separation for $D_{ls}/D_s=0.5$ [arcsec]", fontsize=8)

for ax, (lo, hi) in zip(axes.flat[1:], VBANDS):
    vmid = np.sqrt(lo * hi)
    ax.axhspan(*scrit_band, color="C1", alpha=0.25, lw=0)
    ax.text(0.12, scrit_band[1] * 1.15, r"$\Sigma_{\rm crit}$, $z_l=0.3$-$0.8$, $z_s=2$",
            color="C1", fontsize=8)
    ax.plot(L.R2, vmid**2 / (2 * L.G * L.R2) / PC2, color="0.5", lw=1.4, ls=":",
            label=rf"SIS, $V_c={vmid:.0f}$ km/s")
    j = np.argmin(np.abs(hm_vc - vmid))
    for k, col in P.COLS.items():
        if k == "cored n=0.5":
            continue
        ax.plot(L.R2, sigma_bar(hm_tot[k][j:j + 1])[0], color=col, lw=1.8,
                label=f"halo model {k}" + (rf" ($\log M_{{200}}={logm[j]:.1f}$)" if k == "NFW" else ""))
    ax.plot(L.R2, sigma_bar(hm["NFW"][1][j:j + 1])[0], color="k", lw=1.2, ls="--",
            label=rf"halo model, stars only ($R_e={re[j]:.1f}$ kpc)")
    sel = np.nonzero((vel_sb >= lo) & (vel_sb < hi))[0]
    for n, i in enumerate(sel):
        ok = (L.R2 >= r0[i]) & (L.R2 <= sp_rows[i][-1, 0])
        ax.plot(L.R2[ok], sb_sparc[i, ok], color="C2", lw=0.8, alpha=0.55,
                label=f"SPARC, measured ({sel.size} galaxies)" if n == 0 else None)
    vs = np.sqrt(2) * slacs["sigma"]
    s1 = (vs >= lo) & (vs < hi) & np.isfinite(slacs["rein_kpc"])
    if s1.any():
        ax.scatter(slacs["rein_kpc"][s1], 10 ** slacs["log_mein"][s1] / (np.pi * slacs["rein_kpc"][s1] ** 2) / PC2,
                   s=22, color="k", zorder=5, label=r"SLACS, inside $R_{\rm Ein}$")
    va = np.sqrt(2) * sig_auger
    s2 = (va >= lo) & (va < hi)
    if s2.any():
        r2 = a10["re_kpc"][s2] / 2
        ax.scatter(r2, 10 ** mre2[s2] / (np.pi * r2**2) / PC2, s=22, facecolors="none",
                   edgecolors="k", zorder=5, label=r"SLACS, inside $R_e/2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 30)
    ax.set_ylim(10, 1e5)
    ax.set_xlabel("projected radius R [kpc]")
    ax.set_ylabel(r"$\bar\Sigma(<R)$ [M$_\odot$ pc$^{-2}$]")
    ax.set_title(rf"how concentrated: $V_c={lo}$-${hi}$ km s$^{{-1}}$", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    top = ax.secondary_xaxis("top", functions=(lambda r: r / DL06 / L.ARCSEC,
                                               lambda t: t * DL06 * L.ARCSEC))
    top.set_xlabel(r"arcsec at $z_l=0.6$", fontsize=8)
for ax, lab in zip(axes.flat[1:], "bcd"):
    ax.set_title(f"({lab}) " + ax.get_title(), fontsize=10)
fig.tight_layout()
fig.savefig(L.PLOTS / "lensing_inputs.png", dpi=150)

# numbers for the text
print(f"Sigma_crit band (z_s=2, z_l 0.3-0.8): {scrit_band[0]:.0f}-{scrit_band[1]:.0f} Msun/pc^2")
for lo, hi in VBANDS:
    vmid = np.sqrt(lo * hi)
    sel = np.nonzero((vel_sb >= lo) & (vel_sb < hi))[0]
    lens = sum(np.nanmax(np.where(L.R2 >= r0[i], sb_sparc[i], np.nan)) >= scrit_band[0] for i in sel)
    j = np.argmin(np.abs(hm_vc - vmid))
    print(f"V_c {lo}-{hi}: SPARC {sel.size} galaxies, {lens} reach Sigma_crit within the measured range; "
          f"halo-model NFW galaxy log M200 {logm[j]:.2f}, log M* {np.log10(ms[j]):.2f}, R_e {re[j]:.1f} kpc")
    for name, fn in [("Choi", lambda vv: vf_sigma(vv, 8.0e-3 * L.H**3, 161.0, 2.32, 2.67)),
                     ("Bernardi", lambda vv: vf_sigma(vv, 2.099e-2, 113.78, 0.94, 1.85)),
                     ("HIPASS", lambda vv: np.log(10) * 0.061 * (vv / 10**2.06) ** 0.66
                      * np.exp(-(vv / 10**2.06) ** 2.10) * 2.10 / gamma(0.66 / 2.10)),
                     ("haloes", halo_vmax_function)]:
        print(f"     dn/dlogV at {vmid:.0f} km/s, {name:8s}: {fn(np.array([vmid]))[0]:.2e}")
