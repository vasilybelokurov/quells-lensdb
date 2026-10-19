"""The inputs behind plots/separations_compare.png.

(a) number counts: dn/dlog10 V_c of the abundance functions (solid where measured)
(b) where the optical depth comes from: dtau/dlog10 V_c for each population
(c) mass structure: Sigma_bar(<R) for SIS (set by sigma alone) and for the SPARC
    templates (measured range solid; solid-body and flat inner extrapolations dashed)
(d) kinematics-to-mass calibration: SLACS sigma_SIE from the measured Einstein radius
    against the SDSS aperture stellar dispersion (Auger+09 tables)

plots/separations_compare_inputs.png
"""
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

import compare_separations as CS
import lens_population as L
import obs_lenses as O
import sparc_lenses as S

PC2 = 1e6
SQ2 = np.sqrt(2)


def per_row_tau(rows, w):
    """Optical depth contributed by each row (summed over lens redshift)."""
    tau = np.zeros(len(rows))
    dz = L.ZL[1] - L.ZL[0]
    for zl in L.ZL:
        scrit, dl = L.sigma_crit(zl)
        area, _ = L.lens_properties(rows, np.full(len(rows), dl), np.full(len(rows), scrit), nbeta=1)
        tau += w * area * L.C_KMS / L.COSMO.Hz(zl) * L.dist_mpc(zl) ** 2 * dz
    return tau


def sigma_bar(rows):
    dm = np.diff(rows, axis=1)
    dm[:, 0] += rows[:, 0]
    return (dm @ L.PROJ.T) / (np.pi * L.R2**2) / PC2


# ---------- populations ----------
sig = np.logspace(np.log10(10), np.log10(500), 160)
dlns = np.gradient(np.log(sig))
sis_rows = 2 * sig[:, None] ** 2 * L.R3[None, :] / L.G
tau_choi = per_row_tau(sis_rows, S.schechter_dlnv(sig, *CS.CHOI) * dlns)
tau_bern = per_row_tau(sis_rows, S.schechter_dlnv(sig, *CS.BERNARDI) * dlns)
rows_sb, vel = S.sparc_profiles("solid body")
rows_fl, _ = S.sparc_profiles("flat")
w_sp = S.weights(vel)
tau_sb, tau_fl = per_row_tau(rows_sb, w_sp), per_row_tau(rows_fl, w_sp)

# SPARC measured radial ranges (same selection and distance rescaling as sparc_lenses)
ranges = []
for line in open(O.RAW / "dwarfs" / "SPARC_Lelli2016c.mrt"):
    w = line.split()
    if len(w) < 18 or not w[17].isdigit() or not w[1].isdigit():
        continue
    if int(w[17]) > 2 or float(w[5]) < 30 or int(w[1]) == 0:
        continue
    rc = np.loadtxt(O.RAW / "dwarfs" / "rotmod" / f"{w[0]}_rotmod.dat")
    fac = 73.0 / 70.0 if int(w[4]) == 1 else 1.0
    ranges.append((rc[0, 0] * fac, rc[-1, 0] * fac))
ranges = np.array(ranges)
assert len(ranges) == len(vel)

# SLACS: sigma_SIE from the Einstein radius
s = O.slacs()
ok = np.isfinite(s["rein_kpc"]) & np.isfinite(s["sigma"]) & np.isfinite(s["zs"])
dcl, dcs = L.dist_mpc(s["zl"][ok]), L.dist_mpc(s["zs"][ok])
theta_e = s["rein_kpc"][ok] / (dcl / (1 + s["zl"][ok]) * 1e3)
sig_sie = L.C_KMS * np.sqrt(theta_e * dcs / (4 * np.pi * (dcs - dcl)))
sig_star = s["sigma"][ok]

# ---------- figure ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 10.2))
v = np.logspace(np.log10(20), np.log10(700), 400)

ax = axes[0, 0]
hip = lambda vv: S.schechter_dlnv(vv, 0.061, 10**2.06, 0.66, 2.10) * np.log(10)
choi = lambda vv: S.schechter_dlnv(vv / SQ2, *CS.CHOI) * np.log(10)
bern = lambda vv: S.schechter_dlnv(vv / SQ2, *CS.BERNARDI) * np.log(10)
for fn, vlo, col, lab in [(choi, SQ2 * 70, "C3", r"E/S0: Choi+07 $\sigma_*$ function, $V_c=\sqrt{2}\sigma$"),
                          (bern, SQ2 * 70, "C1", r"all types: Bernardi+10 fibre $\sigma$, $V_c=\sqrt{2}\sigma$"),
                          (hip, 30.0, "C2", r"HI-rich Sa+: HIPASS $V_{\rm rot}$ (Zwaan+10)")]:
    m = v >= vlo
    ax.plot(v[m], fn(v[m]), color=col, lw=2.2, label=lab)
    ax.plot(v[~m], fn(v[~m]), color=col, lw=1.3, ls="--")
ax.plot([], [], color="0.5", ls="--", label="extrapolated beyond the measurements")
ax.plot(vel, np.full(vel.size, 3e-6), "|", color="C2", ms=10, alpha=0.6)
ax.text(22, 4.2e-6, f"SPARC templates ({vel.size}), shared per 0.1 dex", color="C2", fontsize=8)
ax.set_ylim(1e-6, 0.3)
ax.set_ylabel(r"$dn/d\log_{10}V_c$ [Mpc$^{-3}$ dex$^{-1}$]")
ax.set_title("(a) number counts", fontsize=10)

ax = axes[0, 1]
dlogv = np.gradient(np.log10(sig))
ax.plot(SQ2 * sig, tau_choi / dlogv, color="C3", lw=2.2, label=f"E/S0 (Choi), $\\tau$={tau_choi.sum():.1e}")
ax.plot(SQ2 * sig, tau_bern / dlogv, color="C1", lw=1.6, ls=":", label=f"Bernardi, $\\tau$={tau_bern.sum():.1e}")
lv = np.log10(vel)
for t, sty, lab in [(tau_sb, dict(color="C2", lw=2.2), "solid-body centre"),
                    (tau_fl, dict(color="C2", lw=1.3, ls="--"), "flat centre")]:
    h, e = np.histogram(lv, S.VBINS, weights=t)
    ax.step(10 ** (0.5 * (e[1:] + e[:-1])), h / np.diff(e), where="mid",
            label=f"HI-rich Sa+, {lab}, $\\tau$={t.sum():.1e}", **sty)
ax.axvspan(SQ2 * np.nanmin(sig_star), SQ2 * np.nanmax(sig_star), color="0.88", zorder=0,
           label=r"$\sqrt{2}\sigma$ range of SLACS lenses")
ax.set_ylim(1e-7, 3e-3)
ax.set_ylabel(r"$d\tau/d\log_{10}V_c$  ($z_s=2$)")
ax.set_title("(b) where the optical depth comes from", fontsize=10)
for ax in axes[0]:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(20, 700)
    ax.set_xticks([20, 50, 100, 200, 500], labels=["20", "50", "100", "200", "500"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"circular velocity $V_c$ [km s$^{-1}$]")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

ax = axes[1, 0]
sc = [L.sigma_crit(z)[0] / PC2 for z in np.linspace(0.3, 0.8, 26)]
ax.axhspan(min(sc), max(sc), color="C1", alpha=0.3, lw=0)
ax.text(29, max(sc) * 1.12, r"$\Sigma_{\rm crit}$ ($z_l=0.3$-$0.8$, $z_s=2$)", color="C1", fontsize=8, ha="right")
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(np.log10(50), np.log10(320))
sb_sb, sb_fl = sigma_bar(rows_sb), sigma_bar(rows_fl)
for i in np.argsort(vel):
    col = cmap(norm(np.log10(vel[i])))
    meas = (L.R2 >= ranges[i, 0]) & (L.R2 <= ranges[i, 1])
    inner = L.R2 < ranges[i, 0]
    ax.plot(L.R2[meas], sb_sb[i, meas], color=col, lw=0.8, alpha=0.8)
    ax.plot(L.R2[inner], sb_sb[i, inner], color=col, lw=0.5, ls=":", alpha=0.35)
    ax.plot(L.R2[inner], sb_fl[i, inner], color=col, lw=0.5, ls="--", alpha=0.35)
for vc in (100, 150, 200, 300):
    ax.plot(L.R2, vc**2 / (2 * L.G * L.R2) / PC2, color=cmap(norm(np.log10(vc))), lw=2.6,
            path_effects=[pe.Stroke(linewidth=4.2, foreground="k"), pe.Normal()], zorder=5)
    ax.text(22, vc**2 / (2 * L.G * 22) / PC2 * 1.25, f"SIS {vc}", fontsize=8, ha="right",
            color="k", zorder=6)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, ax=ax, pad=0.01)
cb.set_label(r"$\log_{10}V_c$ (SPARC: $V_{\rm flat}$; SIS: $\sqrt{2}\sigma$)")
ax.plot([], [], color="0.3", lw=2.4, label=r"SIS: $\bar\Sigma=V_c^2/(2GR)$, fixed by $\sigma$")
ax.plot([], [], color="0.3", lw=0.8, label="SPARC, measured range")
ax.plot([], [], color="0.3", lw=0.6, ls=":", label="SPARC, solid-body centre")
ax.plot([], [], color="0.3", lw=0.6, ls="--", label="SPARC, flat centre")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.1, 30)
ax.set_ylim(3, 1e5)
ax.set_xlabel("projected radius R [kpc]")
ax.set_ylabel(r"$\bar\Sigma(<R)$ [M$_\odot$ pc$^{-2}$]")
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
ax.set_title("(c) mass structure: SIS assumed, SPARC measured", fontsize=10)

ax = axes[1, 1]
ax.scatter(sig_star, sig_sie, s=16, color="k", label=f"SLACS ({ok.sum()} lenses, Auger+09)")
xx = np.array([50, 450])
ax.plot(xx, xx, color="C3", lw=1.6, label=r"$\sigma_{\rm SIS}=\sigma_*$ (assumed for E/S0)")
ax.fill_between(xx, xx / 1.05, xx / 0.95, color="C3", alpha=0.15, lw=0, label="calibration band $f=0.95$-$1.05$")
ratio = np.median(sig_star / sig_sie)
ax.plot(xx, xx / ratio, color="0.4", lw=1, ls="--",
        label=rf"median $\sigma_{{\rm SDSS}}/\sigma_{{\rm SIE}}={ratio:.3f}$")
c50 = np.cumsum(tau_choi) / tau_choi.sum()
lo, hi = np.interp([0.05, 0.95], c50, sig)
ax.axvspan(lo, hi, color="C3", alpha=0.07, zorder=0, label=rf"90% of E/S0 $\tau$: $\sigma$={lo:.0f}-{hi:.0f} km/s")
ax.set_xlim(60, 420)
ax.set_ylim(60, 420)
ax.set_xlabel(r"SDSS aperture stellar dispersion $\sigma_*$ [km s$^{-1}$]")
ax.set_ylabel(r"$\sigma_{\rm SIE}$ from the Einstein radius [km s$^{-1}$]")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.set_title("(d) kinematics to lensing mass: tested only for massive lenses", fontsize=10)

fig.tight_layout()
fig.savefig(L.PLOTS / "separations_compare_inputs.png", dpi=150)

# numbers for the text
print(f"E/S0: 90% of tau from sigma {lo:.0f}-{hi:.0f} km/s (V_c {SQ2*lo:.0f}-{SQ2*hi:.0f}); "
      f"SLACS sigma_* {np.nanmin(sig_star):.0f}-{np.nanmax(sig_star):.0f}, "
      f"median sigma_SDSS/sigma_SIE = {ratio:.3f} (Bolton+08: 0.948 +- 0.008)")
frac_below_slacs = tau_choi[sig < np.nanmin(sig_star)].sum() / tau_choi.sum()
print(f"share of E/S0 tau from sigma below the smallest SLACS sigma: {frac_below_slacs:.2f}")
for name, t, vv in [("solid", tau_sb, vel), ("flat", tau_fl, vel)]:
    print(f"SPARC {name}: share of tau from V_c < 160: {t[vv < 160].sum() / t.sum():.2f}")
