"""How the sub-arcsecond lens abundance depends on the inner total-density slope of
intermediate-mass early types (sigma = 70-160 km/s), plus an illustrative evolution test.

Intermediate early types: rho ∝ r^-gamma inside R_e, isothermal outside, normalised so
that M(<R_e) equals the SIS value 2 sigma^2 R_e / G (the mass the stellar dispersion
measures). gamma = 2 recovers the SIS exactly. R_e = 1, 2, 3 kpc as size sensitivity.
Early types with sigma > 160 km/s stay SIS (calibrated by SLACS).
Evolution test (illustrative): early-type number density ∝ (1+z)^-nu, nu = 1.

Detectability: a source counts as lensed only if the fainter of the two outer images has
at least QMIN = 0.1 of the brighter one's flux (the flux-ratio range of the QUELLS
calibration grid). Without it, gamma > 2 profiles have divergent central deflection and
formally multiply image every source through vanishingly faint central counter-images.
Applied to all variants here, so ratios are like for like; the main comparison figure
has no flux cut.

plots/slope_test.png
"""
import matplotlib.pyplot as plt
import numpy as np

import compare_separations as CS
import lens_population as L
import sparc_lenses as S

GAMMAS = [1.6, 1.8, 2.0, 2.2, 2.4]
RES = [1.0, 2.0, 3.0]
SIG = np.logspace(np.log10(70), np.log10(500), 140)
W = S.schechter_dlnv(SIG, *CS.CHOI) * np.gradient(np.log(SIG))
INTER = SIG < CS.SIG_SLACS


def rows_for(gamma, re):
    m_re = 2 * SIG**2 * re / L.G                                   # SIS mass inside R_e
    r = L.R3[None, :]
    inner = m_re[:, None] * (r / re) ** (3 - gamma)
    outer = m_re[:, None] * (r / re)                               # isothermal outside
    return np.where(r < re, inner, outer)


QMIN = 0.1
NBETA = 24


def lens_q(m3d, dl, scrit):
    """Cross-section (sr) and separations (arcsec) counting only sources whose two outer
    images have flux ratio >= QMIN."""
    dm = np.diff(m3d, axis=1)
    dm[:, 0] += m3d[:, 0]
    m2d = dm @ L.PROJ.T
    kbar = m2d / (np.pi * L.R2**2 * scrit)
    kap = 0.5 * kbar * np.gradient(np.log(np.maximum(m2d, 1e-30)), np.log(L.R2), axis=1)
    theta = L.R2 / dl
    g = theta[None, :] * (1 - kbar)
    inv_mu = (1 - kbar) * (1 - 2 * kap + kbar)
    pb = (np.arange(NBETA) + 0.5) / NBETA
    area = np.zeros(len(m3d))
    seps = np.full((len(m3d), NBETA), np.nan)
    for i in range(len(m3d)):
        br = -g[i].min()
        if br <= 0:
            continue
        # Sources passing the flux cut lie within ~theta_E; cap the sampled region at
        # 2 theta_E so that divergent central deflections (gamma > 2) do not push every
        # sample to large beta. No effect for an SIS (beta_r = theta_E).
        te = L.outermost_root(g[i], theta, np.array([0.0]))[0]
        if np.isfinite(te):
            br = min(br, 2 * te)
        beta = br * np.sqrt(pb)
        ta = L.outermost_root(g[i], theta, beta)
        tb = L.outermost_root(g[i], theta, -beta)
        mua = 1 / np.abs(np.interp(ta, theta, inv_mu[i]))
        mub = 1 / np.abs(np.interp(tb, theta, inv_mu[i]))
        ok = np.isfinite(ta) & np.isfinite(tb) & (np.minimum(mua, mub) >= QMIN * np.maximum(mua, mub))
        area[i] = np.pi * br**2 * ok.mean()
        seps[i, ok] = (ta[ok] + tb[ok]) / L.ARCSEC
    return area, seps


def tau_hist(rows, w, nu=0.0):
    hist = np.zeros(L.SEP_BINS.size - 1)
    dz = L.ZL[1] - L.ZL[0]
    for zl in L.ZL:
        scrit, dl = L.sigma_crit(zl)
        area, seps = lens_q(rows, dl, scrit)
        wt = w * (1 + zl) ** (-nu) * area * L.C_KMS / L.COSMO.Hz(zl) * L.dist_mpc(zl) ** 2 * dz
        for i in range(len(rows)):
            ok = np.isfinite(seps[i])
            if ok.any():
                hist += np.histogram(seps[i, ok], L.SEP_BINS, weights=np.full(ok.sum(), wt[i] / ok.sum()))[0]
    return hist


c = np.sqrt(L.SEP_BINS[1:] * L.SEP_BINS[:-1])
band = lambda h, lo, hi: h[(c >= lo) & (c < hi)].sum()

massive = tau_hist(rows_for(2.0, 2.0)[~INTER], W[~INTER])
res = {}
for re in RES:
    for g in GAMMAS:
        res[(re, g)] = tau_hist(rows_for(g, re)[INTER], W[INTER])

# check: for an SIS the flux cut keeps beta <= theta_E (1 - q)/(1 + q), i.e. an area
# fraction ((1 - q)/(1 + q))^2 = 0.669 of the SIS optical depth used in the note
# (to within the sigma grids at the 160 km/s boundary, ~3%)
sis_inter = CS.sis_population(CS.CHOI, srange=(CS.SIG_COUNTS, CS.SIG_SLACS))
expect = ((1 - QMIN) / (1 + QMIN)) ** 2
ratio = res[(2.0, 2.0)].sum() / sis_inter.sum()
assert abs(ratio / expect - 1) < 0.05, (ratio, expect)
print(f"check: SIS with q >= {QMIN}: tau ratio {ratio:.3f} (expected {expect:.3f})")

print("intermediate early types (sigma 70-160): optical depth below 0.5\" relative to gamma = 2,")
print("and their share of all early-type lenses below 0.5\" and at 0.5-1\":")
for re in RES:
    ref = band(res[(re, 2.0)], 0.1, 0.5)
    for g in GAMMAS:
        h = res[(re, g)]
        s1 = band(h, 0.1, 0.5) / (band(h, 0.1, 0.5) + band(massive, 0.1, 0.5))
        s2 = band(h, 0.5, 1) / (band(h, 0.5, 1) + band(massive, 0.5, 1))
        print(f"  R_e={re:.0f} kpc gamma={g:.1f}: tau(<0.5\")/SIS = {band(h, 0.1, 0.5) / ref:5.2f}   "
              f"share <0.5\" {s1:.2f}   share 0.5-1\" {s2:.2f}")

# evolution sensitivity (all early types SIS)
allw = tau_hist(rows_for(2.0, 2.0), W)
ev = tau_hist(rows_for(2.0, 2.0), W, nu=1.0)
ev_m = tau_hist(rows_for(2.0, 2.0)[~INTER], W[~INTER], nu=1.0)
for lo, hi in [(0.1, 0.5), (0.5, 1), (1, 2), (2, 6)]:
    print(f"evolution nu=1, {lo}-{hi}\": tau ratio to non-evolving {band(ev, lo, hi) / band(allw, lo, hi):.2f}; "
          f"massive (sigma>160) share {band(ev_m, lo, hi) / band(ev, lo, hi):.2f} "
          f"(non-evolving {band(massive, lo, hi) / band(allw, lo, hi):.2f})")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
dlog = np.diff(np.log10(L.SEP_BINS))
cmap = plt.get_cmap("coolwarm")
for i, g in enumerate(GAMMAS):
    a1.step(c, res[(2.0, g)] / dlog, where="mid", color=cmap(i / (len(GAMMAS) - 1)),
            lw=2 if g == 2.0 else 1.3, label=rf"$\gamma={g:.1f}$" + (" (SIS)" if g == 2.0 else ""))
a1.step(c, massive / dlog, where="mid", color="k", lw=2, label=r"E/S0 $\sigma>160$ km/s (SIS, fixed)")
a1.set_xscale("log")
a1.set_yscale("log")
a1.set_xlim(0.08, 10)
a1.set_ylim(1e-6, 1.5e-3)
a1.set_xlabel("image separation [arcsec]")
a1.set_ylabel(r"$d\tau/d\log_{10}\Delta\theta$  ($z_s=2$)")
a1.set_title(r"E/S0 with $\sigma=70$-$160$ km/s: inner slope $\gamma$ ($R_e=2$ kpc)", fontsize=10)
a1.legend(frameon=False, fontsize=8, loc="upper left")
for re, ls in zip(RES, [":", "-", "--"]):
    ref = band(res[(re, 2.0)], 0.1, 0.5)
    a2.plot(GAMMAS, [band(res[(re, g)], 0.1, 0.5) / ref for g in GAMMAS], "o", ls=ls, color="k",
            label=rf"$R_e={re:.0f}$ kpc")
a2.axhline(1, color="0.7", lw=0.8)
a2.axvline(2, color="0.7", lw=0.8)
a2.set_yscale("log")
a2.set_xlabel(r"inner total-density slope $\gamma$ ($\rho\propto r^{-\gamma}$ inside $R_e$)")
a2.set_ylabel(r"lenses below $0.5''$ from $\sigma=70$-$160$, relative to SIS")
a2.set_title(r"sub-$0.5''$ abundance as a probe of central concentration", fontsize=10)
a2.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(L.PLOTS / "slope_test.png", dpi=150)
