"""Predicted image-separation distribution of lensed quasars, compared with the
known lenses.

Model: non-evolving population of singular isothermal spheres (SIS) with a
modified-Schechter velocity dispersion function (VDF),
    phi(s) ds ∝ (s/s*)^a exp[-(s/s*)^b] ds/s,
normalisation dropped (only shapes are compared).
image separation dtheta = 2 theta_E = 8 pi (s/c)^2 D_ls/D_s.
Multiple-image optical depth per lens: pi theta_E^2 in the source plane, so
    dtau = phi(s) ds  pi theta_E^2  D_c(z_l)^2  dD_c(z_l)        (flat universe).
For an SIS the magnification distribution does not depend on s, so magnification
bias rescales the counts but leaves the shape of dP/d(dtheta) unchanged.
In comoving distances D_ls/D_s = 1 - x with x = D_l/D_s, so the shape is also
independent of z_s (Turner, Ostriker & Gott 1984); we use z_s = 2.

VDFs (checked against the papers' LaTeX source):
  Choi, Park & Vogeley 2007, early types: s* = 161, a = 2.32, b = 2.67
  Bernardi et al. 2010, all galaxies (Table, 'All'): s* = 113.78, a = 0.94, b = 1.85
LCDM halo case: dn/dln(V) ∝ V^-3, as derived by Zwaan et al. 2010 (arXiv:0912.1754,
Sec. 5: psi(M) ∝ M^-1.9 with M ∝ V^3.4) and found for Bolshoi V_max (Klypin et al.
2011, arXiv:1002.3660). Here every halo lenses as an SIS with sigma = V; the high-V
cutoff is borrowed from Choi+07. Its number density diverges as V -> 0; only the
lens-weighted integral converges (a > -4). It is not an upper limit on the
small-separation fraction: a steeper slope gives larger fractions. Observed
velocity functions are far flatter at low V (HIPASS: dn/dlnV ∝ V^(0.10±0.25)).
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

C_KMS = 299792.458
H0, OM = 70.0, 0.3
ARCSEC = np.pi / 180 / 3600

VDF = {
    "early types (Choi+07)": (161.0, 2.32, 2.67),
    "all galaxies (Bernardi+10)": (113.78, 0.94, 1.85),
    r"LCDM haloes, $dn/d\ln V\propto V^{-3}$ (Zwaan+10)": (161.0, -3.0, 2.67),
}

ZGRID = np.linspace(0, 6, 6001)
DC = np.concatenate([[0], np.cumsum(
    np.diff(ZGRID) * C_KMS / H0 / np.sqrt(OM * (1 + ZGRID[1:]) ** 3 + 1 - OM))])


def dc(z):
    return np.interp(z, ZGRID, DC)


def phi(s, sstar, a, b):
    """Shape of the VDF per unit s (normalisation irrelevant here)."""
    x = s / sstar
    return x**a * np.exp(-x**b) / s


def dp_dsep(sep, zs, vdf, nz=400):
    """Unnormalised dP/d(sep) for a source at zs; sep in arcsec."""
    zl = np.linspace(1e-3, zs - 1e-3, nz)
    dcl, dcs = dc(zl), dc(zs)
    ratio = (dcs - dcl) / dcs                         # D_ls / D_s, flat universe
    dDc_dz = C_KMS / H0 / np.sqrt(OM * (1 + zl) ** 3 + 1 - OM)
    out = np.zeros_like(sep)
    for i, th in enumerate(sep):
        s = C_KMS * np.sqrt(th * ARCSEC / (8 * np.pi * ratio))   # SIS sigma giving sep
        ds_dth = s / (2 * th)
        theta_e = th / 2 * ARCSEC
        out[i] = np.trapezoid(phi(s, *vdf) * ds_dth * np.pi * theta_e**2
                              * dcl**2 * dDc_dz, zl)
    return out


def norm(x, y, lo=0.0, hi=np.inf):
    m = (x >= lo) & (x <= hi)
    return y / np.trapezoid(y[m], x[m])


sep = np.logspace(-5, np.log10(6), 1500)  # reaches far below 0.02" for the rising case
pred = {name: norm(sep, dp_dsep(sep, 2.0, v)) for name, v in VDF.items()}

known = np.array([float(r["max_sep_arcsec"]) for r in
                  csv.DictReader(open(ROOT / "data" / "lensed_quasars.csv"))
                  if r["max_sep_arcsec"]])


def frac_below(x, y, cut):
    m = x < cut
    xs = np.append(x[m], cut)
    return np.trapezoid(np.append(y[m], np.interp(cut, x, y)), xs) / np.trapezoid(y, x)


print("fraction of the intrinsic population below 0.5\", 1\", 2\":")
for name, y in pred.items():
    print(f"  {name:28s}: " +
          "  ".join(f"{frac_below(sep, y, c):.2f}" for c in (0.5, 1, 2)))
print(f"known (GraL X): N={known.size}, <0.5\": {np.mean(known < 0.5):.2f}, "
      f"<1\": {np.mean(known < 1):.2f}, <2\": {np.mean(known < 2):.2f}")

# Compare shapes over 0.5-4", away from the poorly sampled small-separation end.
LO, HI = 0.5, 4.0
fig, ax = plt.subplots(figsize=(7.5, 4.2))
bins = np.arange(0.1, 6.01, 0.2)  # 0.5" is a bin edge
centre = 0.5 * (bins[1:] + bins[:-1])
h, _ = np.histogram(known, bins)
m = (centre >= LO) & (centre <= HI)
ax.stairs(h / (h[m].sum() * np.diff(bins)[0]), bins, color="k", lw=1.5,
          label="known lenses (GraL X)")
for (name, y), col in zip(pred.items(), ["C3", "C2", "C0"]):
    ax.plot(sep, norm(sep, y, LO, HI), color=col, lw=1.5, label=f"SIS + {name}")
ax.set_xlim(0, 5)
ax.set_ylim(0, 1.6)
ax.set_xlabel("maximum image separation [arcsec]")
ax.set_ylabel(r"$dP/d\Delta\theta$ (normalised over 0.5-4$''$)")
ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout()
fig.savefig(PLOTS / "predicted_separations.png", dpi=150)
