"""Galaxy-scale lens samples used to test the halo model.

SLACS: Auger et al. 2009 (ApJ 705, 1099; CDS J/ApJ/705/1099), tables 3 and 4.
SL2S:  Sonnenfeld et al. 2013 (ApJ 777, 97; CDS J/ApJ/777/97), tables 4-7, grade A only.
Stellar masses are for a Chabrier IMF unless stated; R_e is in kpc.
"""
import numpy as np

from lenscore import cosmology as cosmo
from paths import RAW


def _f(line, a, b):
    s = line[a - 1:b].strip()
    return float(s) if s else np.nan


def _kpc_per_arcsec(z):
    return cosmo.dist_mpc(z) / (1 + z) * 1e3 * cosmo.ARCSEC


def slacs():
    t3 = {}
    for line in open(RAW / "slacs" / "table3.dat"):
        name = line[4:14].strip()
        re_arcsec = _f(line, 96, 99)                 # I band, else V, else B
        if np.isnan(re_arcsec):
            re_arcsec = _f(line, 85, 88)
        if np.isnan(re_arcsec):
            re_arcsec = _f(line, 72, 75)
        t3[name] = dict(zl=_f(line, 40, 44), zs=_f(line, 46, 50), sigma=_f(line, 52, 54),
                        re_arcsec=re_arcsec, mtype=line[58:63].strip())
    rows = []
    for line in open(RAW / "slacs" / "table4.dat"):
        name = line[4:14].strip()
        d = dict(name=name, **t3[name])
        d.update(rein_kpc=_f(line, 16, 19), log_mein=_f(line, 21, 25),
                 fstar_chab=_f(line, 27, 30), e_fstar_chab=_f(line, 32, 35),
                 fstar_salp=_f(line, 37, 40), e_fstar_salp=_f(line, 42, 45),
                 logms=_f(line, 52, 56), e_logms=_f(line, 58, 61))
        d["re_kpc"] = d["re_arcsec"] * _kpc_per_arcsec(d["zl"])
        rows.append(d)
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}


def sl2s():
    re = {}
    for fn in ("table4.dat", "table5.dat"):            # HST (table 5) overrides CFHT
        for line in open(RAW / "sl2s" / fn):
            if line[18:20].strip():
                continue                               # second component
            re[line[4:18].strip()] = _f(line, 22, 25)
    grade_a = {line[4:18].strip() for line in open(RAW / "sl2s" / "table6.dat")
               if line[59:60] == "A"}                 # confirmed lenses only
    rows = []
    for line in open(RAW / "sl2s" / "table7.dat"):
        name = line[4:18].strip()
        z, lm = _f(line, 20, 24), _f(line, 26, 30)
        if name in re and name in grade_a and np.isfinite(z) and np.isfinite(lm):
            rows.append((z, lm, re[name] * _kpc_per_arcsec(z)))
    z, lm, rk = map(np.array, zip(*rows))
    return dict(zl=z, logms=lm, re_kpc=rk)


def auger2010():
    """SLACS X (Auger et al. 2010, ApJ 724, 511) Table 2, from the arXiv source:
    r_e (V, kpc), log M_* (Chabrier, Salpeter), projected f_DM inside r_e/2, z_l."""
    zl = dict(zip(slacs()["name"], slacs()["zl"]))
    rows = []
    for line in open(RAW / "slacs" / "auger2010_T_lens_data.tex"):
        if not line.startswith("SDSSJ"):
            continue
        cells = [c.strip() for c in line.split("&")]
        name = (cells[0].replace("$-$", "-").replace("$+$", "+").replace("\\dagger", "")
                .replace("$", "").replace("SDSS", "").strip())
        val = lambda c: float(c.replace("$", "").replace("\\phantom{0}", "").replace("\\phantom{-}", "")
                              .split("\\pm")[0])
        err = lambda c: float(c.replace("$", "").replace("\\\\", "").split("\\pm")[1])
        rows.append((zl.get(name, np.nan), val(cells[1]), val(cells[3]), val(cells[4]),
                     val(cells[6]), err(cells[6]), val(cells[7]), err(cells[7])))
    keys = ["zl", "re_kpc", "logms_chab", "logms_salp", "fdm_chab", "e_fdm_chab",
            "fdm_salp", "e_fdm_salp"]
    return {k: np.array(v) for k, v in zip(keys, zip(*rows))}


def read2019():
    """Read, Walker & Steger 2019 Table 1 (8 dSph, 8 dIrr): M_*, R_1/2, M200, rho_DM(150 pc)."""
    import csv
    lines = [l for l in open(RAW / "dwarfs" / "read2019_table1.csv") if not l.startswith("#")]
    rows = list(csv.DictReader(lines))
    d = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    for k in d:
        if k not in ("name", "type"):
            d[k] = d[k].astype(float)
    return dict(name=d["name"], dsph=d["type"] == "dSph", logms=np.log10(d["mstar_1e6"] * 1e6),
                re_kpc=d["r_half_kpc"], logm200=np.log10(d["m200_1e9"] * 1e9),
                rho150=d["rho150_1e8"] * 1e8 / 1e9,                       # Msun / pc^3
                e_rho150=np.vstack([d["e_rho_lo"], d["e_rho_hi"]]) * 1e8 / 1e9)


SPARC_UPS_DISK, SPARC_UPS_BUL = 0.5, 0.7         # [3.6] mass-to-light ratios, Lelli+16


def sparc():
    """SPARC (Lelli et al. 2016): Q = 1, 2 and i > 30 deg. M_* = 0.5 L[3.6] (bulges
    ignored in M_*); 3D dark-matter fraction at R_eff from the rotation-curve
    decomposition, f_DM = 1 - V_bar^2 / V_obs^2 (spherical approximation)."""
    rows = []
    for line in open(RAW / "dwarfs" / "SPARC_Lelli2016c.mrt"):
        # Data rows do not follow the header's byte positions; they are whitespace
        # separated: name T D e_D f_D Inc e_Inc L e_L Reff SBeff Rdisk SBdisk MHI RHI
        # Vflat e_Vflat Q Ref.
        w = line.split()
        if len(w) < 18 or not w[17].isdigit() or not w[1].isdigit():
            continue
        name, inc, lum, reff, q = w[0], float(w[5]), float(w[7]), float(w[9]), int(w[17])
        if q > 2 or inc < 30:
            continue
        rc = np.loadtxt(RAW / "dwarfs" / "rotmod" / f"{name}_rotmod.dat")
        if rc.ndim < 2 or reff < rc[0, 0] or reff > rc[-1, 0]:
            continue                                   # R_eff outside the rotation curve
        r, vobs, vgas, vd, vb = rc[:, 0], rc[:, 1], rc[:, 3], rc[:, 4], rc[:, 5]
        vbar2 = vgas * np.abs(vgas) + SPARC_UPS_DISK * vd**2 + SPARC_UPS_BUL * vb**2
        f = 1 - np.interp(reff, r, vbar2) / np.interp(reff, r, vobs) ** 2
        rows.append((np.log10(SPARC_UPS_DISK * lum * 1e9), reff, f))
    lm, re, f = map(np.array, zip(*rows))
    return dict(logms=lm, re_kpc=re, fdm3d_re=f)


if __name__ == "__main__":
    s, t = slacs(), sl2s()
    ok = np.isfinite(s["logms"]) & np.isfinite(s["rein_kpc"]) & np.isfinite(s["re_kpc"])
    print(f"SLACS: {len(s['name'])} lenses, {ok.sum()} with M_*, R_Ein and R_e; "
          f"z_l {np.nanmin(s['zl']):.2f}-{np.nanmax(s['zl']):.2f}; "
          f"median log M_* {np.nanmedian(s['logms']):.2f}, R_e {np.nanmedian(s['re_kpc']):.1f} kpc, "
          f"f_*,Ein Chab {np.nanmedian(s['fstar_chab']):.2f} Salp {np.nanmedian(s['fstar_salp']):.2f}")
    print(f"SL2S: {len(t['zl'])} lenses with z, M_*, R_e; z_l {t['zl'].min():.2f}-{t['zl'].max():.2f}; "
          f"median log M_* {np.median(t['logms']):.2f}, R_e {np.median(t['re_kpc']):.1f} kpc")
