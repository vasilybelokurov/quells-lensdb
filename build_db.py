"""Build a per-system table of known lensed quasars from Gaia GraL X.

Source: Ducourant et al. 2026, A&A 707, A345 (CDS J/A+A/707/A345).
tablea1.dat has one row per component (images A-F, lens galaxies G*, other S);
tableb1.dat has published time delays for 73 systems.

Output: data/lensed_quasars.csv, one row per lens system.
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"

# Flat LCDM used only for the time-delay distance column.
H0, OM = 70.0, 0.3
C_KMS = 299792.458
GPC_KM = 3.0857e22
DAY_S = 86400.0
ARCSEC = np.pi / 180 / 3600


def field(line, a, b):
    """Return the fixed-width field in 1-indexed inclusive bytes a-b, or None."""
    s = line[a - 1:b].strip()
    return None if s in ("", "---", "-") else s


def num(line, a, b):
    s = field(line, a, b)
    return None if s is None else float(s)


def comoving_mpc(z, n=2000):
    zz = np.linspace(0.0, z, n)
    return C_KMS / H0 * np.trapezoid(1.0 / np.sqrt(OM * (1 + zz) ** 3 + 1 - OM), zz)


def time_delay_distance_gpc(zl, zs):
    """d_dt = (1+zl) D_l D_s / D_ls for flat LCDM, in Gpc."""
    dcl, dcs = comoving_mpc(zl), comoving_mpc(zs)
    dl, ds, dls = dcl / (1 + zl), dcs / (1 + zs), (dcs - dcl) / (1 + zs)
    return (1 + zl) * dl * ds / dls / 1e3


def is_image(comp):
    return comp[0] in "ABCDEF"


def read_systems():
    systems = {}
    with open(RAW / "tablea1.dat") as f:
        for line in f:
            name = field(line, 1, 24)
            comp = field(line, 26, 27)
            s = systems.setdefault(name, {
                "name": name,
                "type": field(line, 57, 62),
                "max_sep_arcsec": num(line, 64, 72),
                "ra_center": num(line, 228, 242),
                "dec_center": num(line, 244, 258),
                "z_source": None,
                "z_lens": None,
                "discovery_author": field(line, 118, 136),
                "discovery_year": int(field(line, 138, 141)),
                "discovery_bibcode": field(line, 143, 161),
                "n_images": 0,
                "n_images_gaia_dr3": 0,
                "lens_galaxy_listed": 0,
                "corrected": None,
            })
            if s["z_source"] is None:
                s["z_source"] = num(line, 260, 265)
            if s["z_lens"] is None:
                s["z_lens"] = num(line, 267, 272)
            if is_image(comp):
                s["n_images"] += 1
                s["n_images_gaia_dr3"] += field(line, 529, 529) == "1"
            elif comp.startswith("G"):
                s["lens_galaxy_listed"] = 1
    return systems


def read_delays():
    """Largest |delay| over all published image pairs, per system."""
    pairs = [(54, 65), (74, 85), (93, 104), (113, 117), (125, 136), (144, 155)]
    out = {}
    with open(RAW / "tableb1.dat") as f:
        for line in f:
            vals = [num(line, a, b) for a, b in pairs]
            vals = [abs(v) for v in vals if v is not None]
            if vals:
                out[field(line, 1, 24)] = (max(vals), field(line, 181, 199))
    return out


def main():
    systems = read_systems()
    with open(ROOT / "data" / "corrections.csv") as f:
        for c in csv.DictReader(f):
            systems[c["name"]][c["column"]] = float(c["value"])
            systems[c["name"]]["corrected"] = c["column"]
    delays = read_delays()
    unmatched = set(delays) - set(systems)
    if unmatched:
        print("delay systems not in table A1:", sorted(unmatched))

    rows = []
    for s in systems.values():
        # 17 systems listed only as a single 'S' row carry MaxSep = 0: no separation.
        if not s["max_sep_arcsec"]:
            s["max_sep_arcsec"] = None
            rows.append(s | {"theta_E_proxy_arcsec": None, "max_abs_delay_d": None,
                             "delay_bibcode": None, "d_dt_gpc": None, "sis_delay_scale_d": None})
            continue
        s["theta_E_proxy_arcsec"] = s["max_sep_arcsec"] / 2
        d = delays.get(s["name"])
        s["max_abs_delay_d"] = d[0] if d else None
        s["delay_bibcode"] = d[1] if d else None
        zl, zs = s["z_lens"], s["z_source"]
        if zl and zs and zs > zl:
            ddt = time_delay_distance_gpc(zl, zs)
            s["d_dt_gpc"] = round(ddt, 3)
            # SIS scale (d_dt/2c) theta_sep^2: delay for a q -> 0 double, in days.
            s["sis_delay_scale_d"] = round(
                ddt * GPC_KM / (2 * C_KMS) * (s["max_sep_arcsec"] * ARCSEC) ** 2 / DAY_S, 2)
        else:
            s["d_dt_gpc"] = s["sis_delay_scale_d"] = None
        rows.append(s)

    rows.sort(key=lambda r: r["ra_center"])
    out = ROOT / "data" / "lensed_quasars.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} systems to {out}")


if __name__ == "__main__":
    main()
