# quells-lensdb

Which galaxies make sub-arcsecond quasar lenses, and what their time delays look like.
Supporting calculations for the introduction of the QUELLS paper (variability-selected
lensed quasars in unresolved survey light curves).

The working note `notes/lens_separations_note.pdf` describes the method and results.
In short: the intrinsic image-separation distribution is built from measured galaxy
populations and compared with the 347 known lensed quasars of Gaia GraL X. Above 1"
the prediction rests on measurements; below it, early types with sigma = 70-160 km/s,
whose lensing mass profiles are untested, take a growing share, and their abundance
and time delays both respond to the inner density slope.

## Running

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python build_db.py            # known lenses -> data/lensed_quasars.csv
    .venv/bin/python compare_separations.py # main figure: plots/separations_compare.png

Every script writes PNGs to `plots/` and prints the numbers quoted in the note.

## Scripts

| script | what it does |
|---|---|
| `build_db.py`, `plot_db.py` | per-system table of known lensed quasars (GraL X) and its figures |
| `lens_population.py` | spherical lens code (projection, cross-section, image separations) and the NFW halo model |
| `compare_separations.py` | main result: E/S0 (Choi+07 + SIS) and HI-rich discs (SPARC + HIPASS) against known lenses |
| `plot_compare_inputs.py` | the inputs behind the main result (abundances, mass structure, SIS calibration) |
| `sparc_lenses.py` | SPARC rotation curves as empirical disc lenses |
| `slope_test.py` | sensitivity of the sub-0.5" abundance to the inner total-density slope |
| `delays.py` | joint separation / time-delay distribution from the Fermat potential |
| `combine_figs.py` | three-panel figure combining separations and delays |
| `obs_lenses.py` | loaders for SLACS, SL2S, Read+19 dwarfs and SPARC |
| `bh_test.py` | effect of central black holes on disc lenses |
| `predict_sep.py`, `plot_contributions.py`, `plot_observables.py`, `plot_halo_models.py`, `plot_obs_comparison.py` | earlier exploratory figures (SIS shapes, halo-model diagnostics) |

## Data

All inputs are public; please cite the original papers.

| file | source |
|---|---|
| `data/raw/tablea1.dat`, `tableb1.dat`, `ReadMe` | Gaia GraL X, Ducourant et al. 2026, A&A 707, A345 (CDS J/A+A/707/A345) |
| `data/raw/slacs/` | SLACS, Auger et al. 2009, ApJ 705, 1099 (CDS J/ApJ/705/1099); `auger2010_T_lens_data.tex` from Auger et al. 2010, ApJ 724, 511 (arXiv source) |
| `data/raw/sl2s/` | SL2S, Sonnenfeld et al. 2013, ApJ 777, 97 (CDS J/ApJ/777/97) |
| `data/raw/dwarfs/SPARC_Lelli2016c.mrt`, `rotmod/` | SPARC, Lelli, McGaugh & Schombert 2016, AJ 152, 157 (astroweb.case.edu/SPARC) |
| `data/raw/dwarfs/read2019_table1.csv` | Read, Walker & Steger 2019, MNRAS 484, 1401, Table 1 (transcribed from the arXiv source) |
| `data/corrections.csv` | B0218+357 separation corrected to the VLBI value (0.334") |

`data/lensed_quasars.csv` is derived by `build_db.py`: one row per system with maximum
image separation, Einstein-radius proxy (max_sep/2), redshifts, discovery reference,
Gaia DR3 matches, the largest published delay, the time-delay distance (flat LCDM,
H0 = 70, Om = 0.3) and the SIS delay scale. Seventeen systems listed without components
have no separation; separations of systems with fewer Gaia detections than images should
be checked before quoting individual values.

## Assumptions

Non-evolving galaxy populations, spherical lenses, no magnification bias, source at
z_s = 2, flat LCDM with H0 = 70 and Om = 0.3. See the note for caveats.
