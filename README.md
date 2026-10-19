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

## Method

All calculations use flat ΛCDM with $`H_0=70\,{\rm km\,s^{-1}\,Mpc^{-1}}`$ and
$`\Omega_m=0.3`$, a single source redshift $`z_s=2`$, spherical lenses and non-evolving
comoving number densities. Distances $`D_l`$, $`D_s`$, $`D_{ls}`$ are angular-diameter
distances; $`D_c`$ is the comoving distance.

### 1. Image-separation distributions

**Optical depth.** For a population of lenses with comoving number density $`n_i`$
(per galaxy template or velocity-dispersion bin $`i`$), the probability that a source at
$`z_s`$ is multiply imaged is

```math
\tau=\int_0^{z_s}dz_l\,\frac{c\,D_c^2(z_l)}{H(z_l)}\sum_i n_i\,A_i(z_l),
```

where $`A_i`$ is the angular multiple-imaging cross-section in the source plane. The
integral uses 39 lens redshifts between 0.05 and 1.95 ($`\Delta z=0.05`$). Each lens
contributes its image separations to a histogram, giving
$`d\tau/d\log_{10}\Delta\theta`$ in 60 logarithmic bins between $`0.01''`$ and $`20''`$.

**Lensing of one mass profile** (`lens_population.lens_properties`). Each lens is
specified by its enclosed 3D mass $`M(<r)`$ on a logarithmic grid of 900 radii between
$`10^{-3.5}`$ and $`10^{3.7}`$ kpc.

**(1)** Projection. The projected mass inside $`R`$ is a sum over spherical shells,

```math
M_{\rm 2D}(<R)=\sum_{\rm shells}\Delta M(r)\,f(R,r),
```

where a shell entirely inside the cylinder counts fully, $`f=1`$ for $`r\le R`$, and a
shell outside contributes the fraction $`f=1-\sqrt{1-R^2/r^2}`$ for $`r>R`$; the sum is
evaluated on 500 projected radii between $`10^{-3.3}`$ and $`10^{2.3}`$ kpc.

**(2)** Convergence. With $`\Sigma_{\rm crit}=\frac{c^2}{4\pi G}\frac{D_s}{D_lD_{ls}}`$, the
mean convergence is $`\bar\kappa(R)=M_{\rm 2D}(<R)/(\pi R^2\Sigma_{\rm crit})`$ and the
local one $`\kappa=\tfrac12\bar\kappa\,d\ln M_{\rm 2D}/d\ln R`$. On the sky,
$`\theta=R/D_l`$ and the deflection is $`\alpha(\theta)=\theta\,\bar\kappa`$.

**(3)** Cross-section. Along a line through the lens the lens equation is
$`\beta=g(\theta)\equiv\theta-\alpha(\theta)`$. A source is multiply imaged if
$`\beta<\beta_r\equiv\max_\theta[\alpha(\theta)-\theta]=-\min g`$; this is the radial
caustic for cored or cuspy profiles and the cut for a singular isothermal sphere
(SIS). The cross-section is $`A=\pi\beta_r^2`$.

**(4)** Separations. Sources are placed at equal-area positions
$`\beta_k=\beta_r\sqrt{(k+\tfrac12)/N}`$, $`N=8`$. For each, the two outer images are the
outermost roots of $`g(\theta)=+\beta_k`$ (image A, on the source's side) and
$`g(\theta)=-\beta_k`$ (image B, opposite), found by bracketing and linear
interpolation rather than by inverting $`g`$, which is not monotonic for some SPARC
profiles. The separation is $`\Delta\theta=\theta_A+\theta_B`$, and each sample carries
weight $`A/N`$.

**Populations** (`compare_separations.py`).

*Early types (E/S0).* Abundance from the SDSS velocity-dispersion function of
Choi et al. (2007),

```math
\frac{dn}{d\ln\sigma}=\phi_*\left(\frac{\sigma}{\sigma_*}\right)^{a}
e^{-(\sigma/\sigma_*)^{b}}\frac{b}{\Gamma(a/b)},
```

with $`\phi_*=8.0\times10^{-3}h^3\,{\rm Mpc^{-3}}`$ ($`h=0.7`$), $`\sigma_*=161\,{\rm km\,s^{-1}}`$,
$`a=2.32`$, $`b=2.67`$, on 160 logarithmic dispersions between 10 and 500 km/s. Each
galaxy is an SIS, $`M(<r)=2\sigma_{\rm SIS}^2r/G`$, with $`\sigma_{\rm SIS}=f\sigma_*`$.
The fiducial $`f=1`$ is tested by SLACS lenses for $`\sigma\gtrsim160`$ km/s
($`\sigma_{\rm SDSS}/\sigma_{\rm SIE}=0.946`$ for 74 lenses, Auger et al. 2009;
Bolton et al. 2008), and $`f=0.95`$–1.05 gives the calibration band. The population is
split at $`\sigma=160`$ km/s (counts and SIS mass both measured) and 70 km/s (below
which the counts are extrapolated; 0.2% of $`\tau`$, not plotted).

*Inner-slope variant.* For $`\sigma=70`$–160 km/s the SIS is replaced by
$`\rho\propto r^{-\gamma}`$ inside $`R_e`$ and isothermal outside, normalised so that
$`M(<R_e)=2\sigma^2R_e/G`$, the mass the stellar dispersion measures; $`\gamma=2`$
recovers the SIS. The hatched band in the main figure uses $`\gamma=1.6`$ and
$`R_e=2`$ kpc; `slope_test.py` covers $`\gamma=1.6`$–2.4 and $`R_e=1`$–3 kpc.

*HI-rich discs (Sa and later).* Each SPARC galaxy with quality flag 1 or 2,
inclination $`\ge30^\circ`$ and Hubble type $`T\ge1`$ (150 galaxies) is a spherical lens
with $`M(<r)=V_{\rm obs}^2(r)\,r/G`$. Inside the first measured radius $`r_0`$ the
rotation is solid-body, $`V=V_0\,r/r_0`$ (alternative: flat, $`V=V_0`$); beyond the last
point $`V`$ stays constant out to $`r_{200}=V/(10H_0)`$. The enclosed mass is forced to
be non-decreasing. Radii of galaxies with Hubble-flow distances are rescaled from
$`H_0=73`$ (SPARC) to 70. Abundances come from the HIPASS rotation-velocity function
(Zwaan et al. 2010; same functional form, $`\phi_*=0.061\,{\rm Mpc^{-3}}`$,
$`\log V_*=2.06`$, $`a=0.66`$, $`b=2.10`$): the number density in each 0.1-dex bin of
$`V_{\rm flat}`$ (or the maximum $`V_{\rm obs}`$ when SPARC gives no $`V_{\rm flat}`$) is
shared equally among the SPARC galaxies in that bin.

**Known lenses.** The 347 GraL X systems with a separation are histogrammed in 0.1-dex
bins on a separate axis, scaled so that data and the $`\sigma>160`$ km/s model contain
the same number of lenses at 2–6″.

**Checks.** The SIS optical depth reproduces the analytic result

```math
\tau=\frac{16\pi^3}{30}\,\phi_*\left(\frac{\sigma_*}{c}\right)^4
\frac{\Gamma[(a+4)/b]}{\Gamma(a/b)}\,D_c^3(z_s)
```

to 0.03%; SIS separations equal $`2\theta_E`$ and cross-sections $`\pi\theta_E^2`$ to 0.1%;
less than $`10^{-5}`$ of $`\tau`$ falls outside the histogram range.

### 2. Time-delay distributions

`delays.py` uses the same mass profiles, projection and lens equation, and adds the
lensing potential and image magnifications.

**(1)** Potential. For a spherical lens $`d\psi/d\theta=\alpha`$, so
$`\psi(\theta)=\int_0^\theta\alpha\,d\theta'`$, integrated by the trapezoidal rule on
the projected-radius grid (the constant of integration cancels in delays).

**(2)** Fermat potential and delay. With image A at $`+\theta_A`$ and image B at $`-\theta_B`$
on the line through the source,

```math
\Delta\phi=\left[\tfrac12(\theta_B+\beta)^2-\psi(\theta_B)\right]
          -\left[\tfrac12(\theta_A-\beta)^2-\psi(\theta_A)\right],
\qquad
\Delta t=\frac{D_{\Delta t}}{c}\,\Delta\phi,\qquad
D_{\Delta t}=(1+z_l)\frac{D_lD_s}{D_{ls}}.
```

**(3)** Flux ratio. The magnification at each image follows from
$`\mu^{-1}=(1-\bar\kappa)(1-2\kappa+\bar\kappa)`$, and a source counts as lensed only if
the fainter outer image has $`q=|\mu_{\rm faint}/\mu_{\rm bright}|\ge0.1`$, the
flux-ratio range of the QUELLS calibration grid. Without this cut, profiles with
$`\gamma>2`$ (divergent central deflection) formally image every source through
vanishingly faint central counter-images.

**(4)** Sampling. $`N=24`$ equal-area source positions per lens, with the sampled region
capped at $`2\theta_E`$; for an SIS ($`\beta_r=\theta_E`$) the cap has no effect.

**(5)** Output. A 2D histogram of optical depth in $`\log\Delta\theta`$ (46 bins,
0.05–10″) and $`\log\Delta t`$ (50 bins, 0.03–3000 d), for massive E/S0
($`\sigma>160`$ km/s, SIS), intermediate E/S0 ($`\sigma=70`$–160 km/s, SIS and
$`\gamma=1.6`$) and SPARC discs (solid-body centre). Delay distributions of
sub-arcsecond lenses are this histogram summed over $`0.1''<\Delta\theta<1''`$.

**Checks.** SIS delays reproduce
$`\Delta t=\frac{D_{\Delta t}}{2c}\Delta\theta^2\frac{1-q}{1+q}`$ to 1.5% (radial-grid
resolution). For power laws normalised to the same $`\theta_E`$, the code reproduces
$`\Delta t=\frac{D_{\Delta t}}{2c}(\theta_A^2-\theta_B^2)(\gamma-1)`$ to three decimals
for $`\gamma=1.6`$–2.4, the fixed-image-position form of the result that delays measure
the mean convergence in the annulus between the images (Kochanek 2002).

**Note on the flux cut.** The main separation figure (`compare_separations.py`) uses
no flux cut; the slope test and the delay calculation use $`q\ge0.1`$, so their absolute
optical depths are lower (by a factor $`[(1-q)/(1+q)]^2=0.67`$ for an SIS).

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

## Licence

Code: MIT (see `LICENSE`). Third-party data in `data/raw/` remain under their original terms; please cite the sources above.
