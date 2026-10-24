#!/bin/sh
# Regression runner: run every figure script of a quells-lensdb tree and log its stdout/stderr.
#   checks/run_all.sh <tree> <outdir>
# Run it on a copy of the tree (scripts write plots/ and MaNGA caches), e.g. before and after a
# refactor, then compare the two output directories with checks/compare_runs.py.
# The MaNGA scripts need data/raw/manga/manga_profiles.npz (from manga_population.py).
D="$1"; OUT="$2"
[ -n "$D" ] && [ -n "$OUT" ] || { echo "usage: $0 <tree> <outdir>"; exit 1; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
PY=${PY:-~/Work/venvs/.venv/bin/python}
cd "$D" || exit 1
: > "$OUT/_summary.txt"
for s in build_db plot_db compare_separations plot_compare_inputs sparc_lenses slope_test delays combine_figs \
         bh_test lens_population predict_sep plot_contributions plot_halo_models plot_obs_comparison \
         plot_observables manga_lenses manga_combined plot_manga_fractions plot_manga_fractions_delays; do
  f=$s.py; [ -f "$f" ] || f=scripts/$s.py
  t0=$(date +%s); MPLBACKEND=Agg $PY "$f" > "$OUT/$s.txt" 2> "$OUT/$s.err"; rc=$?; t1=$(date +%s)
  echo "$s rc=$rc $((t1 - t0))s" >> "$OUT/_summary.txt"
done
cat "$OUT/_summary.txt"
