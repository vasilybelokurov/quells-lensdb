#!/bin/sh
# Download the MaNGA DynPop inputs used by manga_population.py into data/raw/manga/
# (~280 MB): JAM catalogue and MGEs (Zhu et al. 2023, Zenodo 17518315) and the
# SDSS DR17 drpall file (esweight volume weights).
set -e
cd "$(dirname "$0")"
D=data/raw/manga
mkdir -p "$D/mge"
Z=https://zenodo.org/api/records/17518315/files
curl -L -o "$D/SDSSDR17_MaNGA_JAM_v2.fits" "$Z/SDSSDR17_MaNGA_JAM_v2.fits/content"
curl -L -o "$D/SDSSDR17_MaNGA_MGE.zip" "$Z/SDSSDR17_MaNGA_MGE.zip/content"
curl -L -o "$D/drpall-v3_1_1.fits" https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/drpall-v3_1_1.fits
unzip -q -o "$D/SDSSDR17_MaNGA_MGE.zip" -d "$D/mge"
