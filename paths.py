"""Repository paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)
KNOWN = ROOT / "data" / "lensed_quasars.csv"
