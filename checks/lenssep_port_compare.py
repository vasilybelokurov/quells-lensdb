"""Effect of moving lens_small_separation onto lenscore (H0 = 70, Om = 0.3 instead of Planck18;
mean outer-image separation instead of 2 theta_E): compares its derived grids with those at
commit 8031155 (before the move), read from its git history.

Result (2026-10-24): SLACS median log(theta_E model/obs) shifts by <= 0.005 dex (alpha-nu valley
unchanged); separations are 0.98 x 2 theta_E (5-95%: 0.95-1.03); tau(<1") falls by 5-10%;
the shape S = tau(0.1-0.5")/tau(0.5-1") is unchanged (median 0.90).
    python checks/lenssep_port_compare.py
"""
import io
import subprocess
from pathlib import Path

import numpy as np

LS = Path.home() / "Work" / "Code" / "lens_small_separation"
OLD = "8031155"


def old(path):
    return np.load(io.BytesIO(subprocess.run(["git", "-C", str(LS), "show", f"{OLD}:{path}"],
                                             capture_output=True, check=True).stdout))


new = lambda path: np.load(LS / path)
d = new("data/derived/fig03_slacs_alpha_nu.npz")["med"] - old("data/derived/fig03_slacs_alpha_nu.npz")["med"]
print(f"fig03 SLACS median log ratio, new - old: mean {d.mean():+.4f}, range {d.min():+.4f} .. {d.max():+.4f}")
g = new("data/derived/forecast_grid.npz")
ok = g["thE"] > 0
r = g["sep"][ok] / (2 * g["thE"][ok])
print(f"mean separation / 2 theta_E: median {np.median(r):.3f}, 5-95% {np.percentile(r, 5):.3f}-{np.percentile(r, 95):.3f}")
n5, o5 = new("data/derived/fig05_alpha_nu_plane.npz"), old("data/derived/fig05_alpha_nu_plane.npz")
q = n5["tau1"] / o5["tau1"]
print(f"tau(<1\") new/old over the (alpha, nu) plane: median {np.median(q):.3f}, range {q.min():.3f}-{q.max():.3f}")
print(f"shape S: median new {np.median(n5['S']):.3f}, old {np.median(o5['S']):.3f}")
