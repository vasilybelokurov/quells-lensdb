"""Compare two runs of checks/run_all.sh: printed outputs, MaNGA arrays and figures.

    python checks/compare_runs.py <treeA> <outA> <treeB> <outB> [--diff-png NAME]

Reports scripts whose stdout differs, the maximum relative difference of the MaNGA
histograms, and the fraction of differing pixels per figure (threshold 0.02). With
--diff-png NAME, writes plots/diff_NAME (differing pixels in red) for one figure.
Used for the lenscore refactor (2026-10-24): all printed numbers identical, arrays equal
to 1e-15, figures pixel-identical except ~4e-6 of the pixels of separations_manga.png.
"""
import argparse
import difflib
from pathlib import Path

import numpy as np
from matplotlib.image import imread, imsave

p = argparse.ArgumentParser()
p.add_argument("tree_a"); p.add_argument("out_a"); p.add_argument("tree_b"); p.add_argument("out_b")
p.add_argument("--diff-png")
a = p.parse_args()
ta, oa, tb, ob = map(Path, (a.tree_a, a.out_a, a.tree_b, a.out_b))

print("== stdout")
for f in sorted(oa.glob("*.txt")):
    if f.name == "_summary.txt":
        continue
    x, y = f.read_text().splitlines(), (ob / f.name).read_text().splitlines()
    x = [l.replace(str(ta), "<tree>") for l in x]
    y = [l.replace(str(tb), "<tree>") for l in y]
    if x != y:
        print(f"-- {f.name}")
        print("\n".join(list(difflib.unified_diff(x, y, lineterm="", n=0))[:20]))
for f in sorted(ob.glob("*.err")):
    if "Traceback" in f.read_text():
        print(f"!! traceback in {f.name}")

print("== MaNGA arrays")
m = "data/raw/manga/"
for name in ("manga_sep_hist.npy", "manga_delay_hist.npz", "manga_delay_hist_types.npz"):
    fa, fb = ta / m / name, tb / m / name
    if not (fa.exists() and fb.exists()):
        print(f"{name}: missing in one run")
        continue
    if name.endswith(".npy"):
        x, y = np.load(fa), np.load(fb)
        print(f"{name}: max rel diff {np.abs(x - y).max() / np.abs(x).max():.1e}")
        continue
    x, y = np.load(fa), np.load(fb)
    kx = sorted(k for k in x if k != "key")
    ky = sorted(k for k in y if k != "key")
    if len(kx) == 1 and len(ky) > 1:                  # old single-array layout vs one array per class
        xs, ys = list(x[kx[0]]), [y[k] for k in ky]
    else:
        xs, ys = [x[k] for k in kx], [y[k] for k in ky]
    for i, (u, v) in enumerate(zip(xs, ys)):
        print(f"{name}[{i}]: max rel diff {np.abs(u - v).max() / max(np.abs(u).max(), 1e-300):.1e}")

print("== figures (fraction of pixels differing by > 0.02)")
for f in sorted((ta / "plots").glob("*.png")):
    g = tb / "plots" / f.name
    if not g.exists():
        print(f"{f.name}: missing in B")
        continue
    x, y = imread(f), imread(g)
    if x.shape != y.shape:
        print(f"{f.name}: size {x.shape[:2]} vs {y.shape[:2]}")
        continue
    frac = (np.abs(x - y).max(axis=-1) > 0.02).mean()
    if frac > 0:
        print(f"{f.name}: {frac:.2e}")
    if a.diff_png == f.name:
        d = (np.abs(x - y).max(axis=-1) > 0.02)[..., None]
        out = Path(__file__).resolve().parents[1] / "plots" / f"diff_{f.name}"
        imsave(out, np.where(d, [1.0, 0, 0], 0.6 * y[..., :3]))
        print(f"   wrote {out}")
