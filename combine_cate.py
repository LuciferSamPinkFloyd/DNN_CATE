"""
combine_cate.py
===============
Audit the per-replication CSVs written by cate_esm.py, and optionally merge
them into one long-format file.

The audit is the point of this script.  Merging is optional and off by
default, because cate_postanalysis.py streams and can read the per-replication
files directly:

    python cate_postanalysis.py --in "results/cate_c*_a0.85_J100_m*_*.csv" \
        --J 100 --alpha 0.85 --ms 1,2,3,4,6,J --out_dir figs/J100_a085

SPEED.  Every per-replication file is exactly one (case, m, replication) and
its diag_* columns are constant within it, so the diagnostics are read from
row 1 of each file rather than from a groupby over the concatenated frame.
Coverage is accumulated in chunks.  Nothing is concatenated unless --out is
given, and then the output is appended file by file rather than held in
memory.

Usage
-----
    # audit one cell, write nothing
    python combine_cate.py --in_dir results --J 100 --alpha 0.85 \
        --expect_reps 200

    # audit and also merge
    python combine_cate.py --in_dir results --J 100 --alpha 0.85 --m 4 \
        --out combined/cate_J100_a0.85_m4.csv
"""

import os
import re
import glob
import time
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd


VAR_RATIO_THRESH = {1: 200.0, 2: 200.0, 3: 900.0}

THRESH = {
    "var_ratio_phi"    : ("<", 200.0, "AIPW noise vs CATE signal"),
    "e_clip_frac"      : ("<",   0.05, "propensity truncation activity"),
    "G_min_last_t"     : (">",   0.10, "censoring survival at last t"),
    "G_clip_frac"      : ("<",   0.05, "censoring truncation activity"),
    "at_risk_frac_last": (">",   0.15, "at-risk mass at last t"),
    "censor_rate"      : ("<",   0.60, "overall censoring"),
}

CELL = ["case", "alpha", "J", "m"]
UNIT = ["idx_a", "idx_b", "t_idx_a", "t_idx_b"]
COVCOLS = ["estimand", "case", "alpha", "J", "m", "rep_id",
           "idx_a", "idx_b", "t_idx_a", "t_idx_b",
           "est", "se", "ci_lo", "ci_hi", "covered", "rho"]

# the optional _w<scale> segment appears when --width_scale is not 'fixed'
FNAME = re.compile(
    r"cate_c(\d)_a([\d.]+)_J(\d+)_m(\d+)_h(\d+)x(\d+)(?:_w([a-z_]+))?"
    r"_B(\d+)_rep(\d+)\.csv")


def parse_tag(fname):
    m = FNAME.search(os.path.basename(fname))
    if m is None:
        return None
    return {"case": int(m.group(1)), "alpha": float(m.group(2)),
            "J": int(m.group(3)),    "m": int(m.group(4)),
            "h1": int(m.group(5)),   "h2": int(m.group(6)),
            "width_scale": m.group(7) or "fixed",
            "B": int(m.group(8)),    "rep_id": int(m.group(9))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, default="results")
    p.add_argument("--out", type=str, default=None,
                   help="optional merged CSV; omit it and point "
                        "cate_postanalysis.py at --in_dir instead")
    p.add_argument("--case",  type=int,   default=None, choices=[1, 2, 3])
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--J",     type=int,   default=None)
    p.add_argument("--m",     type=int,   default=None)
    p.add_argument("--width_scale", type=str, default=None,
                   help="keep one architecture scaling, e.g. fixed or sqrt_m")
    p.add_argument("--expect_reps", type=int, default=None)
    p.add_argument("--chunksize", type=int, default=500_000)
    args = p.parse_args()

    t0 = time.time()
    files = sorted(glob.glob(os.path.join(args.in_dir, "cate_*.csv")))
    if not files:
        raise FileNotFoundError(f"No cate_*.csv found in '{args.in_dir}'")

    keep, skipped = [], 0
    for f in files:
        tag = parse_tag(f)
        if tag is None:
            skipped += 1
            continue
        if args.case is not None and tag["case"] != args.case:
            continue
        if args.alpha is not None and abs(tag["alpha"] - args.alpha) > 1e-9:
            continue
        if args.J is not None and tag["J"] != args.J:
            continue
        if args.m is not None and tag["m"] != args.m:
            continue
        if args.width_scale is not None and tag["width_scale"] != args.width_scale:
            continue
        keep.append((f, tag))

    print("=" * 78)
    print(f"  '{args.in_dir}': {len(files)} file(s), {len(keep)} match the filters")
    if skipped:
        print(f"  WARNING: {skipped} file(s) did not match the naming pattern")
    print("=" * 78)
    if not keep:
        raise ValueError("No files matched the requested filters")

    # ---- diagnostics: row 1 of each file ---------------------------------
    # each file is one replication and the diag_* columns are constant within
    # it, so no groupby over the merged frame is needed
    heads = []
    for f, tag in keep:
        try:
            h = pd.read_csv(f, nrows=1)
        except Exception as exc:
            print(f"  WARNING: cannot read {os.path.basename(f)}: {exc}")
            continue
        h["width_scale"] = tag["width_scale"]
        heads.append(h)
    rep_lvl = pd.concat(heads, ignore_index=True)
    print(f"  diagnostics read in {time.time() - t0:.1f}s")

    print("\n" + "=" * 78)
    print("  CELL COMPLETENESS")
    print("=" * 78)
    cells = (rep_lvl.groupby(CELL + ["width_scale"])["rep_id"]
             .nunique().reset_index(name="n_reps"))
    print(f"  {'case':>5} {'alpha':>6} {'J':>5} {'m':>4} {'scale':>8} "
          f"{'n_reps':>8}  status")
    print(f"  {'-'*5} {'-'*6} {'-'*5} {'-'*4} {'-'*8} {'-'*8}  {'-'*18}")
    short = 0
    for _, r in cells.iterrows():
        status = "OK"
        if args.expect_reps is not None and r["n_reps"] < args.expect_reps:
            status = f"MISSING {args.expect_reps - int(r['n_reps'])}"
            short += 1
        print(f"  {int(r['case']):>5} {r['alpha']:>6.2f} {int(r['J']):>5} "
              f"{int(r['m']):>4} {r['width_scale']:>8} "
              f"{int(r['n_reps']):>8}  {status}")
    if short:
        print(f"\n  >>> {short} cell(s) short of --expect_reps")

    print("\n" + "=" * 78)
    print("  PITFALL AUDIT   (one row per replication)")
    print("=" * 78)
    breach = False
    for key, (direction, thresh, desc) in THRESH.items():
        col = f"diag_{key}"
        if col not in rep_lvl.columns:
            continue
        v = rep_lvl[col].dropna()
        if len(v) == 0:
            continue
        if key == "var_ratio_phi":
            thr = rep_lvl.loc[v.index, "case"].map(VAR_RATIO_THRESH).fillna(200.0)
            bad, rule = v > thr, "want < " + "/".join(
                f"{VAR_RATIO_THRESH[c]:.0f}(case {c})"
                for c in sorted(rep_lvl["case"].unique()))
        elif direction == "<":
            bad, rule = v > thresh, f"want < {thresh}"
        else:
            bad, rule = v < thresh, f"want > {thresh}"
        frac = float(bad.mean())
        breach |= frac >= 0.05
        print(f"  [{'OK ' if frac < 0.05 else 'FLAG'}] {key:<20s} {desc}")
        print(f"         mean={v.mean():10.4f}  median={v.median():10.4f}  "
              f"min={v.min():10.4f}  max={v.max():10.4f}")
        print(f"         {rule};  breached in {100*frac:5.1f}% of reps")

    if "diag_basis_rank" in rep_lvl.columns:
        bad = rep_lvl[rep_lvl["diag_basis_rank"] < rep_lvl["m"]]
        if len(bad):
            breach = True
            print(f"  [FLAG] readout rank deficient in {len(bad)} rep(s)")
        else:
            print("  [OK ] basis_rank            full column rank everywhere")
    if "n_par" in rep_lvl.columns:
        g = rep_lvl.groupby("m")["n_par"].first()
        print("  [   ] trainable weights by m: "
              + ", ".join(f"m={int(k)}: {int(v):,}" for k, v in g.items()))

    # ---- coverage, streamed ----------------------------------------------
    print("\n" + "=" * 78)
    print("  HEADLINE COVERAGE   (nominal 0.95)")
    print("=" * 78)
    acc, uacc, rho = {}, {}, {}
    out_path, wrote_header = args.out, False
    if out_path:
        d = os.path.dirname(os.path.abspath(out_path))
        if d:
            os.makedirs(d, exist_ok=True)
        if os.path.exists(out_path):
            os.remove(out_path)

    for f, tag in keep:
        head = pd.read_csv(f, nrows=0)
        cols = [c for c in COVCOLS if c in head.columns]
        for d in pd.read_csv(f, usecols=cols, chunksize=args.chunksize):
            k = ["estimand"] + CELL
            g = d.groupby(k, observed=True)
            a = g.agg(n=("est", "size"), s_cov=("covered", "sum"),
                      s_se=("se", "sum"),
                      s_ail=("ci_hi", "sum"), s_lo=("ci_lo", "sum"))
            acc["a"] = a if "a" not in acc else acc["a"].add(a, fill_value=0.0)
            u = d.groupby(k + UNIT, observed=True).agg(
                n=("est", "size"), s=("est", "sum"),
                ss=("est", lambda x: float((x ** 2).sum())))
            uacc["u"] = u if "u" not in uacc else uacc["u"].add(u, fill_value=0)
            r = d[d["rho"].notna()]
            if len(r):
                rr = r.assign(_a=r["rho"].abs()).groupby(k, observed=True).agg(
                    n=("_a", "size"), s=("_a", "sum"),
                    hi=("_a", lambda x: float((x > 0.99).sum())))
                rho["r"] = rr if "r" not in rho else rho["r"].add(rr, fill_value=0)
        if out_path:
            full = pd.read_csv(f)
            full.to_csv(out_path, mode="a", index=False, header=not wrote_header)
            wrote_header = True

    a = acc["a"]
    u = uacc["u"].reset_index()
    nn = u["n"].clip(lower=2)
    u["emp"] = np.sqrt((((u["ss"] - u["s"] ** 2 / u["n"]) / (nn - 1))).clip(lower=0))
    emp = u.groupby(["estimand"] + CELL, observed=True)["emp"].mean()

    print(f"  {'case':>5} {'alpha':>6} {'J':>5} {'m':>4} {'estimand':>9} "
          f"{'CP':>8} {'SE/EmpSD':>9} {'AIL':>10}")
    print(f"  {'-'*5} {'-'*6} {'-'*5} {'-'*4} {'-'*9} {'-'*8} {'-'*9} {'-'*10}")
    for idx, r in a.iterrows():
        est, cs, al, J, m = idx
        e = emp.get(idx, np.nan)
        se = r["s_se"] / r["n"]
        print(f"  {int(cs):>5} {al:>6.2f} {int(J):>5} {int(m):>4} {est:>9} "
              f"{r['s_cov']/r['n']:>8.4f} "
              f"{(se/e if e and e > 0 else np.nan):>9.4f} "
              f"{(r['s_ail']-r['s_lo'])/r['n']:>10.5f}")

    if "r" in rho:
        print("\n  CONTRAST NON-DEGENERACY   (|rho| must stay below one)")
        print(f"  {'case':>5} {'m':>4} {'estimand':>9} {'mean|rho|':>10} "
              f"{'frac>0.99':>10}")
        for idx, r in rho["r"].iterrows():
            est, cs, al, J, m = idx
            fr = r["hi"] / r["n"]
            flag = "  <-- near-degenerate" if fr > 0.05 else ""
            print(f"  {int(cs):>5} {int(m):>4} {est:>9} "
                  f"{r['s']/r['n']:>10.4f} {fr:>10.4f}{flag}")
            breach |= fr > 0.05

    print("\n  " + ("Some diagnostics breached; results for those cells are "
                    "not trustworthy." if breach
                    else "All diagnostics within tolerance."))
    print(f"\n  elapsed {time.time() - t0:.1f}s")
    if out_path:
        print(f"  merged CSV written to {out_path}")
    else:
        print("  no merged CSV written.  Point cate_postanalysis.py at "
              f"--in \"{args.in_dir}/cate_*.csv\" directly.")


if __name__ == "__main__":
    main()
