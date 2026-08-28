"""
cate_alpha.py
==============
Coverage, accuracy and variance calibration against the subsample exponent
alpha, evaluated across MULTIPLE designs, ONE grid size, ONE basis dimension 
and the pointwise CATE only.

    python cate_alpha.py --in_dir results --J 100 --m 4 --cases 1 2 3 \
        --out_dir figs/alpha --tag J100_m4_all

Produces
    fig_alpha_bias_se_<tag>.pdf  |Bias| / SE against alpha
    fig_alpha_cp_<tag>.pdf       coverage against alpha
    fig_alpha_all_<tag>.pdf      the two panels side by side
"""

import os
import re
import glob
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USECOLS = ["estimand", "case", "m", "J", "alpha", "rep_id",
           "idx_a", "idx_b", "t_idx_a", "t_idx_b",
           "target", "est", "se", "ci_lo", "ci_hi", "covered"]
UNIT = ["idx_a", "idx_b", "t_idx_a", "t_idx_b"]
FNAME = re.compile(r"cate_c(\d)_a([\d.]+)_J(\d+)_m(\d+)_h(\d+)x(\d+)"
                   r"(?:_w([a-z_]+))?_B(\d+)_rep(\d+)\.csv")

TINY = 1e-6          # a standard error this small is numerically degenerate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, default="results")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--J", type=int, required=True)
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--cases", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    p.add_argument("--width_scale", type=str, default="fixed")
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--split", action="store_true",
                   help="also write the three panels as separate PDFs")
    args = p.parse_args()

    out_dict = {}

    for case in args.cases:
        keep = []
        for f in sorted(glob.glob(os.path.join(args.in_dir, "cate_*.csv"))):
            g = FNAME.search(os.path.basename(f))
            if g is None:
                continue
            if (int(g.group(1)) != case or int(g.group(3)) != args.J
                    or int(g.group(4)) != args.m
                    or (g.group(7) or "fixed") != args.width_scale):
                continue
            keep.append((f, float(g.group(2))))
        
        if not keep:
            print(f"Warning: no files matched for case {case}. Skipping.")
            continue

        alphas = sorted({a for _, a in keep})
        print(f"design {case}, J = {args.J}, m = {args.m}, "
              f"alpha in {alphas}, {len(keep)} files")

        cell, unit = {}, {}
        for f, _ in keep:
            head = pd.read_csv(f, nrows=0)
            cols = [c for c in USECOLS if c in head.columns]
            for d in pd.read_csv(f, usecols=cols, chunksize=args.chunksize):
                d = d[d["estimand"] == "point"]
                if len(d) == 0:
                    continue
                err = d["est"] - d["target"]
                d = d.assign(_e=err, _a=err.abs(),
                             _ail=d["ci_hi"] - d["ci_lo"],
                             _e2=d["est"] ** 2,
                             _tiny=(d["se"] < TINY).astype(float))
                g = d.groupby("alpha")
                c = g.agg(n=("est", "size"), s_e=("_e", "sum"), s_a=("_a", "sum"),
                          s_cov=("covered", "sum"), s_se=("se", "sum"),
                          s_ail=("_ail", "sum"), n_tiny=("_tiny", "sum"),
                          se_min=("se", "min"))
                if "c" not in cell:
                    cell["c"] = c
                else:
                    prev = cell["c"]
                    mn = pd.concat([prev["se_min"], c["se_min"]], axis=1).min(axis=1)
                    cell["c"] = prev.drop(columns="se_min").add(
                        c.drop(columns="se_min"), fill_value=0.0).assign(se_min=mn)
                u = d.groupby(["alpha"] + UNIT).agg(
                    n=("est", "size"), s=("est", "sum"), ss=("_e2", "sum"))
                unit["u"] = u if "u" not in unit else unit["u"].add(u, fill_value=0)

        c = cell["c"]
        u = unit["u"].reset_index()
        nn = u["n"].clip(lower=2)
        u["emp"] = np.sqrt((((u["ss"] - u["s"] ** 2 / u["n"]) / (nn - 1))).clip(lower=0))
        emp = u.groupby("alpha")["emp"].mean()

        out = pd.DataFrame({
            "alpha": c.index,
            "Bias": (c["s_e"] / c["n"]).values,
            "MAE":  (c["s_a"] / c["n"]).values,
            "EmpSD": emp.reindex(c.index).values,
            "SE":   (c["s_se"] / c["n"]).values,
            "CP":   (c["s_cov"] / c["n"]).values,
            "AIL":  (c["s_ail"] / c["n"]).values,
            "se_min": c["se_min"].values,
            "frac_se_tiny": (c["n_tiny"] / c["n"]).values,
        }).sort_values("alpha").reset_index(drop=True)
        
        out["Ratio"] = out["SE"] / out["EmpSD"]
        out["AbsBias_over_SE"] = np.abs(out["Bias"]) / out["SE"]
        
        out_dict[case] = out

        os.makedirs(args.out_dir, exist_ok=True)
        tag_case = args.tag or f"J{args.J}_m{args.m}_c{case}"
        out.to_csv(os.path.join(args.out_dir, f"alpha_metrics_{tag_case}.csv"), index=False)

        with pd.option_context("display.width", 150):
            print(f"\nMetrics for Case {case}:")
            print(out.round(5).to_string(index=False))
            
        bad = out[out["frac_se_tiny"] > 0]
        if len(bad):
            print("\n  degenerate standard errors detected "
                  f"(se < {TINY:g}); the average ratio can look calibrated while "
                  "individual points are broken:")
            for _, r_ in bad.iterrows():
                print(f"    alpha={r_['alpha']:.2f}  "
                      f"frac={r_['frac_se_tiny']:.5f}  min se={r_['se_min']:.3e}")

    if not out_dict:
        print("No valid cases were processed. Exiting.")
        return
    
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    labels = {1: "Design 1", 2: "Design 2", 3: "Design 3"}
    xticks = [0.4, 0.6, 0.8, 0.999]
    xtick_labels = ['0.4', '0.6', '0.8', '0.999']

    def _bias_se(a):
        for case, out in out_dict.items():
            x = out["alpha"].to_numpy()
            a.plot(x, out["AbsBias_over_SE"], marker="o", ms=5, lw=1.4,
                   color=colors.get(case, "black"), label=labels.get(case, f"Design {case}"))
        a.axhline(0, color="k", ls="--", lw=1.0) # Unbiased reference line
        a.set_xlabel(r"$\alpha$")
        a.set_ylabel(r"$|\mathrm{Bias}|$ / SE")
        a.set_xticks(xticks)
        a.set_xticklabels(xtick_labels)
        a.legend()

    def _cp(a):
        for case, out in out_dict.items():
            x = out["alpha"].to_numpy()
            a.plot(x, out["CP"], marker="o", ms=5, lw=1.4, 
                   color=colors.get(case, "black"), label=labels.get(case, f"Design {case}"))
        a.axhline(0.95, color="k", ls="--", lw=1.0)
        a.set_xlabel(r"$\alpha$")
        a.set_ylabel("empirical coverage")
        a.set_xticks(xticks)
        a.set_xticklabels(xtick_labels)
        
        lo = min(out["CP"].min() for out in out_dict.values())
        a.set_ylim(min(lo - 0.02, 0.90), 0.98)
        a.legend()

    written = []
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for a, fn in zip(ax, (_bias_se, _cp)):
        fn(a)
        a.grid(alpha=0.25, ls="--", lw=0.5)
    fig.tight_layout()
    
    tag_all = args.tag or f"J{args.J}_m{args.m}_all"
    pth = os.path.join(args.out_dir, f"fig_alpha_all_{tag_all}.pdf")
    fig.savefig(pth, bbox_inches="tight")
    plt.close(fig)
    written.append(pth)

    if args.split:
        for name, fn in (("bias_se", _bias_se), ("cp", _cp)):
            fig, a = plt.subplots(figsize=(3.5, 2.9))
            fn(a)
            a.grid(alpha=0.25, ls="--", lw=0.5)
            fig.tight_layout()
            pth = os.path.join(args.out_dir, f"fig_alpha_{name}_{tag_all}.pdf")
            fig.savefig(pth, bbox_inches="tight")
            plt.close(fig)
            written.append(pth)

    print("\nwrote plots to:")
    for f in written:
        print("  ", f)


if __name__ == "__main__":
    main()
