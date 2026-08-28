"""
cate_postanalysis.py
====================
Tables and figures for ONE simulation cell (fixed J, fixed alpha, several m).

Outputs, all into --out_dir
    table_main_<tag>.tex        MAE / CP / AIL per estimand      -> main paper
    table_full_<tag>.tex        Bias / MAE / EmpSD / SE / CP / AIL
    metrics_<tag>.csv           the same numbers, machine readable
    coverage_by_time_<tag>.csv  CATE coverage against t_j
    fig_cp_ail_<tag>.pdf        CP and AIL against m, CATE, three models
    fig_sd_se_<tag>.pdf         empirical SD against estimated SE, CATE

MEMORY.  One (case, m) slice at J = 50 with 200 replications is already
about 10^6 rows and a whole cell is 1.2 x 10^7, so nothing is concatenated.
Files are read in chunks and reduced on the fly to three small accumulators:

    cell   one row per (estimand, case, m)              ~ 40 rows
    unit   one row per (estimand, case, m, evaluation)  ~ 70k rows
    tcov   one row per (case, m, t_j)                   ~ 3k rows

Peak memory is one chunk.  Empirical SDs are accumulated as sums and sums of
squares across replications, which gives the same answer as a groupby on the
full frame.

Usage
-----
    python cate_postanalysis.py --in "combined/cate_J100_a0.85_c*_m*.csv" \
        --J 100 --alpha 0.85 --ms 1,2,3,4,6,J \
        --out_dir figs/J100_a085 --tag J100_a085
"""

import os
import glob
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


USECOLS = ["estimand", "case", "m", "J", "alpha", "rep_id",
           "idx_a", "idx_b", "t_idx_a", "t_idx_b", "t_a",
           "target", "est", "se", "ci_lo", "ci_hi", "covered", "rho",
           "width_scale"]

CELL_KEY = ["estimand", "case", "m"]
UNIT_KEY = CELL_KEY + ["idx_a", "idx_b", "t_idx_a", "t_idx_b"]

EST_ORDER = ["point", "tm", "pt"]
EST_TEX = {
    "point": r"CATE $\tau_0(\mathbf{x}_*;t_j)$",
    "tm":    r"Temporal $\psi_{\mathrm{tm}}$",
    "pt":    r"Between-patient $\psi_{\mathrm{pt}}$",
}
CASE_TEX = {1: r"Design 1", 2: r"Design 2", 3: r"Design 3"}
PALETTE = ["#1565C0", "#E64A19", "#2E7D32"]
MARKERS = ["o", "s", "^"]


# =============================================================================
# streaming accumulation
# =============================================================================

def _add(store, key, frame):
    store[key] = frame if key not in store else store[key].add(frame,
                                                               fill_value=0.0)


def accumulate(paths, chunksize, keep_J, keep_alpha, keep_ms, keep_ws):
    """Single pass, reducing to small accumulators.  Filters are applied
    inside the chunk loop so unwanted cells never occupy memory."""
    cell, unit, tcov = {}, {}, {}
    reps = defaultdict(set)
    meta = {"J": set(), "alpha": set(), "width_scale": set()}
    n_kept = 0

    for path in paths:
        head = pd.read_csv(path, nrows=0)
        usecols = [c for c in USECOLS if c in head.columns]
        missing = {"estimand", "case", "m", "J", "alpha"} - set(usecols)
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)}")

        n_file = 0
        for d in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            if keep_J is not None:
                d = d[d["J"] == keep_J]
            if keep_alpha is not None:
                d = d[np.isclose(d["alpha"], keep_alpha)]
            if keep_ms is not None:
                d = d[d["m"].isin(keep_ms)]
            if keep_ws is not None and "width_scale" in d.columns:
                d = d[d["width_scale"] == keep_ws]
            if len(d) == 0:
                continue

            d = d.copy()
            d["estimand"] = d["estimand"].astype("category")
            meta["J"].update(d["J"].unique())
            meta["alpha"].update(d["alpha"].unique())
            if "width_scale" in d.columns:
                meta["width_scale"].update(d["width_scale"].dropna().unique())

            err = d["est"] - d["target"]
            d = d.assign(_err=err, _abs=err.abs(), _sq=err ** 2,
                         _ail=d["ci_hi"] - d["ci_lo"], _est2=d["est"] ** 2,
                         _se2=d["se"] ** 2, _arho=d["rho"].abs())

            g = d.groupby(CELL_KEY, observed=True)
            _add(cell, "c", g.agg(
                n=("est", "size"), s_err=("_err", "sum"),
                s_abs=("_abs", "sum"), s_sq=("_sq", "sum"),
                s_se=("se", "sum"), s_cov=("covered", "sum"),
                s_ail=("_ail", "sum"), s_rho=("_arho", "sum"),
                n_rho=("_arho", "count")))

            hi = d[d["_arho"] > 0.99]
            if len(hi):
                _add(cell, "hi", hi.groupby(CELL_KEY, observed=True)
                     .agg(n_hi=("_arho", "size")))

            # ss and ss_se give the across-replication SD of the estimate and
            # the Monte Carlo error of the averaged standard error
            _add(unit, "u", d.groupby(UNIT_KEY, observed=True).agg(
                n=("est", "size"), s=("est", "sum"), ss=("_est2", "sum"),
                s_tgt=("target", "sum"),
                s_se=("se", "sum"), ss_se=("_se2", "sum")))

            p = d[d["estimand"] == "point"]
            if len(p):
                _add(tcov, "t", p.groupby(["case", "m", "t_a"], observed=True)
                     .agg(n=("covered", "size"), s_cov=("covered", "sum")))

            for k, sub in d.groupby(CELL_KEY, observed=True):
                reps[k].update(sub["rep_id"].unique())
            n_file += len(d)

        n_kept += n_file
        print(f"   {os.path.basename(path):<46s} kept={n_file:>9,d}")

    if n_kept == 0:
        raise ValueError("no rows survived the --J / --alpha / --ms filters")
    print(f"   {'total kept':<46s}      {n_kept:>9,d}")
    return (cell.get("c"), unit.get("u"), tcov.get("t"), reps, meta,
            cell.get("hi"))


def unit_frame(unit):
    """Per-evaluation-point empirical SD, averaged SE, and the Monte Carlo
    error of that average."""
    u = unit.reset_index().copy()
    nn = u["n"].clip(lower=2)
    u["emp"] = np.sqrt((((u["ss"] - u["s"] ** 2 / u["n"]) / (nn - 1))
                        ).clip(lower=0.0))
    u["mse"] = u["s_se"] / u["n"]
    v_se = ((u["ss_se"] - u["s_se"] ** 2 / u["n"]) / (nn - 1)).clip(lower=0.0)
    u["mc_se"] = np.sqrt(v_se / u["n"])          # SE of the averaged SE
    # Corollary (bias negligibility) is a statement about bias^2 / sigma^2,
    # not about coverage.  The per-point Monte Carlo bias and its ratio to
    # the standard error are the quantities the corollary actually bounds,
    # and they move by orders of magnitude in m even where coverage does not.
    u["bias"] = (u["s"] / u["n"] - u["s_tgt"] / u["n"]).abs()
    u["bias_se"] = u["bias"] / u["mse"].replace(0.0, np.nan)
    return u


def summarise(cell, unit, reps):
    u = unit_frame(unit)
    gu = u.groupby(CELL_KEY, observed=True)
    emp = gu["emp"].mean()
    bs_mean, bs_max = gu["bias_se"].mean(), gu["bias_se"].max()
    c = cell.copy()
    out = pd.DataFrame({
        "Bias":  c["s_err"] / c["n"], "MAE": c["s_abs"] / c["n"],
        "RMSE":  np.sqrt(c["s_sq"] / c["n"]), "EmpSD": emp,
        "SE":    c["s_se"] / c["n"], "CP": c["s_cov"] / c["n"],
        "AIL":   c["s_ail"] / c["n"],
    })
    out["Ratio"] = out["SE"] / out["EmpSD"]
    out["BiasSE"] = bs_mean
    out["BiasSEmax"] = bs_max
    out = out[["Bias", "MAE", "RMSE", "EmpSD", "SE", "Ratio",
               "BiasSE", "BiasSEmax", "CP", "AIL"]]
    out = out.reset_index()
    out["n_rep"] = [len(reps[tuple(r)]) for r in
                    out[CELL_KEY].itertuples(index=False, name=None)]
    out["est_rank"] = out["estimand"].map({e: i for i, e in
                                           enumerate(EST_ORDER)})
    return (out.sort_values(["est_rank", "case", "m"])
               .drop(columns="est_rank").reset_index(drop=True))


def rho_report(cell, hi):
    c = cell[cell.index.get_level_values("estimand") != "point"]
    if len(c) == 0:
        return pd.DataFrame()
    r = pd.DataFrame({"mean_abs_rho": c["s_rho"] / c["n_rho"].clip(lower=1)})
    r["frac_gt_099"] = 0.0
    if hi is not None:
        f = (hi["n_hi"] / c["n_rho"]).dropna()
        r.loc[f.index, "frac_gt_099"] = f
    return r.reset_index()


# =============================================================================
# LaTeX
# =============================================================================

def _fmt(v, nd=4):
    return "---" if pd.isna(v) else f"{v:.{nd}f}"


def write_table_full(summ, J, alpha, path):
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         rf"\caption{{Monte Carlo summary at $J={J}$ and $\alpha={alpha:.2f}$.  "
         rf"$m$ is the temporal basis dimension; $m={J}$ is the cardinal "
         rf"basis, for which the readout is the identity and the estimator "
         rf"coincides with an unconstrained $J$-output network.  Nominal "
         rf"coverage is $0.95$.}}",
         rf"\label{{tab:sim_J{J}_a{int(round(alpha*100))}}}",
         r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r" & & $m$ & Bias & MAE & EmpSD & SE & CP & AIL \\", r"\midrule"]
    for e_i, est in enumerate(EST_ORDER):
        se_ = summ[summ["estimand"] == est]
        if len(se_) == 0:
            continue
        if e_i:
            L.append(r"\midrule")
        n_est, first_est = len(se_), True
        for cs in sorted(se_["case"].unique()):
            sc = se_[se_["case"] == cs]
            first_case = True
            for _, r_ in sc.iterrows():
                c0 = (rf"\multirow{{{n_est}}}{{*}}{{{EST_TEX[est]}}}"
                      if first_est else "")
                c1 = (rf"\multirow{{{len(sc)}}}{{*}}{{{CASE_TEX[cs]}}}"
                      if first_case else "")
                L.append(f"{c0} & {c1} & {int(r_['m'])} & "
                         f"{_fmt(r_['Bias'])} & {_fmt(r_['MAE'])} & "
                         f"{_fmt(r_['EmpSD'])} & {_fmt(r_['SE'])} & "
                         f"{_fmt(r_['CP'], 3)} & {_fmt(r_['AIL'])} \\\\")
                first_est = first_case = False
            L.append(r"\cmidrule(l){2-9}")
        if L[-1].startswith(r"\cmidrule"):
            L.pop()
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")


def write_table_main(summ, J, alpha, path):
    piv = {e: summ[summ["estimand"] == e].set_index(["case", "m"])
           for e in EST_ORDER}
    keys = sorted(set().union(*[set(p.index) for p in piv.values()]))
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         rf"\caption{{Coverage and precision at $J={J}$, $\alpha={alpha:.2f}$.  "
         rf"$m={J}$ is the cardinal basis, equivalent to an unconstrained "
         rf"$J$-output network.  Nominal coverage is $0.95$.}}",
         rf"\label{{tab:sim_main_J{J}_a{int(round(alpha*100))}}}",
         r"\begin{tabular}{lr rrr rrr rrr}", r"\toprule",
         r"& & \multicolumn{3}{c}{CATE} & "
         r"\multicolumn{3}{c}{Temporal contrast} & "
         r"\multicolumn{3}{c}{Between-patient contrast} \\",
         r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}",
         r"Model & $m$ & MAE & CP & AIL & MAE & CP & AIL & MAE & CP & AIL \\",
         r"\midrule"]
    last = None
    for cs, m in keys:
        if last is not None and cs != last:
            L.append(r"\addlinespace")
        cells = []
        for est in EST_ORDER:
            if (cs, m) in piv[est].index:
                r_ = piv[est].loc[(cs, m)]
                cells += [_fmt(r_["MAE"]), _fmt(r_["CP"], 3), _fmt(r_["AIL"])]
            else:
                cells += ["---"] * 3
        L.append(f"{CASE_TEX[cs] if cs != last else ''} & {int(m)} & "
                 + " & ".join(cells) + r" \\")
        last = cs
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")


# =============================================================================
# figure 1: CP and AIL against m, CATE
# =============================================================================

def fig_cp_ail(summ, J, alpha, path):
    """
    Coverage and interval length against the basis dimension, CATE.

    Both vertical scales are zoomed enough to read the movement, but the
    coverage panel always keeps the nominal level in view so the reader can
    judge how far the curves actually are from 0.95 rather than only how they
    are ordered.
    """
    s = summ[summ["estimand"] == "point"]
    cases = sorted(s["case"].unique())
    ms = sorted(s["m"].unique())
    x = np.arange(len(ms))
    lab = ["$J$" if m == J else str(int(m)) for m in ms]

    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.8))
    for k, cs in enumerate(cases):
        g = s[s["case"] == cs].set_index("m").reindex(ms)
        ax[0].plot(x, g["CP"].values, marker=MARKERS[k], ms=4.5, lw=1.3,
                   color=PALETTE[k], label=f"Design {cs}")
        ax[1].plot(x, g["AIL"].values, marker=MARKERS[k], ms=4.5, lw=1.3,
                   color=PALETTE[k], label=f"Design {cs}")

    ax[0].axhline(0.95, color="k", ls="--", lw=1.0)
    lo, hi = float(s["CP"].min()), float(s["CP"].max())
    pad = max(0.003, 0.25 * (hi - lo))
    ax[0].set_ylim(min(lo - pad, 0.9455), max(hi + pad, 0.9525))
    ax[0].set_ylabel("empirical coverage")
    ax[0].set_title("(a) coverage", fontsize=9.5)
    ax[0].legend(fontsize=7.5, frameon=False, loc="lower right")

    alo, ahi = float(s["AIL"].min()), float(s["AIL"].max())
    apad = max(0.01 * ahi, 0.18 * (ahi - alo))
    ax[1].set_ylim(alo - apad, ahi + apad)
    ax[1].set_ylabel("average interval length")
    ax[1].set_title("(b) interval length", fontsize=9.5)

    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(lab)
        a.set_xlabel("basis dimension $m$")
        a.set_xlim(-0.4, len(ms) - 0.6)
        a.grid(alpha=0.25, ls="--", lw=0.5)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# figure 2: empirical SD against estimated SE, CATE
# =============================================================================

def fig_sd_se(unit, J, alpha, m_ref, t_ref, out_dir, tag, split=False):
    """
    One point per evaluation profile: the empirical SD of the bagged
    estimator across replications against the bias-corrected IJ standard
    error averaged over the same replications.  Vertical bars are the Monte
    Carlo error of that average, so a point is consistent with the diagonal
    when its bar crosses it.
    """
    u = unit_frame(unit)
    u = u[(u["estimand"] == "point") & (u["m"] == m_ref)]
    if t_ref is not None:
        u = u[u["t_idx_a"] == t_ref]
    cases = sorted(u["case"].unique())

    lo = float(min(u["emp"].min(), (u["mse"] - u["mc_se"]).min()))
    hi = float(max(u["emp"].max(), (u["mse"] + u["mc_se"]).max()))
    pad = 0.08 * (hi - lo)
    lim = (lo - pad, hi + pad)

    def _panel(a, cs, show_ylab):
        g = u[u["case"] == cs]
        a.errorbar(g["emp"], g["mse"], yerr=1.96 * g["mc_se"],
                   fmt="o", ms=3.4, color="black",
                   ecolor="0.55", elinewidth=0.8, capsize=1.6, capthick=0.8,
                   linestyle="none", zorder=3)
        a.plot(lim, lim, "k--", lw=1.2, zorder=1)
        a.set_xlim(*lim)
        a.set_ylim(*lim)
        a.set_aspect("equal")
        a.set_xlabel("Empirical SD of bagged estimator")
        if show_ylab:
            a.set_ylabel("Estimated SE")
        a.grid(alpha=0.25, ls="--", lw=0.5)
        a.set_title(f"Design {cs}", fontsize=10)

    if split:
        paths = []
        for cs in cases:
            fig, a = plt.subplots(figsize=(3.3, 3.3))
            _panel(a, cs, True)
            fig.tight_layout()
            p = os.path.join(out_dir, f"fig_sd_se_{tag}_model{cs}.pdf")
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)
        return paths

    fig, ax = plt.subplots(1, len(cases), figsize=(3.15 * len(cases), 3.3))
    ax = np.atleast_1d(ax)
    for k, cs in enumerate(cases):
        _panel(ax[k], cs, k == 0)
    fig.tight_layout()
    p = os.path.join(out_dir, f"fig_sd_se_{tag}.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return [p]


# =============================================================================
# figure 3: bias^2 / sigma^2 against m, CATE, one model
# =============================================================================

def fig_bias_ratio(unit, J, alpha, case, path):
    """Mean |bias| / SE against the basis dimension, CATE, one design."""
    u = unit.reset_index()
    u = u[(u["estimand"] == "point") & (u["case"] == case)].copy()
    if len(u) == 0:
        return None
    ms = sorted(u["m"].unique())
    u["bias"] = (u["s"] - u["s_tgt"]) / u["n"]
    u["mse"] = u["s_se"] / u["n"]
    v = (u["bias"].abs() / u["mse"]).groupby(u["m"]).mean().reindex(ms)

    x = np.arange(len(ms))
    fig, a = plt.subplots(figsize=(4.0, 2.9))
    a.plot(x, v.values, marker="o", ms=5, lw=1.4, color="black")
    a.set_xticks(x)
    a.set_xticklabels(["$J$" if m == J else str(int(m)) for m in ms])
    a.set_xlim(-0.4, len(ms) - 0.6)
    a.set_xlabel("basis dimension $m$")
    a.set_ylabel(r"mean $|\mathrm{bias}| / \mathrm{SE}$")
    a.grid(alpha=0.25, ls="--", lw=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_cov_time(tcov, J, alpha, m_sel, path):
    """
    CATE coverage against follow-up time at a fixed basis dimension.

    The evaluation grid is design specific, so the three designs are drawn
    together only when their grids agree; otherwise Design 1 alone is shown.
    """
    t = tcov.reset_index()
    t = t[t["m"] == m_sel]
    if len(t) == 0:
        return None
    grids = {cs: tuple(np.round(sorted(g["t_a"].unique()), 8))
             for cs, g in t.groupby("case")}
    same = len(set(grids.values())) == 1
    cases = sorted(t["case"].unique()) if same else [min(t["case"].unique())]

    fig, a = plt.subplots(figsize=(5.0, 2.9))
    for k, cs in enumerate(cases):
        g = t[t["case"] == cs].sort_values("t_a")
        a.plot(g["t_a"], g["s_cov"] / g["n"], marker=MARKERS[k], ms=3, lw=1.2,
               color=PALETTE[k], label=f"Design {cs}")
    a.axhline(0.95, color="k", ls="--", lw=1.0)
    lo = float((t["s_cov"] / t["n"]).min())
    a.set_ylim(min(lo - 0.006, 0.9425), 0.9575)
    a.set_xlabel("follow-up time $t_j$")
    a.set_ylabel("empirical coverage")
    if len(cases) > 1:
        a.legend(fontsize=7.5, frameon=False, loc="lower right")
    a.grid(alpha=0.25, ls="--", lw=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, same, cases


def build_basis_local(tgrid, m, degree=3):
    """Clamped B-spline design matrix, mirroring cate_esm.build_spline_basis."""
    J = len(tgrid)
    if m >= J:
        return np.eye(J)
    if m == 1:
        return np.ones((J, 1))
    deg = min(degree, m - 1)
    a, b = float(tgrid[0]), float(tgrid[-1])
    n_int = m - deg - 1
    interior = (np.linspace(a, b, n_int + 2)[1:-1] if n_int > 0
                else np.array([]))
    k = np.concatenate([np.full(deg + 1, a), interior, np.full(deg + 1, b)])
    t = np.asarray(tgrid, float)
    N = np.zeros((len(t), len(k) - 1))
    last = max(i for i in range(len(k) - 1) if k[i + 1] > k[i])
    for i in range(len(k) - 1):
        if k[i + 1] <= k[i]:
            continue
        N[:, i] = (((t >= k[i]) & (t <= k[i + 1])) if i == last
                   else ((t >= k[i]) & (t < k[i + 1]))).astype(float)
    for d in range(1, deg + 1):
        Nn = np.zeros((len(t), len(k) - d - 1))
        for i in range(len(k) - d - 1):
            d1, d2 = k[i + d] - k[i], k[i + d + 1] - k[i + 1]
            if d1 > 0:
                Nn[:, i] += (t - k[i]) / d1 * N[:, i]
            if d2 > 0:
                Nn[:, i] += (k[i + d + 1] - t) / d2 * N[:, i + 1]
        N = Nn
    return N[:, :m]


# =============================================================================
# main
# =============================================================================

def _parse_ms(spec, J):
    if spec is None:
        return None
    out = []
    for tok in str(spec).replace(" ", "").split(","):
        if tok in ("J", "j"):
            out.append(int(J))
        elif tok:
            out.append(int(tok))
    return sorted(set(out))


def main():
    p = argparse.ArgumentParser(
        description="Tables and figures for one simulation cell "
                    "(streams the input, so large files are fine)")
    p.add_argument("--in", dest="infiles", type=str, nargs="+", required=True,
                   help="combined CSV(s): one file, several files, or a "
                        "quoted glob")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--J", type=int, default=None,
                   help="keep this J only; required if the inputs mix J")
    p.add_argument("--alpha", type=float, default=None,
                   help="keep this alpha only")
    p.add_argument("--ms", type=str, default=None,
                   help="comma list of basis dimensions to keep, the token J "
                        "meaning m = J, e.g. --ms 1,2,3,4,6,J")
    p.add_argument("--m_ref", type=int, default=None,
                   help="basis dimension for the SD-vs-SE figure; default is "
                        "the smallest m below the cardinal J")
    p.add_argument("--t_ref_idx", type=str, default="mid",
                   help="time index for the SD-vs-SE figure: an integer, "
                        "'mid' (default), or 'all' to pool every time")
    p.add_argument("--bias_case", type=int, default=1, choices=[1, 2, 3],
                   help="design used for the bias/SE figure")
    p.add_argument("--tcov_m", type=int, default=None,
                   help="basis dimension for the coverage-against-time "
                        "figure; default is the m whose CATE coverage is "
                        "closest to nominal")
    p.add_argument("--split_sd_se", action="store_true",
                   help="write one PDF per model instead of a 1 x 3 panel")
    p.add_argument("--width_scale", type=str, default=None,
                   help="keep one architecture scaling only, e.g. fixed or "
                        "sqrt_m; required if the inputs mix them")
    p.add_argument("--chunksize", type=int, default=500_000)
    args = p.parse_args()

    paths = []
    for pat in args.infiles:
        hits = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        if not hits:
            raise FileNotFoundError(f"no file matches '{pat}'")
        paths.extend(hits)
    paths = sorted(dict.fromkeys(paths))

    keep_ms = _parse_ms(args.ms, args.J) if args.J else _parse_ms(args.ms, -1)

    print(f"streaming {len(paths)} file(s), chunksize={args.chunksize:,}"
          + (f", J={args.J}" if args.J else "")
          + (f", alpha={args.alpha}" if args.alpha else "")
          + (f", m in {keep_ms}" if keep_ms else ""))
    cell, unit, tcov, reps, meta, hi = accumulate(
        paths, args.chunksize, args.J, args.alpha, keep_ms,
        args.width_scale)

    Js, alphas = sorted(meta["J"]), sorted(meta["alpha"])
    ws = sorted(meta["width_scale"])
    if len(Js) > 1 or len(alphas) > 1 or len(ws) > 1:
        raise ValueError(f"inputs mix cells: J={Js} alpha={alphas} "
                         f"width_scale={ws}; pass --J, --alpha, --width_scale")
    J, alpha = int(Js[0]), float(alphas[0])
    tag = args.tag or f"J{J}_a{int(round(alpha * 100)):03d}"

    summ = summarise(cell, unit, reps)
    ms = sorted(summ["m"].unique())

    m_ref = args.m_ref
    if m_ref is None:
        below = [m for m in ms if m < J]
        m_ref = below[0] if below else ms[0]
    if m_ref not in ms:
        raise ValueError(f"--m_ref {m_ref} not present; available m = {ms}")

    if str(args.t_ref_idx).lower() == "all":
        t_ref = None
    elif str(args.t_ref_idx).lower() == "mid":
        t_ref = J // 2
    else:
        t_ref = int(args.t_ref_idx)

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 78)
    print(f"  J = {J}   alpha = {alpha:.2f}   m present = {ms}")
    print(f"  SD-vs-SE figure at m = {m_ref}, "
          f"t index = {'all' if t_ref is None else t_ref}")
    print(f"  replications per cell: {summ['n_rep'].min()} to "
          f"{summ['n_rep'].max()}")
    print("=" * 78)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summ.round(4).to_string(index=False))

    rr = rho_report(cell, hi)
    if len(rr):
        print("\n  contrast non-degeneracy (|rho| must stay below one)")
        print(rr.round(4).to_string(index=False))

    csv_path = os.path.join(args.out_dir, f"metrics_{tag}.csv")
    summ.to_csv(csv_path, index=False)
    tc = tcov.reset_index()
    tc["CP"] = tc["s_cov"] / tc["n"]
    tcov_path = os.path.join(args.out_dir, f"coverage_by_time_{tag}.csv")
    tc[["case", "m", "t_a", "CP"]].to_csv(tcov_path, index=False)

    t_main = os.path.join(args.out_dir, f"table_main_{tag}.tex")
    t_full = os.path.join(args.out_dir, f"table_full_{tag}.tex")
    write_table_main(summ, J, alpha, t_main)
    write_table_full(summ, J, alpha, t_full)

    f_cp = os.path.join(args.out_dir, f"fig_cp_ail_{tag}.pdf")
    fig_cp_ail(summ, J, alpha, f_cp)
    f_sd = fig_sd_se(unit, J, alpha, m_ref, t_ref, args.out_dir, tag,
                     split=args.split_sd_se)
    f_br = fig_bias_ratio(unit, J, alpha, args.bias_case,
                          os.path.join(args.out_dir, f"fig_bias_se_{tag}.pdf"))

    # basis dimension for the coverage-against-time panel: the one whose CATE
    # coverage sits closest to nominal, unless the user names it
    if args.tcov_m is None:
        p_ = summ[summ["estimand"] == "point"]
        agg = (p_.assign(d=(p_["CP"] - 0.95).abs())
                 .groupby("m")["d"].mean())
        m_sel = int(agg.idxmin())
    else:
        m_sel = int(args.tcov_m)
    res = fig_cov_time(tcov, J, alpha, m_sel,
                       os.path.join(args.out_dir, f"fig_cov_time_{tag}.pdf"))
    f_ct = None
    if res is not None:
        f_ct, same_grid, shown = res
        print(f"\n  coverage-against-time at m = {m_sel}; grids "
              + ("agree, all designs drawn"
                 if same_grid else
                 f"differ by design, showing Design {shown[0]} only"))

    print("\nwrote")
    for f in ([csv_path, tcov_path, t_main, t_full, f_cp] + list(f_sd)
              + [x for x in (f_br, f_ct) if x]):
        print(f"   {f}")


if __name__ == "__main__":
    main()
