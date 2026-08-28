"""
cate_esm.py
===========
Ensemble Subsampling Method (ESM) + Infinitesimal Jackknife (IJ) inference
for the conditional average treatment effect (CATE) surface in survival
analysis.  ONE SLURM job = ONE replication.

ESTIMATOR (spline-projected multi-output network)
-------------------------------------------------
The two arguments of tau_0(x; t) carry different structure and are handled
by different devices.

    tau_f(x; t_j) = sum_{l<=m} f_l(x) B_l(t_j)      i.e.   tau = B f(x)

The covariate dependence is estimated by a deep ReLU network with m output
nodes f_1..f_m; the temporal dependence is represented by a FIXED B-spline
basis evaluated on the grid, collected in the (J x m) matrix B.  The basis
carries no free parameters, gradients propagate through it during training,
and m <= J is required for B to have full column rank.

Setting m = J with a cardinal basis recovers the unconstrained J-output
network of the earlier draft; m = 1 recovers the fixed-t estimator.

THREE ESTIMANDS
---------------
    point   tau_0(x*; t_j)                                pointwise CATE
    tm      tau_0(x*; t_j') - tau_0(x*; t_j)              temporal contrast
    pt      tau_0(x1*; t_j) - tau_0(x2*; t_j)             between-patient

The temporal contrast needs the IJ covariance ACROSS TIMES at a fixed
profile (the J x J block Psi(x*)); the between-patient contrast needs the
IJ covariance ACROSS PROFILES at a common time, an object no pointwise
analysis constructs.  Both are read off the same fitted ensemble.

THREE SIMULATION CASES  (--case 1|2|3)
--------------------------------------
Case 1  Correct survival, correct propensity.
Case 2  Correct survival, MISSPECIFIED propensity (nonlinear e(x)).
Case 3  MISSPECIFIED survival (non-proportional hazards), correct e.

Pipeline
--------
1.  Generate data; build the fixed time grid, spline basis, evaluation
    points and contrast pairs.
2.  K-fold cross-fitted AIPW pseudo-outcomes phi_i(t_j).
3.  ESM: B subsamples of size r = n^alpha, one spline-projected DNN per
    bag, plain L2 loss.  Nuisances are estimated ONCE and frozen before
    subsampling.
4.  IJ: per-profile J x J covariance (pointwise + temporal contrast) and
    cross-profile covariance at common times (between-patient contrast).

Usage
-----
    python cate_esm.py --rep_id 0 --case 1 --alpha 0.90 --J 50 --m 4 \
        --n_jobs 8 --out_dir results
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit

warnings.filterwarnings("ignore")


# =============================================================================
# 0.  Design constants
# =============================================================================

D_COV   = 10        # covariate dimension
N_TRAIN = 2000      # training sample size
N_TEST  = 400       # test pool
N_EVAL  = 80        # evaluation points x* carried into the CSV

# ---- Cases 1 / 2 : Weibull-Cox ---------------------------------------------
LAMBDA_W = 0.10     # Weibull scale
NU_W     = 1.20     # Weibull shape
LAMBDA_C = 0.08     # Exp-Cox censoring scale -> ~36% censoring

BETA_C12  = np.linspace( 0.40, -0.20, D_COV)   # prognostic index
THETA_C12 = np.linspace( 0.60, -0.20, D_COV)   # treatment-effect index
GAMMA_C12 = np.linspace( 0.30, -0.30, D_COV)   # censoring index

# ---- Case 3 : non-proportional hazards -------------------------------------
LAMBDA0_C3  = 0.05
LAMBDA_C_C3 = 0.05                              # -> ~50% censoring

BETA_C3  = np.linspace(-0.30, 0.50, D_COV)     # g(x)
GAMMA_C3 = np.linspace(-0.20, 0.30, D_COV)     # u(x)  AND censoring index
THETA_C3 = np.linspace(-0.20, 0.40, D_COV)     # v(x)

C3_TMAX = 400.0
C3_MESH = 8001

# ---- propensity (shared) ----------------------------------------------------
ALPHA_PS = np.linspace(1.0, -1.0, D_COV) / np.sqrt(D_COV)

# ---- nuisance truncation: bounds the AIPW weight 1/(e*G) -------------------
E_CLIP_LO, E_CLIP_HI = 0.05, 0.95
G_CLIP_LO            = 0.05

# ---- evaluation band --------------------------------------------------------
INDEX_BAND_LO_Q, INDEX_BAND_HI_Q = 0.10, 0.90

# ---- time-grid quantiles ----------------------------------------------------
GRID_Q_LO = 0.20
GRID_Q_HI = 0.70

# ---- spline basis -----------------------------------------------------------
SPLINE_DEGREE = 3          # cubic; order = degree + 1 >= ceil(beta_t) + 1

# ---- contrast pair selection ------------------------------------------------
# Between-patient pairs are chosen once, from the FIXED evaluation points and
# the FIXED grid, so they do not move across replications.  Both members are
# required to sit in the central band of tau_0(.; t_ref) and their gap is
# targeted at PT_GAP_FRAC of the band width -- a moderate, clearly separated
# comparison of the kind a clinical report would make.  Requiring separation
# is not cosmetic: it is Assumption (contrast non-degeneracy), and rho_pt is
# recorded in the output so the condition can be checked empirically.
PT_BAND_Q   = (0.15, 0.85)
PT_GAP_FRAC = 0.50
N_PT_PAIRS  = 20

SEED_TEST = 20260719   # fixed -> identical grid, basis, x* in every rep
SEED_BASE = 917


def case_coefs(case):
    """Coefficient bundle for the requested case."""
    if case in (1, 2):
        return dict(beta=BETA_C12, theta=THETA_C12, gamma=GAMMA_C12,
                    lam0=LAMBDA_W, lamC=LAMBDA_C)
    return dict(beta=BETA_C3, theta=THETA_C3, gamma=GAMMA_C3,
                lam0=LAMBDA0_C3, lamC=LAMBDA_C_C3)


# =============================================================================
# 1.  Case 3 helpers -- numerical cumulative hazard
# =============================================================================

def _c3_h(u):
    """h(x) = 0.5 cos(u) + 0.2 u^2   with u = gamma'x."""
    return 0.5 * np.cos(u) + 0.2 * u ** 2


def c3_cum_hazard_treated(X, mesh):
    """
    H^1(t | x) = lam0 exp(g(x)) int_0^t exp{ h(x) sin(v(x) s) } ds
    tabulated on `mesh` by cumulative trapezoid.
    """
    g = X @ BETA_C3
    u = X @ GAMMA_C3
    v = X @ THETA_C3
    h = _c3_h(u)

    integ = np.exp(h[:, None] * np.sin(v[:, None] * mesh[None, :]))
    dt    = mesh[1] - mesh[0]
    cum   = np.concatenate(
        [np.zeros((len(X), 1)),
         np.cumsum(0.5 * (integ[:, 1:] + integ[:, :-1]) * dt, axis=1)], axis=1)
    return LAMBDA0_C3 * np.exp(g)[:, None] * cum


def c3_cum_hazard_control(X, t):
    """H^0(t | x) = lam0 exp(g(x)) t  -- closed form (W=0 kills the sine)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return LAMBDA0_C3 * np.exp(X @ BETA_C3)[:, None] * t[None, :]


# =============================================================================
# 2.  DGP and true CATE
# =============================================================================

def true_survival(X, w, t_grid, case):
    """S^w(t | x) for the requested case.  Returns (n, J)."""
    t_grid = np.atleast_1d(np.asarray(t_grid, dtype=float))

    if case in (1, 2):
        eta = X @ BETA_C12 + w * (X @ THETA_C12)
        H   = LAMBDA_W * (t_grid[None, :] ** NU_W) * np.exp(eta)[:, None]
        return np.exp(-H)

    if w == 0:
        return np.exp(-c3_cum_hazard_control(X, t_grid))

    mesh = np.linspace(0.0, C3_TMAX, C3_MESH)
    Hm   = c3_cum_hazard_treated(X, mesh)
    H_at = np.empty((len(X), len(t_grid)))
    for i in range(len(X)):
        H_at[i] = np.interp(t_grid, mesh, Hm[i])
    return np.exp(-H_at)


def true_cate(X, t_grid, case):
    """
    tau_0(x; t) = S^1(t|x) - S^0(t|x).

    STRUCTURE.  In Cases 1-2, tau_0(x;t) = h(z(x), t) with
    z(x) = (beta'x, theta'x) in R^2, so the intrinsic covariate dimension is
    d* = 2 with q = 0 even though d = 10.  In Case 3 the index is
    (beta'x, gamma'x, theta'x) in R^3, so d* = 3.  In both cases the map
    t -> h(z, t) is smooth on a one-dimensional interval, which is exactly
    the structure the spline projection exploits.
    """
    return (true_survival(X, 1, t_grid, case) -
            true_survival(X, 0, t_grid, case))


def true_propensity(X, case):
    """Cases 1, 3: logistic in the linear index.  Case 2: nonlinear."""
    eta = X @ ALPHA_PS
    if case != 2:
        return expit(eta)
    return np.clip(
        expit(0.50 + 0.45 * np.tanh(eta / 1.2)
              + 0.25 * (eta > 0) - 0.15 * (eta < 1)
              + 0.10 * np.sin(eta ** 2 / 3.0)),
        0.02, 0.98)


def generate_data(n, rng, case):
    """X ~ U[-1,1]^d, W ~ Bern(e(X)), T case-specific, C ~ Exp-Cox."""
    X      = rng.uniform(-1.0, 1.0, size=(n, D_COV))
    e_true = true_propensity(X, case)
    W      = rng.binomial(1, e_true).astype(float)
    cf     = case_coefs(case)

    if case in (1, 2):
        eta = X @ BETA_C12 + W * (X @ THETA_C12)
        E   = rng.exponential(1.0, n)
        T   = (E / (LAMBDA_W * np.exp(eta))) ** (1.0 / NU_W)
    else:
        E    = rng.exponential(1.0, n)
        T    = np.empty(n)
        mesh = np.linspace(0.0, C3_TMAX, C3_MESH)

        ctrl = W == 0
        if ctrl.any():
            T[ctrl] = E[ctrl] / (LAMBDA0_C3 * np.exp(X[ctrl] @ BETA_C3))

        trt = ~ctrl
        if trt.any():
            Hm = c3_cum_hazard_treated(X[trt], mesh)
            Et = E[trt]
            Tt = np.empty(trt.sum())
            for k in range(trt.sum()):
                Tt[k] = np.interp(Et[k], Hm[k], mesh,
                                  left=0.0, right=C3_TMAX)
            T[trt] = Tt

    C     = rng.exponential(1.0, n) / (cf["lamC"] * np.exp(X @ cf["gamma"]))
    U     = np.minimum(T, C)
    Delta = (T <= C).astype(float)
    return {"X": X, "W": W, "U": U, "Delta": Delta, "e_true": e_true}


def build_time_grid(J, case, q_lo=GRID_Q_LO, q_hi=GRID_Q_HI, seed=SEED_TEST):
    """
    J equally spaced points between the q_lo and q_hi percentiles of the
    OBSERVED survival distribution, computed once from a pilot sample with a
    FIXED seed.  Fixing the grid across replications is essential.
    """
    rng   = np.random.default_rng(seed + 100 * case)
    pilot = generate_data(8000, rng, case)
    if J == 1:
        return np.array([np.quantile(pilot["U"], 0.50)])
    return np.linspace(np.quantile(pilot["U"], q_lo),
                       np.quantile(pilot["U"], q_hi), J)


def build_eval_points(case, seed=SEED_TEST):
    """Fixed evaluation points x*, identical across all replications."""
    rng    = np.random.default_rng(seed + 1)
    X_pool = rng.uniform(-1.0, 1.0, size=(N_TEST, D_COV))
    beta   = BETA_C12 if case in (1, 2) else BETA_C3
    idx    = X_pool @ beta

    lo = np.quantile(idx, INDEX_BAND_LO_Q)
    hi = np.quantile(idx, INDEX_BAND_HI_Q)
    band = np.where((idx >= lo) & (idx <= hi))[0]
    band = band[np.argsort(idx[band])]
    sel  = (band[np.linspace(0, len(band) - 1, N_EVAL).astype(int)]
            if len(band) >= N_EVAL else band)
    return X_pool[sel], float(lo), float(hi)


# =============================================================================
# 2b.  Fixed temporal B-spline basis
# =============================================================================

def _cox_de_boor(t, knots, degree):
    """
    Evaluate all B-spline basis functions of the given degree at points t,
    by the Cox-de Boor recursion.  Returns (len(t), len(knots)-degree-1).

    Implemented directly rather than through scipy so the script does not
    depend on a particular scipy version on the cluster.
    """
    t = np.asarray(t, dtype=float)
    k = np.asarray(knots, dtype=float)
    n_out = len(k) - degree - 1

    # order 1 (piecewise constant); the last non-degenerate interval is
    # closed on the right so that t = t_max is covered
    N = np.zeros((len(t), len(k) - 1))
    last = max(i for i in range(len(k) - 1) if k[i + 1] > k[i])
    for i in range(len(k) - 1):
        if k[i + 1] <= k[i]:
            continue
        if i == last:
            N[:, i] = ((t >= k[i]) & (t <= k[i + 1])).astype(float)
        else:
            N[:, i] = ((t >= k[i]) & (t < k[i + 1])).astype(float)

    for d in range(1, degree + 1):
        Nn = np.zeros((len(t), len(k) - d - 1))
        for i in range(len(k) - d - 1):
            den1 = k[i + d] - k[i]
            den2 = k[i + d + 1] - k[i + 1]
            if den1 > 0:
                Nn[:, i] += (t - k[i]) / den1 * N[:, i]
            if den2 > 0:
                Nn[:, i] += (k[i + d + 1] - t) / den2 * N[:, i + 1]
        N = Nn
    return N[:, :n_out]


def build_spline_basis(t_grid, m, degree=SPLINE_DEGREE):
    """
    Fixed (J x m) readout matrix Bmat[j, l] = B_l(t_j).

    Knots are clamped at the endpoints of the grid with m - degree - 1
    equally spaced interior knots, so the partition is quasi-uniform and the
    basis is a partition of unity: sup_t sum_l |B_l(t)| = 1, i.e. C_B = 1.

    m = 1 gives the constant basis (the fixed-t estimator) and m = J gives
    the cardinal basis B_l(t_j) = I(l = j), for which the readout is the
    identity and the estimator coincides with an unconstrained J-output
    network.  Both special cases are handled exactly rather than through the
    recursion.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    J = len(t_grid)
    if m > J:
        raise ValueError(f"m = {m} exceeds J = {J}: the readout matrix "
                         f"cannot have full column rank")
    if m == J:
        return np.eye(J)                       # cardinal basis
    if m == 1:
        return np.ones((J, 1))

    deg = min(degree, m - 1)
    a, b = t_grid[0], t_grid[-1]
    n_int = m - deg - 1
    interior = (np.linspace(a, b, n_int + 2)[1:-1] if n_int > 0
                else np.array([]))
    knots = np.concatenate([np.full(deg + 1, a), interior,
                            np.full(deg + 1, b)])

    Bmat = _cox_de_boor(t_grid, knots, deg)
    if Bmat.shape[1] != m:
        raise RuntimeError(f"basis has {Bmat.shape[1]} columns, expected {m}")
    return Bmat


def basis_diagnostics(Bmat):
    """Rank, overlap constant C_B, and worst-column support on the grid."""
    col_support = (np.abs(Bmat) > 1e-12).sum(axis=0)
    return {
        "basis_rank": int(np.linalg.matrix_rank(Bmat)),
        "basis_CB": float(np.abs(Bmat).sum(axis=1).max()),
        "basis_min_col_support": int(col_support.min()),
        "basis_cond": float(np.linalg.cond(Bmat)),
    }


# =============================================================================
# 3.  Cox PH fitter and logistic
# =============================================================================

def _cox_neg_pl_grad(b, X, T, D):
    """Negative Breslow partial log-likelihood and gradient (vectorised)."""
    n     = len(T)
    order = np.argsort(-T)
    Xs, Ds = X[order], D[order]
    eta = Xs @ b
    m   = eta.max()
    w   = np.exp(eta - m)
    S0  = np.cumsum(w)
    S1  = np.cumsum(w[:, None] * Xs, axis=0)
    nll  = -np.sum(Ds * (eta - (np.log(S0) + m)))
    grad = -np.sum(Ds[:, None] * (Xs - S1 / S0[:, None]), axis=0)
    return nll / n, grad / n


def fit_cox(X, T, D, ridge=1e-4):
    p = X.shape[1]
    if D.sum() < 5:
        return np.zeros(p)

    def obj(b):
        nll, g = _cox_neg_pl_grad(b, X, T, D)
        return nll + ridge * b @ b, g + 2 * ridge * b

    return minimize(obj, np.zeros(p), jac=True, method="L-BFGS-B",
                    options={"maxiter": 300, "ftol": 1e-10}).x


def breslow_baseline(X, T, D, b):
    """Breslow baseline cumulative hazard -> (event times asc, H0)."""
    eta = X @ b
    m   = eta.max()
    w   = np.exp(eta - m)
    o   = np.argsort(-T)
    Ts, Ds, ws = T[o], D[o], w[o]
    S0  = np.cumsum(ws)
    inc = np.where(S0 > 1e-12, Ds / S0, 0.0)
    t_asc  = Ts[::-1]
    H0_asc = np.cumsum(inc[::-1]) * np.exp(-m)
    ev = Ds[::-1] > 0
    if ev.sum() == 0:
        return np.array([0.0]), np.array([0.0])
    return t_asc[ev], H0_asc[ev]


def cox_survival_at(t_grid, X_new, b, t_ev, H0_ev):
    """S(t|x) = exp{ -H0(t) exp(b'x) }  ->  (n, J)."""
    H0_t = np.interp(t_grid, t_ev, H0_ev, left=0.0, right=H0_ev[-1])
    eta  = np.clip(X_new @ b, -20, 20)
    return np.exp(-H0_t[None, :] * np.exp(eta)[:, None])


def fit_logistic(X, y, ridge=1e-3, n_iter=200):
    """Ridge-penalised logistic regression by IRLS."""
    n, p = X.shape
    Xd = np.hstack([np.ones((n, 1)), X])
    b  = np.zeros(p + 1)
    for _ in range(n_iter):
        mu = expit(Xd @ b)
        Wd = np.clip(mu * (1 - mu), 1e-6, None)
        g  = Xd.T @ (y - mu) - 2 * ridge * b
        H  = Xd.T @ (Xd * Wd[:, None]) + 2 * ridge * np.eye(p + 1)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


# =============================================================================
# 4.  Cross-fitted AIPW pseudo-outcomes
# =============================================================================

def build_pseudo_outcomes(data, t_grid, case, K=5, rng=None):
    """
    K-fold cross-fitted AIPW pseudo-outcome:

        pi_i^w(t) = S^w(X_i;t)
                  + I(W_i=w)/e^w(X_i) * { Y_i(t)/G^w(X_i;t) - S^w(X_i;t) }
        phi_i(t)  = pi_i^1(t) - pi_i^0(t)

    The construction is doubly robust with respect to the pair (e, S) under
    consistent estimation of the censoring survival G, which is why Case 2
    (wrong e) and Case 3 (wrong S) should still cover while G stays right.
    """
    X, W, U, Dl = data["X"], data["W"], data["U"], data["Delta"]
    n, J = len(U), len(t_grid)

    phi   = np.zeros((n, J))
    e_all = np.zeros(n)
    G_all = np.zeros((n, J))

    folds = np.arange(n) % K
    if rng is not None:
        rng.shuffle(folds)

    for k in range(K):
        tr = np.where(folds != k)[0]
        te = np.where(folds == k)[0]

        b_ps  = fit_logistic(X[tr], W[tr])
        e_hat = expit(np.hstack([np.ones((len(te), 1)), X[te]]) @ b_ps)
        e_hat = np.clip(e_hat, E_CLIP_LO, E_CLIP_HI)
        e_all[te] = e_hat

        pi = {}
        for w in (0, 1):
            arm = tr[W[tr] == w]
            if len(arm) < 20:
                pi[w] = np.zeros((len(te), J))
                continue

            b_S = fit_cox(X[arm], U[arm], Dl[arm])
            tS, H0S = breslow_baseline(X[arm], U[arm], Dl[arm], b_S)
            S_hat = cox_survival_at(t_grid, X[te], b_S, tS, H0S)

            b_G = fit_cox(X[arm], U[arm], 1.0 - Dl[arm])
            tG, H0G = breslow_baseline(X[arm], U[arm], 1.0 - Dl[arm], b_G)
            G_hat = np.clip(cox_survival_at(t_grid, X[te], b_G, tG, H0G),
                            G_CLIP_LO, 1.0)

            Y   = (U[te][:, None] > t_grid[None, :]).astype(float)
            ew  = e_hat if w == 1 else (1.0 - e_hat)
            ind = (W[te] == w).astype(float)
            pi[w] = S_hat + (ind / ew)[:, None] * (Y / G_hat - S_hat)

            if w == 1:
                G_all[te] = G_hat

        phi[te] = pi[1] - pi[0]

    tau0 = true_cate(X, t_grid, case)
    vphi, vtau = float(np.var(phi)), float(np.var(tau0))
    diag = {
        "var_phi": vphi, "var_tau0": vtau,
        "var_ratio_phi": vphi / max(vtau, 1e-12),
        "e_min": float(e_all.min()), "e_max": float(e_all.max()),
        "e_clip_frac": float(np.mean((e_all <= E_CLIP_LO + 1e-9) |
                                     (e_all >= E_CLIP_HI - 1e-9))),
        "G_min": float(G_all.min()),
        "G_min_last_t": float(G_all[:, -1].min()),
        "G_mean_last_t": float(G_all[:, -1].mean()),
        "G_clip_frac": float(np.mean(G_all <= G_CLIP_LO + 1e-9)),
        "at_risk_frac_last": float(np.mean(U > t_grid[-1])),
        "phi_absmax": float(np.abs(phi).max()),
    }
    return phi, diag


# =============================================================================
# 5.  Spline-projected network and bag training
# =============================================================================

class SplineNet(nn.Module):
    """
    ReLU trunk with m output nodes, composed with the FIXED readout B.

        f(x) in R^m   ->   tau_f(x) = B f(x) in R^J

    The readout is registered as a buffer, not a parameter: its entries are
    known constants determined by the basis and the evaluation grid, so it
    contributes no free weights.  Gradients still propagate through it, so
    the m coefficient functions are trained against all J residuals per
    subject rather than against one residual each.
    """
    def __init__(self, d_in, m, Bmat, h1=128, h2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h1), nn.ReLU(),
            nn.Linear(h1,  h2), nn.ReLU(),
            nn.Linear(h2,   m))
        # store B^T of shape (m, J) so forward is a single matmul
        self.register_buffer(
            "Bt", torch.tensor(np.asarray(Bmat).T, dtype=torch.float32))

    def forward(self, x):
        return self.net(x) @ self.Bt


def _train_one_bag(Xb, Pb, X_eval, d_in, m, Bmat,
                   h1, h2, n_epochs, lr, weight_decay, tol, patience):
    """Train one bag; return its predictions at the evaluation points."""
    torch.set_num_threads(1)
    Xt = torch.tensor(Xb, dtype=torch.float32)
    Pt = torch.tensor(Pb, dtype=torch.float32)
    Xe = torch.tensor(X_eval, dtype=torch.float32)

    model = SplineNet(d_in, m, Bmat, h1, h2)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.MSELoss()

    prev, stable = None, 0
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = lossf(model(Xt), Pt)      # plain L2 over all J residuals
        loss.backward()
        opt.step()
        lv = loss.item()
        if prev is not None and abs(prev - lv) < tol:
            stable += 1
            if stable >= patience:
                break
        else:
            stable = 0
        prev = lv

    model.eval()
    with torch.no_grad():
        return model(Xe).numpy().astype(np.float64)


def train_ensemble(X, phi, X_eval, Bmat, B, r, rng, h1, h2,
                   n_epochs, lr, weight_decay, n_jobs,
                   tol=1e-7, patience=15):
    """
    Draw B subsamples of size r WITHOUT replacement; train one net per bag.

    Returns
    -------
    preds    : (B, n_eval, J)
    N_matrix : (B, n)   N[b, i] = 1 iff subject i is in bag b
    """
    n = len(X)
    m = Bmat.shape[1]
    idx = [rng.choice(n, size=r, replace=False) for _ in range(B)]
    N   = np.zeros((B, n))
    for b, s in enumerate(idx):
        N[b, s] = 1.0

    out = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_train_one_bag)(
            X[s], phi[s], X_eval, X.shape[1], m, Bmat,
            h1, h2, n_epochs, lr, weight_decay, tol, patience)
        for s in idx)
    return np.stack(out, axis=0), N


# =============================================================================
# 6.  Infinitesimal Jackknife
# =============================================================================
#
# Bias-corrected IJ covariance between two evaluation points a, a':
#
#     Gamma(a,a') = n(n-1)/(n-r)^2 * {
#           sum_i V_i(a) V_i(a')
#         - 1/(B(B-1)) sum_i sum_b (Z_bi(a) - V_i(a))(Z_bi(a') - V_i(a')) }
#
# with Z_bi(a) = (J_bi - J_.i)(tau^b(a) - tau_B(a)) and V_i(a) = mean_b Z_bi(a).
#
# Vectorisation.  Z_bi(a) = cN[b,i] * cp[b,a], so
#     sum_i sum_b Z_bi(a) Z_bi(a') = sum_b w_b cp[b,a] cp[b,a'],
#     w_b := sum_i cN[b,i]^2,
# and sum_i sum_b V_i(a) V_i(a') = B * sum_i V_i(a) V_i(a'), collapsing the
# (i, b) double loop to two einsums.
#
# Two objects are needed, and only two.  The per-profile J x J block supplies
# the pointwise variance and the temporal contrast; the cross-profile entries
# at a COMMON time supply the between-patient contrast.  The full grid
# covariance is never formed.


def _ij_pieces(preds, N_matrix):
    """Centred predictions, centred inclusion matrix and its row weights."""
    tau_B = preds.mean(axis=0)
    cp    = preds - tau_B[None, :, :]                 # (B, n_eval, J)
    cN    = N_matrix - N_matrix.mean(axis=0)[None, :]  # (B, n)
    w     = np.einsum("bi,bi->b", cN, cN)              # (B,)
    return tau_B, cp, cN, w


def ij_within_profile(preds, N_matrix, n, r, chunk=16):
    """
    Per-profile J x J covariance Psi[x] -- pointwise SE and temporal
    contrast.  Chunked over evaluation points because the intermediate V
    array is (chunk, n, J).
    """
    B, n_eval, J = preds.shape
    tau_B, cp, cN, w = _ij_pieces(preds, N_matrix)

    factor = n * (n - 1) / (n - r) ** 2
    Psi    = np.empty((n_eval, J, J))

    for s in range(0, n_eval, chunk):
        e   = min(s + chunk, n_eval)
        cps = cp[:, s:e, :]                                  # (B, c, J)
        V   = np.einsum("bxj,bi->xij", cps, cN) / B          # (c, n, J)
        raw = np.einsum("xij,xik->xjk", V, V)
        zz  = np.einsum("b,bxj,bxk->xjk", w, cps, cps)
        Psi[s:e] = factor * (raw - (zz - B * raw) / (B * (B - 1)))

    Psi = 0.5 * (Psi + np.transpose(Psi, (0, 2, 1)))   # kill float asymmetry
    return tau_B, Psi


def ij_cross_profile(preds, N_matrix, n, r, pairs):
    """
    Cross-profile covariance at COMMON times, for the selected pairs only.

        Gamma_pt[p, j] = Gamma( (x_{k_p}; t_j), (x_{l_p}; t_j) )

    This is the covariance between the influence contributions of a single
    subject at two distinct covariate profiles.  It has no counterpart in a
    pointwise analysis and is what makes the between-patient contrast
    operational.  Only the P selected pairs are formed, so the cost is
    O(P n J) rather than the O((n_eval J)^2) of a full grid covariance.
    """
    B, n_eval, J = preds.shape
    _, cp, cN, w = _ij_pieces(preds, N_matrix)
    factor = n * (n - 1) / (n - r) ** 2

    P = len(pairs)
    Gam = np.empty((P, J))
    Vcache = {}

    def V_of(k):
        if k not in Vcache:
            Vcache[k] = np.einsum("bj,bi->ij", cp[:, k, :], cN) / B   # (n, J)
        return Vcache[k]

    for p, (k, l) in enumerate(pairs):
        Vk, Vl = V_of(int(k)), V_of(int(l))
        raw = np.einsum("ij,ij->j", Vk, Vl)
        zz  = np.einsum("b,bj,bj->j", w, cp[:, int(k), :], cp[:, int(l), :])
        Gam[p] = factor * (raw - (zz - B * raw) / (B * (B - 1)))
    return Gam


# =============================================================================
# 6b.  Contrast pair selection  (fixed across replications)
# =============================================================================

def select_profile_pairs(tau0, t_ref, n_pairs=N_PT_PAIRS,
                         band_q=PT_BAND_Q, gap_frac=PT_GAP_FRAC):
    """
    Choose pairs (k, l) of evaluation profiles for the between-patient
    contrast, using ONLY the true CATE at a reference time, so the selection
    is deterministic and identical in every replication.

    Both members are required to lie in the central band of tau_0(.; t_ref)
    and their gap is targeted at `gap_frac` of the band width.  Selection is
    on the true surface rather than on the fitted one, so no post-hoc
    selection enters the coverage calculation.
    """
    v  = tau0[:, t_ref]
    lo, hi = np.quantile(v, band_q[0]), np.quantile(v, band_q[1])
    band = np.where((v >= lo) & (v <= hi))[0]
    if len(band) < 2:
        band = np.argsort(v)[len(v) // 4: 3 * len(v) // 4]

    target = gap_frac * (hi - lo)
    order  = band[np.argsort(v[band])]

    cand = []
    for a_i in range(len(order)):
        for b_i in range(a_i + 1, len(order)):
            k, l = order[a_i], order[b_i]
            cand.append((abs(abs(v[l] - v[k]) - target), int(k), int(l)))
    cand.sort()

    used, pairs = set(), []
    for _, k, l in cand:                      # greedy, no profile reused
        if k in used or l in used:
            continue
        pairs.append((k, l))
        used.update((k, l))
        if len(pairs) >= n_pairs:
            break
    return np.asarray(pairs, dtype=int), float(target), float(lo), float(hi)


def select_time_pair(J):
    """
    Time indices for the temporal contrast.  The two ends of the grid are
    used: they are the most widely separated pair available, so the
    single-overlap correlation rho_tm is furthest from one, which is the
    non-degeneracy condition the contrast requires.  rho_tm is recorded in
    the output so this can be checked rather than assumed.
    """
    if J < 2:
        return None
    return 0, J - 1


# =============================================================================
# 7.  One replication
# =============================================================================

def run_replication(args):
    case  = args.case
    rng_d = np.random.default_rng(SEED_BASE + args.rep_id * 7919 + 1000 * case)
    rng_b = np.random.default_rng(SEED_BASE + args.rep_id * 7919 + 1 + 1000 * case)

    t_grid = build_time_grid(args.J, case,
                             q_lo=args.grid_q_lo, q_hi=args.grid_q_hi)
    X_eval, idx_lo, idx_hi = build_eval_points(case)
    n_eval, J = X_eval.shape[0], args.J
    tau0 = true_cate(X_eval, t_grid, case)

    m    = min(args.m, J)
    Bmat = build_spline_basis(t_grid, m, degree=args.degree)
    bdiag = basis_diagnostics(Bmat)

    # Assumption (network architecture) scales the sparsity budget with the
    # basis dimension, s \asymp m s_0.  With d = 10 the parameter count is
    # d h_1 + h_1 h_2 + h_2 m, so holding the hidden widths fixed leaves s
    # almost constant in m and the stochastic term m Phi_n is not realised.
    # Scaling both hidden widths by sqrt(m) makes h_1 h_2, the dominant term,
    # proportional to m, which is the intended growth.
    fac = {"fixed": 1.0,
           "sqrt_m": float(np.sqrt(m)),
           "m": float(m)}[args.width_scale]
    h1_eff = int(np.ceil(args.h1 * fac))
    h2_eff = int(np.ceil(args.h2 * fac))
    n_par  = D_COV * h1_eff + h1_eff * h2_eff + h2_eff * m

    t_ref = J // 2
    pt_pairs, pt_target, pt_lo, pt_hi = select_profile_pairs(tau0, t_ref)
    tm_pair = select_time_pair(J)

    case_name = {1: "correct S / correct e",
                 2: "correct S / WRONG e",
                 3: "WRONG S / correct e"}[case]
    print(f"[rep {args.rep_id}] CASE {case}: {case_name}")
    print(f"[rep {args.rep_id}] J={J}  m={m}  alpha={args.alpha}")
    print(f"[rep {args.rep_id}] width_scale={args.width_scale}  "
          f"hidden={h1_eff}x{h2_eff}  trainable weights={n_par:,}")
    print(f"[rep {args.rep_id}] t_grid[0]={t_grid[0]:.4f} "
          f"t_grid[-1]={t_grid[-1]:.4f}")
    print(f"[rep {args.rep_id}] basis rank={bdiag['basis_rank']}/{m}  "
          f"C_B={bdiag['basis_CB']:.3f}  "
          f"min col support={bdiag['basis_min_col_support']}")
    print(f"[rep {args.rep_id}] index band [{idx_lo:.3f}, {idx_hi:.3f}]  "
          f"tau0 in [{tau0.min():+.4f}, {tau0.max():+.4f}]")
    print(f"[rep {args.rep_id}] {len(pt_pairs)} profile pairs, "
          f"target gap {pt_target:+.4f} at t_ref={t_ref}")

    if bdiag["basis_rank"] < m:
        raise RuntimeError("readout matrix is rank deficient: reduce m or "
                           "refine the grid")

    # ---- Stage 1: data + cross-fitted pseudo-outcomes --------------------
    data = generate_data(N_TRAIN, rng_d, case)
    print(f"[rep {args.rep_id}] censoring={1 - data['Delta'].mean():.3f}  "
          f"treated={data['W'].mean():.3f}")

    phi, diag = build_pseudo_outcomes(data, t_grid, case, K=args.K, rng=rng_d)
    diag.update(bdiag)
    diag["censor_rate"]  = float(1 - data["Delta"].mean())
    diag["treated_frac"] = float(data["W"].mean())

    print(f"[rep {args.rep_id}] PITFALL CHECKS")
    print(f"    var_ratio_phi     = {diag['var_ratio_phi']:9.2f}  [<200 ok]")
    print(f"    e in [{diag['e_min']:.3f},{diag['e_max']:.3f}] "
          f"clip={diag['e_clip_frac']:.3f}  [clip<0.05 ok]")
    print(f"    G_min(last t)     = {diag['G_min_last_t']:9.4f}  [>0.10 ok]")
    print(f"    at_risk_frac_last = {diag['at_risk_frac_last']:9.4f}  [>0.15 ok]")

    # ---- Stage 2: ESM ----------------------------------------------------
    r = int(N_TRAIN ** args.alpha)
    print(f"[rep {args.rep_id}] training B={args.B} bags r={r} "
          f"n_jobs={args.n_jobs} ...")
    preds, N_mat = train_ensemble(
        data["X"], phi, X_eval, Bmat, B=args.B, r=r, rng=rng_b,
        h1=h1_eff, h2=h2_eff, n_epochs=args.n_epochs,
        lr=args.lr, weight_decay=args.weight_decay, n_jobs=args.n_jobs)

    # ---- Stage 3: IJ covariance -----------------------------------------
    tau_B, Psi = ij_within_profile(preds, N_mat, n=N_TRAIN, r=r)
    Gam_pt = (ij_cross_profile(preds, N_mat, N_TRAIN, r, pt_pairs)
              if len(pt_pairs) else np.zeros((0, J)))

    # ---- Stage 4: inference ---------------------------------------------
    z    = 1.959963985
    var  = np.maximum(np.einsum("xjj->xj", Psi), 1e-16)
    se   = np.sqrt(var)
    rows = []

    # (a) pointwise CATE at (x*, t_j)
    lo, hi = tau_B - z * se, tau_B + z * se
    rows.append(pd.DataFrame({
        "estimand": "point",
        "idx_a": np.repeat(np.arange(n_eval), J),
        "idx_b": -1,
        "t_idx_a": np.tile(np.arange(J), n_eval),
        "t_idx_b": -1,
        "t_a": np.tile(t_grid, n_eval),
        "t_b": np.nan,
        "target": tau0.ravel(), "est": tau_B.ravel(),
        "se": se.ravel(), "ci_lo": lo.ravel(), "ci_hi": hi.ravel(),
        "covered": ((tau0 >= lo) & (tau0 <= hi)).astype(float).ravel(),
        "rho": np.nan,
        "prog_index": np.repeat(
            X_eval @ (BETA_C12 if case in (1, 2) else BETA_C3), J),
    }))

    # (b) temporal contrast at a fixed profile: tau(x*;t_j') - tau(x*;t_j)
    if tm_pair is not None:
        j0, j1 = tm_pair
        psi_h = tau_B[:, j1] - tau_B[:, j0]
        psi_t = tau0[:, j1] - tau0[:, j0]
        vtm   = np.maximum(Psi[:, j1, j1] + Psi[:, j0, j0]
                           - 2 * Psi[:, j0, j1], 1e-16)
        se_tm = np.sqrt(vtm)
        rho_tm = Psi[:, j0, j1] / np.sqrt(np.maximum(
            Psi[:, j0, j0] * Psi[:, j1, j1], 1e-32))
        lo_tm, hi_tm = psi_h - z * se_tm, psi_h + z * se_tm
        rows.append(pd.DataFrame({
            "estimand": "tm",
            "idx_a": np.arange(n_eval), "idx_b": -1,
            "t_idx_a": j0, "t_idx_b": j1,
            "t_a": t_grid[j0], "t_b": t_grid[j1],
            "target": psi_t, "est": psi_h, "se": se_tm,
            "ci_lo": lo_tm, "ci_hi": hi_tm,
            "covered": ((psi_t >= lo_tm) & (psi_t <= hi_tm)).astype(float),
            "rho": rho_tm,
            "prog_index": X_eval @ (BETA_C12 if case in (1, 2) else BETA_C3),
        }))

    # (c) between-patient contrast at a common time:
    #     tau(x1*;t_j) - tau(x2*;t_j)
    if len(pt_pairs):
        ka, kb = pt_pairs[:, 0], pt_pairs[:, 1]
        psi_h = tau_B[ka] - tau_B[kb]                      # (P, J)
        psi_t = tau0[ka] - tau0[kb]
        vpt   = np.maximum(var[ka] + var[kb] - 2 * Gam_pt, 1e-16)
        se_pt = np.sqrt(vpt)
        rho_pt = Gam_pt / np.sqrt(np.maximum(var[ka] * var[kb], 1e-32))
        lo_pt, hi_pt = psi_h - z * se_pt, psi_h + z * se_pt
        P = len(pt_pairs)
        rows.append(pd.DataFrame({
            "estimand": "pt",
            "idx_a": np.repeat(ka, J), "idx_b": np.repeat(kb, J),
            "t_idx_a": np.tile(np.arange(J), P),
            "t_idx_b": np.tile(np.arange(J), P),
            "t_a": np.tile(t_grid, P), "t_b": np.tile(t_grid, P),
            "target": psi_t.ravel(), "est": psi_h.ravel(),
            "se": se_pt.ravel(), "ci_lo": lo_pt.ravel(), "ci_hi": hi_pt.ravel(),
            "covered": ((psi_t >= lo_pt) & (psi_t <= hi_pt))
                       .astype(float).ravel(),
            "rho": rho_pt.ravel(),
            "prog_index": np.nan,
        }))

    df = pd.concat(rows, ignore_index=True)

    for lab, g in df.groupby("estimand"):
        print(f"[rep {args.rep_id}] {lab:>5}: CP={g['covered'].mean():.3f}  "
              f"mean|bias|={np.abs(g['est'] - g['target']).mean():.5f}  "
              f"meanSE={g['se'].mean():.5f}  "
              f"AIL={(g['ci_hi'] - g['ci_lo']).mean():.5f}"
              + ("" if lab == "point"
                 else f"  |rho|max={np.abs(g['rho']).max():.3f}"))

    # ---- metadata and write ----------------------------------------------
    df["rep_id"] = args.rep_id
    df["case"]   = case
    df["alpha"]  = args.alpha
    df["J"], df["m"], df["r"], df["B"] = J, m, r, args.B
    df["h1"], df["h2"] = h1_eff, h2_eff
    df["h1_base"], df["h2_base"] = args.h1, args.h2
    df["width_scale"] = args.width_scale
    df["n_par"] = n_par
    df["degree"] = args.degree
    for k, v in diag.items():
        df[f"diag_{k}"] = v
    df["diag_idx_band_lo"] = idx_lo
    df["diag_idx_band_hi"] = idx_hi
    df["diag_pt_gap_target"] = pt_target
    df["grid_q_lo"] = args.grid_q_lo
    df["grid_q_hi"] = args.grid_q_hi

    os.makedirs(args.out_dir, exist_ok=True)
    wtag = "" if args.width_scale == "fixed" else f"_w{args.width_scale}"
    tag = (f"c{case}_a{args.alpha:.2f}_J{J}_m{m}_h{h1_eff}x{h2_eff}{wtag}"
           f"_B{args.B}_rep{args.rep_id}")
    out = os.path.join(args.out_dir, f"cate_{tag}.csv")
    df.to_csv(out, index=False)
    print(f"[rep {args.rep_id}] saved {out}  ({len(df)} rows)")


# =============================================================================
# 8.  Entry point
# =============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="CATE ESM + IJ inference -- one replication")
    p.add_argument("--rep_id", type=int, required=True)
    p.add_argument("--case",   type=int, default=1, choices=[1, 2, 3],
                   help="1: correct S / correct e   "
                        "2: correct S / wrong e   "
                        "3: wrong S / correct e")
    p.add_argument("--alpha",  type=float, default=0.90,
                   help="subsample exponent; r = n^alpha "
                        "(grid 0.80 0.85 0.90 0.95)")
    p.add_argument("--J",      type=int, default=50,
                   help="number of time points (paper uses 25, 50, 100)")
    p.add_argument("--m",      type=int, default=4,
                   help="temporal basis dimension, m <= J; "
                        "m = J gives the cardinal basis and reproduces an "
                        "unconstrained J-output network")
    p.add_argument("--degree", type=int, default=SPLINE_DEGREE,
                   help="B-spline degree (3 = cubic)")
    p.add_argument("--B",      type=int, default=1000)
    p.add_argument("--K",      type=int, default=5, help="cross-fitting folds")
    p.add_argument("--width_scale", type=str, default="fixed",
                   choices=["fixed", "sqrt_m", "m"],
                   help="how the hidden widths grow with m.  'fixed' keeps "
                        "h1 x h2 for every m, so the sparsity budget is "
                        "almost constant in m; 'sqrt_m' multiplies both "
                        "widths by sqrt(m), making the trainable weight "
                        "count proportional to m as Assumption (network "
                        "architecture) specifies; 'm' is the aggressive "
                        "variant and is rarely affordable")
    p.add_argument("--h1",     type=int, default=128)
    p.add_argument("--h2",     type=int, default=64)
    p.add_argument("--n_epochs",     type=int,   default=400)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--n_jobs",       type=int,   default=8)
    p.add_argument("--grid_q_lo",    type=float, default=GRID_Q_LO)
    p.add_argument("--grid_q_hi",    type=float, default=GRID_Q_HI)
    p.add_argument("--out_dir",      type=str,   default="results")
    args = p.parse_args()

    run_replication(args)
