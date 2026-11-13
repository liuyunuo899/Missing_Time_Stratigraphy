# -*- coding: utf-8 -*-
"""Variational inversion of an age-depth record into deposition and erosion rates.

This is the numerical realisation of the functional proposed in the source
conversation:

    J[d, e] = sum_i ( h(t_i) - z_i )^2 / (2 sigma_i^2)
              + (lambda_d/2) integral d^2 dt
              + (lambda_e/2) integral e^2 dt
    subject to  dh/dt = d - e,  d >= 0,  e >= 0

Two points about interpretation are enforced by the code design:

1. The quadratic penalties are treated as a *statistical prior*, not as a law of
   nature. Every experiment therefore varies lambda and reports how the answer
   moves, rather than reporting one solution as if it were unique.
2. Regularisation is not claimed to be optional: the prior-indexed ensemble is
   interpreted jointly with data misfit. Solutions that fail the declared misfit
   screen are not treated as equally data-admissible histories.

Solver I: accelerated projected gradient (FISTA) written here, with the step
taken from the largest Hessian eigenvalue obtained by power iteration.
Solver II: scipy L-BFGS-B with non-negativity bounds, used only to cross-check
Solver I.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class InverseProblem:
    n: int                 # number of time steps
    dt: float
    obs_idx: np.ndarray    # grid index at which each horizon is dated
    obs_z: np.ndarray      # measured height (m)
    obs_sigma: np.ndarray  # height uncertainty (m)
    truth_d: np.ndarray = None
    truth_e: np.ndarray = None


def cumulative_matrix(ip: InverseProblem):
    """Matrix A with rows a_i such that a_i . [d, e] = h(t_i)."""
    n = ip.n
    rows = np.zeros((len(ip.obs_idx), 2 * n))
    for r, k in enumerate(ip.obs_idx):
        rows[r, :k] = ip.dt            # deposition increments up to step k
        rows[r, n:n + k] = -ip.dt      # erosion increments up to step k
    return rows


def objective(ip: InverseProblem, lam_d: float, lam_e: float,
              row_scale=1.0):
    """Return callable objective, gradient and regularisation diagonal."""
    A = cumulative_matrix(ip)
    w = row_scale / np.asarray(ip.obs_sigma, dtype=float)
    reg = np.concatenate([np.full(ip.n, lam_d * ip.dt),
                          np.full(ip.n, lam_e * ip.dt)])
    z = np.asarray(ip.obs_z, dtype=float)

    def obj(x):
        r = (A @ x - z) * w
        return 0.5 * float(np.sum(r ** 2)) + 0.5 * float(np.sum(reg * x ** 2))

    def grad(x):
        r = (A @ x - z) * w
        return A.T @ (r * w) + reg * x

    H_diag = reg
    return A, w, obj, grad, H_diag


def hessian_norm(ip, lam_d, lam_e, row_scale=1.0, iters=60, seed=0):
    """Largest eigenvalue of the quadratic Hessian by power iteration."""
    A, w, _, grad, _ = objective(ip, lam_d, lam_e, row_scale)
    rng = np.random.default_rng(seed)
    n = 2 * ip.n
    reg = np.concatenate([np.full(ip.n, lam_d * ip.dt),
                          np.full(ip.n, lam_e * ip.dt)])
    v = rng.normal(size=n)
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        u = A.T @ ((A @ v) * (w ** 2)) + reg * v
        nrm = np.linalg.norm(u)
        if nrm == 0:
            break
        v = u / nrm
        lam = nrm
    return float(lam)


def solve_fista(ip, lam_d, lam_e, row_scale=1.0, max_iter=4000, tol=1e-10, seed=0):
    """Projected quadratic solver (non-negativity on both d and e)."""
    A, w, obj, grad, _ = objective(ip, lam_d, lam_e, row_scale)
    L = hessian_norm(ip, lam_d, lam_e, row_scale, seed=seed)
    step = 1.0 / max(L, 1e-12)
    n = 2 * ip.n
    x = np.zeros(n)
    y = np.zeros(n)
    t_acc = 1.0
    hist = []
    for k in range(max_iter):
        g = grad(y)
        x_new = np.maximum(y - step * g, 0.0)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_acc ** 2))
        beta = (t_acc - 1.0) / t_new
        y = x_new + beta * (x_new - x)
        if k % 100 == 0:
            hist.append(obj(x_new))
            if len(hist) > 3 and abs(hist[-2] - hist[-1]) < tol * max(1.0, hist[-1]):
                x = x_new
                break
        x, t_acc = x_new, t_new
    # KKT-style diagnostic: projected gradient norm on the active set
    g = grad(x)
    proj_g = np.where((x <= 1e-12) & (g > 0), 0.0, g)
    return {
        "d": x[:ip.n].copy(), "e": x[ip.n:].copy(),
        "objective": obj(x), "steps": k, "step_size": step,
        "stationarity": float(np.linalg.norm(proj_g)),
        "residual": float(np.max(np.abs((A @ x - np.asarray(ip.obs_z)) / ip.obs_sigma))),
        "history": hist,
    }


def solve_lbfgsb(ip, lam_d, lam_e, row_scale=1.0, max_iter=300):
    """Independent bounded L-BFGS-B solver used for verification only."""
    _, _, obj, grad, _ = objective(ip, lam_d, lam_e, row_scale)
    x0 = np.full(2 * ip.n, 1e-6)
    bounds = [(0.0, None)] * (2 * ip.n)
    res = minimize(obj, x0, jac=grad, bounds=bounds, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": 1e-14, "gtol": 1e-12})
    return {"d": res.x[:ip.n].copy(), "e": res.x[ip.n:].copy(),
            "objective": float(res.fun), "success": bool(res.success)}


def recover_metrics(ip, sol):
    """Compare recovered histories with the known truth of a synthetic run."""
    if ip.truth_d is None:
        return {}
    d_true, e_true = ip.truth_d, ip.truth_e
    d_rec, e_rec = sol["d"], sol["e"]
    gross_true = float(np.sum(d_true) * ip.dt)
    gross_rec = float(np.sum(d_rec) * ip.dt)
    ero_true = float(np.sum(e_true) * ip.dt)
    ero_rec = float(np.sum(e_rec) * ip.dt)
    return {
        "d_rmse": float(np.sqrt(np.mean((d_rec - d_true) ** 2))),
        "e_rmse": float(np.sqrt(np.mean((e_rec - e_true) ** 2))),
        "d_corr": float(np.corrcoef(d_rec, d_true)[0, 1]),
        "e_corr": float(np.corrcoef(e_rec, e_true)[0, 1]),
        "gross_deposition_true_m": gross_true,
        "gross_deposition_recovered_m": gross_rec,
        "gross_deposition_ratio": float(gross_rec / gross_true) if gross_true else np.nan,
        "erosion_true_m": ero_true,
        "erosion_recovered_m": ero_rec,
        "preservation_true": float(1.0 - (np.sum(e_true) * ip.dt) / gross_true)
        if gross_true else np.nan,
        "preservation_recovered": float(1.0 - ero_rec / gross_rec)
        if gross_rec else np.nan,
    }


def ensemble(ip, lam_d_grid, lam_e_grid, row_scale=1.0, max_iter=2000):
    """Family of prior-indexed histories obtained by sweeping prior weights.

    Data admissibility is evaluated separately from the returned residuals.
    """
    out = []
    for ld in lam_d_grid:
        for le in lam_e_grid:
            sol = solve_fista(ip, ld, le, row_scale=row_scale, max_iter=max_iter)
            m = recover_metrics(ip, sol)
            out.append({"lam_d": float(ld), "lam_e": float(le),
                        "objective": sol["objective"],
                        "stationarity": sol["stationarity"],
                        "residual": sol["residual"], "metrics": m,
                        "d": sol["d"], "e": sol["e"]})
    return out


def noisy_data_ensemble(ip, sigma_frac, n_rep, rng, lam_d, lam_e, row_scale=1.0,
                        extra_rows=0, max_iter=2000):
    """Repeat the inversion with independently perturbed age-depth data.

    ``extra_rows`` adds extra dated horizons interleaved between the originals,
    which is how the effect of additional chronological information is tested.
    """
    base_z = ip.obs_z.copy()
    gross = []
    pres = []
    for r in range(n_rep):
        z = base_z * (1.0 + rng.normal(0.0, sigma_frac, base_z.size))
        idx = list(ip.obs_idx)
        if extra_rows > 0:
            for k in range(extra_rows):
                lo, hi = 0, ip.n
                pos = int(np.clip(rng.integers(lo, hi), 1, ip.n - 1))
                idx.append(pos)
                truth_h = 0.0
                if ip.truth_d is not None:
                    truth_h = np.cumsum((ip.truth_d - ip.truth_e) * ip.dt)[pos - 1]
                # additional horizons carry the same measurement uncertainty as
                # the original ones, so their value lies in information content
                # only, not in being error free
                truth_h = truth_h * (1.0 + rng.normal(0.0, sigma_frac))
                z = np.append(z, truth_h)
        order = np.argsort(idx)
        ip_r = InverseProblem(n=ip.n, dt=ip.dt, obs_idx=np.array(idx)[order],
                              obs_z=np.array(z)[order],
                              obs_sigma=np.full(len(idx), base_z.mean() * sigma_frac
                                                if sigma_frac > 0 else 1.0),
                              truth_d=ip.truth_d, truth_e=ip.truth_e)
        sol = solve_fista(ip_r, lam_d, lam_e, row_scale=row_scale, max_iter=max_iter)
        m = recover_metrics(ip_r, sol)
        gross.append(m.get("gross_deposition_ratio", np.nan))
        pres.append(m.get("preservation_recovered", np.nan))
    gross = np.array(gross, dtype=float)
    pres = np.array(pres, dtype=float)
    return {
        "n_replicates": n_rep, "extra_horizons": extra_rows,
        "sigma_fraction": sigma_frac,
        "gross_ratio_median": float(np.nanmedian(gross)),
        "gross_ratio_iqr": [float(np.nanquantile(gross, 0.25)),
                            float(np.nanquantile(gross, 0.75))],
        "gross_ratio_range": [float(np.nanmin(gross)), float(np.nanmax(gross))],
        "preservation_median": float(np.nanmedian(pres)),
        "preservation_iqr": [float(np.nanquantile(pres, 0.25)),
                             float(np.nanquantile(pres, 0.75))],
    }
