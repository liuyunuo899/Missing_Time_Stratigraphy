# -*- coding: utf-8 -*-
"""Round-2 computations requested in review.

1. Section-level completeness.  Case Study I computes completeness on the
   lateral union of a column's sections.  Here it is recomputed per section, to
   test whether the completeness level is an artefact of the union convention.

2. Prior-family sensitivity of the variational inverse.  The quadratic penalty
   of Eq. (6) is one choice; a sparsity-promoting L1 penalty is the natural
   alternative for an episodic history.  Both are solved with the identical
   FISTA machinery, differing only in the proximal operator, so any difference
   in the recovered gross deposition is attributable to the prior.

Writes data/processed/summary_round2_computations.json and
data/synthetic/inversion_l1.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed")
SYN = os.path.join(ROOT, "data", "synthetic")

from stats_tools import completeness_at_resolution  # noqa: E402
from simulation import run_case  # noqa: E402
from inversion import InverseProblem, cumulative_matrix, hessian_norm  # noqa: E402
from experiments_synthetic import N_STEPS, DT, SEED  # noqa: E402

TAU_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
MIN_TAU_MULTIPLE = 10.0


# ---------------------------------------------------------------- L1 solver
def solve_fista_l1(ip, lam_d, lam_e, max_iter=4000, tol=1e-10, seed=0):
    """FISTA with an L1 penalty lambda_d*|d| + lambda_e*|e| and x >= 0.

    Identical to the quadratic solver except in the proximal step, where the
    soft threshold replaces the plain projection. The data term and the step
    length are unchanged, so the comparison isolates the prior.
    """
    A = cumulative_matrix(ip)
    w = 1.0 / np.asarray(ip.obs_sigma, dtype=float)
    z = np.asarray(ip.obs_z, dtype=float)
    n = ip.n
    lam_vec = np.concatenate([np.full(n, lam_d * ip.dt),
                              np.full(n, lam_e * ip.dt)])

    # step from the data-term Hessian only (the L1 term has no Hessian);
    # use the quadratic machinery with zero regularisation for the norm
    class _Q(InverseProblem):
        pass
    L = hessian_norm(ip, 0.0, 0.0, seed=seed)
    step = 1.0 / max(L, 1e-12)

    def grad(x):
        return A.T @ ((A @ x - z) * (w ** 2))

    def obj(x):
        r = (A @ x - z) * w
        return 0.5 * float(np.sum(r ** 2)) + float(np.sum(lam_vec * np.abs(x)))

    m = 2 * n
    x = np.zeros(m)
    y = np.zeros(m)
    t_acc = 1.0
    hist = []
    for k in range(max_iter):
        g = grad(y)
        x_new = np.maximum(y - step * g - step * lam_vec, 0.0)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_acc ** 2))
        beta = (t_acc - 1.0) / t_new
        y = x_new + beta * (x_new - x)
        if k % 100 == 0:
            hist.append(obj(x_new))
            if len(hist) > 3 and abs(hist[-2] - hist[-1]) < tol * max(1.0, hist[-1]):
                x = x_new
                break
        x, t_acc = x_new, t_new
    return {"d": x[:n].copy(), "e": x[n:].copy(), "objective": obj(x),
            "residual": float(np.max(np.abs((A @ x - z) / ip.obs_sigma)))}


def section_completeness():
    units = pd.read_csv(os.path.join(PROC, "units_processed.csv"))
    if "is_phanerozoic" in units:
        units = units[units["is_phanerozoic"]]
    counts = units.groupby("col_id").size()
    prim_cols = counts[counts >= 15].index
    ph = units[units["col_id"].isin(prim_cols)]

    rows = []
    for sid, sub in ph.groupby("section_id"):
        t = sub["t_age"].to_numpy(dtype=float)
        b = sub["b_age"].to_numpy(dtype=float)
        ok = b > t
        t, b = t[ok], b[ok]
        if t.size < 3:
            continue
        o = np.argsort(t)
        t, b = t[o], b[o]
        start, end = float(t.min()), float(b.max())
        L = end - start
        iv = list(zip(t, b))
        for tau in TAU_GRID:
            if L < MIN_TAU_MULTIPLE * tau or tau >= L:
                continue
            rows.append({"section_id": int(sid), "tau_Myr": tau,
                         "record_length_Myr": L,
                         "completeness": completeness_at_resolution(iv, tau,
                                                                    start, end)})
    df = pd.DataFrame(rows)
    summ = (df.groupby("tau_Myr")["completeness"]
              .agg(n="size", mean="mean", median="median").reset_index())
    summ.to_csv(os.path.join(PROC, "completeness_sections.csv"), index=False)
    return summ


def l1_sweep():
    res = run_case("multiscale", n=N_STEPS, dt=DT, seed=11,
                   forcing_kwargs=dict(beta=1.2, n_scales=6), theta_dep=0.0,
                   deposit_rate=25.0, erosion_rate=40.0)
    inc_true = np.diff(np.concatenate([[0.0], res.h]))
    d_true = np.maximum(inc_true, 0.0) / res.dt
    e_true = np.maximum(-inc_true, 0.0) / res.dt
    h_true = res.h.copy()
    n = N_STEPS
    rng = np.random.default_rng(SEED + 5)
    idx = np.sort(rng.choice(np.arange(1, n), size=12, replace=False))
    sigma_z = max(5.0, 0.02 * float(np.abs(h_true[idx]).max() + 1.0))
    obs_z = h_true[idx] + rng.normal(0.0, sigma_z, idx.size)
    ip = InverseProblem(n=n, dt=res.dt, obs_idx=idx, obs_z=obs_z,
                        obs_sigma=np.full(idx.size, sigma_z),
                        truth_d=d_true, truth_e=e_true)
    gross_true = float(d_true.sum() * res.dt)

    rows = []
    for lam in [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]:
        sol = solve_fista_l1(ip, lam, lam, max_iter=6000)
        gross_rec = float(sol["d"].sum() * res.dt)
        rows.append({"lam": lam, "gross_ratio": gross_rec / gross_true,
                     "residual_sigma": sol["residual"],
                     "objective": sol["objective"]})
        print("   L1 lam=%g -> gross ratio %.3f, misfit %.2f sigma"
              % (lam, gross_rec / gross_true, sol["residual"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SYN, "inversion_l1.csv"), index=False)
    return df


def main():
    out = {}
    print("== section-level completeness ==")
    summ = section_completeness()
    comp_cols = pd.read_csv(os.path.join(PROC, "completeness_pooled.csv"))
    merged = summ.merge(comp_cols[["tau_Myr", "mean_completeness"]],
                        on="tau_Myr", suffixes=("_section", "_column"))
    print(merged.to_string(index=False))
    out["section_vs_column_completeness"] = merged.rename(columns={
        "mean": "section_mean", "median": "section_median",
        "mean_completeness": "column_mean"}).to_dict("records")

    print("== L1 prior sweep ==")
    l1 = l1_sweep()
    out["l1_sweep"] = l1.to_dict("records")

    with open(os.path.join(PROC, "summary_round2_computations.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("written summary_round2_computations.json")


if __name__ == "__main__":
    main()
