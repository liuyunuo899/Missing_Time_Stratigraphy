# -*- coding: utf-8 -*-
"""Round 3 review computations.

Five new computations requested by the third review round, each answering a
specific reviewer comment:

  R3-A  Goodness-of-fit p-values sit on the Monte-Carlo floor 1/(R+1).
        Re-run the pooled and regional GOF tests with R = 3000 so that the
        p-value is resolved rather than censored, and report the floor.
  R3-B  The KS-minimising xmin and the KS goodness-of-fit test use the same
        statistic on the same data, so the test is not calibrated.  Calibrate
        it: simulate from a *true* power law with the fitted parameters, push
        the simulations through the identical pipeline, and measure how often
        a true power law is rejected.
  R3-C  The reported bootstrap CI for alpha is wide and asymmetric.  Separate
        the two sources of spread: parameter uncertainty at fixed xmin, and
        xmin-selection uncertainty.  Report P(alpha > 2).
  R3-D  Leave-one-region-out: is the pooled exponent set by one region?
  R3-E  Cross-estimator consistency. The completeness curve gives a second,
        assumption-dependent estimate of alpha through b = 2 - alpha. Bootstrap
        it and compare it with the tail estimate without applying a common
        additive correction, which cannot change their difference.

All randomness uses numpy.random.default_rng(seed) with seeds recorded below.
Outputs: data/processed/summary_round3_computations.json plus two CSVs.
"""
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
PROC = ROOT + "/data/processed"
SYNT = ROOT + "/data/synthetic"
sys.path.insert(0, ROOT + "/src")

from stats_tools import (_tail, continuous_powerlaw_mle, fit_powerlaw,  # noqa: E402
                         ks_distance, powerlaw_gof_pvalue)

SEED = 20261020
GOF_ROUND3 = 3000          # replicates for the resolved GOF test
GOF_CALIB = 150            # replicates inside each calibration replicate
N_CALIB = 100              # calibration datasets drawn from a true power law
N_BOOT_FIXED = 3000        # bootstrap replicates at fixed xmin
N_BOOT_XMIN = 300          # bootstrap replicates with xmin re-estimated
N_BOOT_COLUMN = 3000       # bootstrap replicates over columns for b

REGIONS = ["Atlantic Basin", "Northern Appalachians", "Great Basin",
           "Southwest Mid-Continent"]


def log(msg):
    print(msg, flush=True)


def primary_subset(gaps):
    """Exactly the subset used for `primary_all` in analysis_real.py.

    Columns holding at least MIN_UNITS_PRIMARY = 15 units (a unit count, not a
    gap count) and strictly positive gap durations. Round-3 numbers must be
    computed on the same rows as the published ones or they are not comparable.
    """
    per_col = gaps.groupby("col_id").agg(n_units=("n_units_column", "first"))
    ids = per_col.index[per_col["n_units"] >= 15]
    prim = gaps[gaps["col_id"].isin(ids)].copy()
    prim = prim[prim["hiatus_duration_myr"] > 0]
    return prim


# ------------------------------------------------------------------ R3-A
def gof_resolution(vals_by_group):
    """GOF p-value with enough replicates that the p-value is not censored."""
    out = {}
    for name, vals in vals_by_group.items():
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size < 40:
            continue
        res = fit_powerlaw(vals)
        a, xmin = res["alpha"], res["xmin"]
        tail = _tail(vals, xmin)
        low = powerlaw_gof_pvalue(vals, n_mc=300, seed=SEED,
                                  return_details=True)
        high = powerlaw_gof_pvalue(vals, n_mc=GOF_ROUND3, seed=SEED,
                                   return_details=True)
        p_low = low["p_value"]
        p_hi = high["p_value"]
        out[name] = {
            "n": int(vals.size), "alpha": float(a), "xmin_Myr": float(xmin),
            "n_tail": int(tail.size), "ks": float(ks_distance(tail, a, xmin)),
            "gof_p_R300": float(p_low), "gof_p_R3000": float(p_hi),
            "gof_mc_valid_R300": int(low["n_mc_valid"]),
            "gof_mc_valid_R3000": int(high["n_mc_valid"]),
            "floor_R300": float(low["mc_floor"]),
            "floor_R3000": float(high["mc_floor"]),
            "mc_floor_R3000": float(high["mc_floor"]),
            "censored_at_R300": bool(p_low <= low["mc_floor"] + 1e-12),
            "censored_at_R3000": bool(p_hi <= high["mc_floor"] + 1e-12),
            "method": high["method"],
        }
        log("R3-A %-34s alpha=%.3f xmin=%7.3f n_tail=%5d  p(R=300)=%.4f  p(R=%d)=%.4f"
            % (name, a, xmin, tail.size, p_low, GOF_ROUND3, p_hi))
    return out


# ------------------------------------------------------------------ R3-B
def circularity_calibration(vals, alpha, xmin):
    """How often does this exact pipeline reject a *true* power law?"""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    below = vals[vals < xmin]
    n_tail = int((vals >= xmin).sum())
    ps, alphas, xmins = [], [], []
    for k in range(N_CALIB):
        rng = np.random.default_rng(SEED + 1000 + k)
        u = rng.random(n_tail)
        synth_tail = xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))
        if below.size:
            synth_below = rng.choice(below, size=below.size, replace=True)
            synth = np.concatenate([synth_tail, synth_below])
        else:
            synth = synth_tail
        fit = fit_powerlaw(synth)
        a_h, x_h = fit["alpha"], fit["xmin"]
        # Independent streams for the outer dataset and its nested GOF test.
        p = powerlaw_gof_pvalue(synth, n_mc=GOF_CALIB,
                                seed=SEED + 5000 + k)
        ps.append(p)
        alphas.append(a_h)
        xmins.append(x_h)
    ps = np.asarray(ps)
    alphas = np.asarray(alphas)
    xmins = np.asarray(xmins)
    out = {
        "n_datasets": N_CALIB, "gof_replicates_each": GOF_CALIB,
        "outer_seed_base": SEED + 1000, "nested_gof_seed_base": SEED + 5000,
        "design": "parametric Pareto tail plus nonparametric below-threshold "
                  "bootstrap; the full fitting-and-GOF pipeline is rerun",
        "true_alpha": float(alpha), "true_xmin_Myr": float(xmin),
        "n_tail": n_tail, "n_below_xmin": int(below.size),
        "median_p": float(np.median(ps)),
        "p_quantiles": [float(q) for q in np.quantile(ps, [0.05, 0.25, 0.5, 0.75, 0.95])],
        "false_rejection_rate_at_0.10": float(np.mean(ps < 0.10)),
        "false_rejection_rate_at_0.05": float(np.mean(ps < 0.05)),
        "median_alpha_recovered": float(np.median(alphas)),
        "alpha_recovered_ci95": [float(q) for q in np.quantile(alphas, [0.025, 0.975])],
        "median_xmin_recovered_Myr": float(np.median(xmins)),
        "xmin_recovered_ci95_Myr": [float(q) for q in np.quantile(xmins, [0.025, 0.975])],
        "p_values": [float(v) for v in ps],
        "alphas_recovered": [float(v) for v in alphas],
    }
    log("R3-B circularity: %d true power laws -> median p=%.3f, "
        "false-rejection rate at 0.10 = %.3f (nominal 0.100)"
        % (N_CALIB, out["median_p"], out["false_rejection_rate_at_0.10"]))
    log("R3-B recovery: alpha %.3f (95%% [%.3f, %.3f]) vs true %.3f"
        % (out["median_alpha_recovered"], out["alpha_recovered_ci95"][0],
           out["alpha_recovered_ci95"][1], alpha))
    return out


# ------------------------------------------------------------------ R3-C
def alpha_uncertainty(vals, alpha_hat, xmin_hat):
    """Split bootstrap uncertainty: fixed xmin vs re-estimated xmin."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    tail = _tail(vals, xmin_hat)
    n = tail.size

    def mle_fixed(s):
        """MLE with xmin held at the observed value (not min of the resample)."""
        return continuous_powerlaw_mle(s, xmin=xmin_hat)

    rng = np.random.default_rng(SEED)
    fixed = np.empty(N_BOOT_FIXED)
    for i in range(N_BOOT_FIXED):
        fixed[i] = mle_fixed(tail[rng.integers(0, n, n)])

    rng2 = np.random.default_rng(SEED + 1)
    free_a, free_x = [], []
    for i in range(N_BOOT_XMIN):
        s = vals[rng2.integers(0, vals.size, vals.size)]
        f = fit_powerlaw(s)
        free_a.append(f["alpha"])
        free_x.append(f["xmin"])
    free_a = np.asarray(free_a)
    free_x = np.asarray(free_x)

    out = {
        "n_boot_fixed_xmin": N_BOOT_FIXED, "n_boot_free_xmin": N_BOOT_XMIN,
        "alpha_hat": float(alpha_hat), "xmin_Myr": float(xmin_hat),
        "fixed_xmin": {
            "mean": float(fixed.mean()),
            "sd": float(fixed.std(ddof=1)),
            "ci95": [float(q) for q in np.quantile(fixed, [0.025, 0.975])],
            "p_alpha_gt_2": float(np.mean(fixed > 2.0)),
            "distribution": [float(v) for v in fixed],
        },
        "free_xmin": {
            "mean": float(free_a.mean()), "sd": float(free_a.std(ddof=1)),
            "ci95": [float(q) for q in np.quantile(free_a, [0.025, 0.975])],
            "p_alpha_gt_2": float(np.mean(free_a > 2.0)),
            "xmin_ci95_Myr": [float(q) for q in np.quantile(free_x, [0.025, 0.975])],
            "xmin_median_Myr": float(np.median(free_x)),
            "distribution": [float(v) for v in free_a],
        },
    }
    log("R3-C fixed xmin : alpha %.3f +/- %.3f, CI [%.3f, %.3f], P(a>2)=%.3f"
        % (out["fixed_xmin"]["mean"], out["fixed_xmin"]["sd"],
           out["fixed_xmin"]["ci95"][0], out["fixed_xmin"]["ci95"][1],
           out["fixed_xmin"]["p_alpha_gt_2"]))
    log("R3-C free  xmin : alpha %.3f +/- %.3f, CI [%.3f, %.3f], P(a>2)=%.3f, xmin CI [%.2f, %.2f]"
        % (out["free_xmin"]["mean"], out["free_xmin"]["sd"],
           out["free_xmin"]["ci95"][0], out["free_xmin"]["ci95"][1],
           out["free_xmin"]["p_alpha_gt_2"],
           out["free_xmin"]["xmin_ci95_Myr"][0], out["free_xmin"]["xmin_ci95_Myr"][1]))
    return out


# ------------------------------------------------------------------ R3-D
def leave_one_region_out(prim):
    out = {}
    pooled = np.asarray(prim["hiatus_duration_myr"], dtype=float)
    pooled = pooled[np.isfinite(pooled) & (pooled > 0)]
    base = fit_powerlaw(pooled)
    out["pooled"] = {"alpha": float(base["alpha"]), "xmin_Myr": float(base["xmin"]),
                     "n": int(pooled.size), "n_tail": int(base["n_tail"])}
    for r in REGIONS:
        sub = prim[prim["col_group"] == r]["hiatus_duration_myr"]
        sub = np.asarray(sub, dtype=float)
        sub = sub[np.isfinite(sub) & (sub > 0)]
        keep = prim[prim["col_group"] != r]["hiatus_duration_myr"]
        keep = np.asarray(keep, dtype=float)
        keep = keep[np.isfinite(keep) & (keep > 0)]
        f = fit_powerlaw(keep)
        out["drop_" + r] = {
            "n_removed": int(sub.size), "n_remaining": int(keep.size),
            "alpha": float(f["alpha"]), "xmin_Myr": float(f["xmin"]),
            "n_tail": int(f["n_tail"]),
            "delta_alpha": float(f["alpha"] - base["alpha"]),
            "region_own_n": int(sub.size),
            "region_own_alpha": (float(fit_powerlaw(sub)["alpha"])
                                 if sub.size >= 40 else None),
        }
        log("R3-D drop %-24s n_remaining=%5d alpha=%.3f (delta %+0.3f)"
            % (r, keep.size, f["alpha"], f["alpha"] - base["alpha"]))
    return out


# ------------------------------------------------------------------ R3-E
def cross_estimator(curves):
    """Second, assumption-dependent summary from the completeness curve.

    For each column, regress ln(1 - completeness) on ln(tau) over the tau grid
    to obtain b. The algebraic value alpha_eff = 2 - b is interpreted only as a
    conditional mapping on the asymptotic alpha > 2 branch; it is not an
    independent estimate because it uses the same columns and age model.
    """
    rows = []
    for cid, g in curves.groupby("col_id"):
        g = g.sort_values("tau_Myr")
        tau = g["tau_Myr"].to_numpy(float)
        comp = g["completeness"].to_numpy(float)
        miss = 1.0 - comp
        ok = np.isfinite(miss) & (miss > 0) & np.isfinite(tau) & (tau > 0)
        if ok.sum() < 6:
            continue
        b = np.polyfit(np.log(tau[ok]), np.log(miss[ok]), 1)[0]
        rows.append({"col_id": cid, "b": float(b)})
    per_col = pd.DataFrame(rows)
    b_vals = per_col["b"].to_numpy(float)
    b_vals = b_vals[np.isfinite(b_vals)]

    rng = np.random.default_rng(SEED + 2)
    n = b_vals.size
    boot = np.array([b_vals[rng.integers(0, n, n)].mean() for _ in range(N_BOOT_COLUMN)])

    b_hat = float(b_vals.mean())
    ci = [float(q) for q in np.quantile(boot, [0.025, 0.975])]
    out = {
        "n_columns": int(n),
        "b_mean": b_hat, "b_median": float(np.median(b_vals)),
        "b_sd_columns": float(b_vals.std(ddof=1)),
        "b_ci95_mean": ci,
        "alpha_eff_from_curve": float(2.0 - b_hat),
        "alpha_eff_ci95": [float(2.0 - ci[1]), float(2.0 - ci[0])],
        "point_consistent_with_alpha_gt2_branch": bool(2.0 - b_hat > 2.0),
        "ci_entirely_alpha_gt2": bool(2.0 - ci[1] > 2.0),
        "interpretation": "conditional algebraic mapping, not an independent or "
                          "unconditionally validated tail-exponent estimate",
        "bootstrap_distribution_b": [float(v) for v in boot],
        "per_column_b": [float(v) for v in b_vals],
    }
    log("R3-E curve-based b = %.4f (95%% CI of the column mean [%.4f, %.4f])"
        % (b_hat, ci[0], ci[1]))
    log("R3-E implied alpha_eff = %.3f, CI [%.3f, %.3f]"
        % (out["alpha_eff_from_curve"], out["alpha_eff_ci95"][0],
           out["alpha_eff_ci95"][1]))
    per_col.to_csv(PROC + "/completeness_local_exponent_by_column.csv", index=False)
    return out


def main():
    t0 = datetime.now(timezone.utc)
    gaps = pd.read_csv(PROC + "/hiatus_gaps.csv")
    curves = pd.read_csv(PROC + "/completeness_curves.csv")
    prim = primary_subset(gaps)

    with open(PROC + "/summary_real_stats.json", encoding="utf-8") as fh:
        prev = json.load(fh)
    pooled_entry = prev["duration_distributions"]["primary_all"]
    alpha_hat = pooled_entry["alpha"]
    xmin_hat = pooled_entry["xmin_Myr"]

    pooled = np.asarray(prim["hiatus_duration_myr"], dtype=float)
    pooled = pooled[np.isfinite(pooled) & (pooled > 0)]
    log("primary subset: n_gaps=%d (expected %d)"
        % (pooled.size, pooled_entry["n"]))

    groups = {"primary_all": pooled}
    for r in REGIONS:
        s = prim[prim["col_group"] == r]["hiatus_duration_myr"]
        groups["region_" + r.replace(" ", "_")] = np.asarray(s, dtype=float)

    result = {"generated_utc": t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "seed": SEED,
              "seeds": {"gof_resolution": SEED,
                        "circularity_outer_base": SEED + 1000,
                        "circularity_nested_gof_base": SEED + 5000,
                        "bootstrap_fixed": SEED, "bootstrap_free": SEED + 1,
                        "bootstrap_columns": SEED + 2},
              "replicate_counts": {"gof_R3000": GOF_ROUND3,
                                   "gof_calibration_each": GOF_CALIB,
                                   "calibration_datasets": N_CALIB,
                                   "bootstrap_fixed_xmin": N_BOOT_FIXED,
                                   "bootstrap_free_xmin": N_BOOT_XMIN,
                                   "bootstrap_columns": N_BOOT_COLUMN}}

    log("--- R3-A  resolved goodness-of-fit ---")
    result["gof_resolution"] = gof_resolution(groups)

    log("--- R3-B  calibration of the KS pipeline ---")
    result["circularity_calibration"] = circularity_calibration(
        pooled, alpha_hat, xmin_hat)

    log("--- R3-C  alpha uncertainty decomposition ---")
    result["alpha_uncertainty"] = alpha_uncertainty(pooled, alpha_hat, xmin_hat)

    log("--- R3-D  leave-one-region-out ---")
    result["leave_one_region_out"] = leave_one_region_out(prim)

    log("--- R3-E  cross-estimator consistency ---")
    result["cross_estimator"] = cross_estimator(curves)

    # ---------------- cross-estimator comparison without a false correction
    tail_alpha = alpha_hat
    curve_alpha = result["cross_estimator"]["alpha_eff_from_curve"]
    result["cross_estimator_comparison"] = {
        "tail_alpha_observed": float(tail_alpha),
        "curve_alpha_observed_conditional": float(curve_alpha),
        "curve_point_consistent_with_alpha_gt2_branch":
            result["cross_estimator"]["point_consistent_with_alpha_gt2_branch"],
        "curve_ci_entirely_alpha_gt2":
            result["cross_estimator"]["ci_entirely_alpha_gt2"],
        "absolute_difference": float(abs(tail_alpha - curve_alpha)),
        "interpretation": "descriptive contrast between estimators with different assumptions; no common additive correction is applied",
        "correction_note": "Adding the same constant to both estimates cannot change their difference. Statistic-specific observer effects are evaluated separately in round3_bias_on_b.py as a model-conditional sensitivity analysis.",
    }
    log("R3-E observed estimator contrast: tail %.3f ; curve %.3f ; |delta| %.3f"
        % (tail_alpha, curve_alpha, abs(tail_alpha - curve_alpha)))

    result["finished_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result["elapsed_s"] = (datetime.now(timezone.utc) - t0).total_seconds()

    with open(PROC + "/summary_round3_computations.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)

    # Compact CSV for the figure: observed estimators only.
    rows = [{"estimator": "tail fit (observed)", "alpha": tail_alpha},
            {"estimator": "completeness curve (conditional)", "alpha": curve_alpha}]
    pd.DataFrame(rows).to_csv(PROC + "/alpha_two_estimators.csv", index=False)
    log("written summary_round3_computations.json (%.1f s)" % result["elapsed_s"])


if __name__ == "__main__":
    main()
