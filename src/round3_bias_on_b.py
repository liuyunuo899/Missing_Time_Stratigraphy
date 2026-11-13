# -*- coding: utf-8 -*-
"""Model-conditional observer effects on two different statistics.

The generating non-deposition gaps and the surviving-column gaps need not have
the same tail fit. Likewise, the local exponent of a completeness curve can
change after removal. This script measures both effects in the synthetic
archive model. It does **not** estimate or correct the generating exponent of
the real geological system.

For each synthetic archive we compute

``b_gen``
    Local log-log slope of incompleteness from the true deposition intervals.
``b_obs``
    The same slope from the surviving column, using the observational
    any-part-represented convention.

The clean-record calibration in ``calibration_alpha.csv`` is simulator-specific.
We use it only for labelled sensitivity scenarios. Values outside its measured
range are returned as ``below_range`` or ``above_range`` with no numerical
alpha, rather than being silently clamped by ``numpy.interp``. The interquartile
scenarios are not confidence intervals and do not validate a bias correction.

Run:
    python3 src/round3_bias_on_b.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
SYN = ROOT + "/data/synthetic"
PROC = ROOT + "/data/processed"
sys.path.insert(0, ROOT + "/src")

from simulation import (completeness_from_survival, observed_gaps,  # noqa: E402
                        run_case, true_nondeposition_gaps)
from stats_tools import completeness_at_resolution, fit_powerlaw  # noqa: E402

TAUS = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
CASE_SEEDS = (1, 2)


def _b_slope(taus, inc):
    """Local log-log slope with degenerate values removed."""
    inc = np.asarray(inc, dtype=float)
    ok = np.isfinite(inc) & (inc > 1e-12)
    t = np.asarray(taus, dtype=float)[ok]
    i = inc[ok]
    if t.size < 4:
        return np.nan
    return float(np.polyfit(np.log(t), np.log(i), 1)[0])


def true_deposition_intervals(depositing, dt):
    """Contiguous (start, end) intervals in which deposition is active."""
    mask = np.asarray(depositing, dtype=bool)
    transitions = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return np.column_stack([starts * dt, stops * dt])


def screened_tail_fit(gaps):
    """Return a fit and apply the same scaling-range screen as Case Study II."""
    gaps = np.asarray(gaps, dtype=float)
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    if gaps.size < 40:
        return None, False
    fit = fit_powerlaw(gaps)
    if fit is None:
        return None, False
    valid = bool(fit["n_tail"] >= 50 and fit["decades_tail"] >= 1.0
                 and np.isfinite(fit["alpha"]))
    return fit, valid


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _quartiles(values):
    values = _finite(values)
    if values.size == 0:
        return None
    return [float(v) for v in np.quantile(values, [0.25, 0.5, 0.75])]


def prepare_calibration():
    """Read the simulator-specific calibration and diagnose monotonicity."""
    cal = pd.read_csv(SYN + "/calibration_alpha.csv")
    calb = (cal.groupby("alpha_true")["local_exponent_b_measured"]
            .mean().dropna().sort_index())
    grid_a = calb.index.to_numpy(float)
    grid_b = calb.to_numpy(float)
    differences = np.diff(grid_b)
    monotone = bool(np.all(differences <= 1e-12)
                    or np.all(differences >= -1e-12))
    order = np.argsort(grid_b)
    sorted_b = grid_b[order]
    sorted_a = grid_a[order]
    # Average duplicate abscissae so that interpolation has a strict grid.
    unique = pd.DataFrame({"b": sorted_b, "a": sorted_a}).groupby("b")["a"].mean()
    return {
        "alpha": grid_a,
        "b": grid_b,
        "b_sorted": unique.index.to_numpy(float),
        "alpha_sorted": unique.to_numpy(float),
        "monotone": monotone,
    }


def main():
    t0 = datetime.now(timezone.utc)
    cases = {
        "periodic": [dict(period=3.0, noise=0.15),
                     dict(period=20.0, noise=0.45)],
        "multiscale": [dict(beta=0.7, n_scales=4),
                       dict(beta=1.4, n_scales=8)],
        "ar1": [dict(rho=0.99, sigma=1.0), dict(rho=0.997, sigma=1.0)],
        "critical": [dict(rho=0.99, drift_period=20.0),
                     dict(rho=0.997, drift_period=80.0)],
    }
    rows = []
    n_a, dt_a = 30000, 0.02
    for mechanism, kwlist in cases.items():
        for kw in kwlist:
            for seed in CASE_SEEDS:
                for theta_dep in (-0.3, 0.3):
                    res = run_case(mechanism, n=n_a, dt=dt_a, seed=seed,
                                   forcing_kwargs=kw, theta_dep=theta_dep)
                    total = n_a * dt_a
                    true_iv = true_deposition_intervals(res.depositing, res.dt)
                    inc_gen = np.array([
                        1.0 - completeness_at_resolution(true_iv, tau, 0.0, total)
                        for tau in TAUS])
                    comp_obs = completeness_from_survival(res.survive, res.dt, TAUS)
                    inc_obs = 1.0 - comp_obs
                    b_gen = _b_slope(TAUS, inc_gen)
                    b_obs = _b_slope(TAUS, inc_obs)
                    tg = true_nondeposition_gaps(res.depositing, res.dt)
                    og = observed_gaps(res.survive, res.dt)
                    ft, valid_t = screened_tail_fit(tg)
                    fo, valid_o = screened_tail_fit(og)
                    alpha_gen = ft["alpha"] if valid_t else np.nan
                    alpha_obs = fo["alpha"] if valid_o else np.nan
                    rows.append({
                        "mechanism": mechanism, "seed": seed,
                        "theta_dep": theta_dep,
                        "setting": "_".join("%s%s" % kv for kv in kw.items()),
                        "b_gen": b_gen, "b_obs": b_obs,
                        "delta_b": b_obs - b_gen,
                        "alpha_gen": alpha_gen, "alpha_obs": alpha_obs,
                        "alpha_gen_raw": ft["alpha"] if ft else np.nan,
                        "alpha_obs_raw": fo["alpha"] if fo else np.nan,
                        "alpha_gen_valid": valid_t, "alpha_obs_valid": valid_o,
                        "delta_alpha": alpha_obs - alpha_gen,
                        "n_true_gaps": int(tg.size),
                        "n_observed_gaps": int(og.size),
                        "span_Myr": float(total),
                    })
                    print("%-11s %-28s th%+.1f s%d  b_gen=%+.3f b_obs=%+.3f  "
                          "alpha valid=%s/%s"
                          % (mechanism, rows[-1]["setting"], theta_dep, seed,
                             b_gen, b_obs, valid_t, valid_o), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(SYN + "/bias_on_b.csv", index=False)

    valid = df.dropna(subset=["b_gen", "b_obs"])
    delta_b = _finite(valid["delta_b"])
    delta_a = _finite(df.dropna(subset=["alpha_gen", "alpha_obs"])["delta_alpha"])
    if delta_b.size == 0:
        raise RuntimeError("no valid paired completeness-curve slopes")
    quart_b = _quartiles(delta_b)
    quart_a = _quartiles(delta_a)
    calibration = prepare_calibration()

    def invert_b_conditional(b):
        """Interpolate only inside a monotone, measured calibration range."""
        if not np.isfinite(b):
            return {"status": "not_finite", "alpha_conditional": None}
        if not calibration["monotone"]:
            return {"status": "non_monotonic_calibration",
                    "alpha_conditional": None}
        grid_b = calibration["b_sorted"]
        grid_a = calibration["alpha_sorted"]
        tol = 1e-12
        if b < grid_b.min() - tol:
            return {"status": "below_range", "alpha_conditional": None}
        if b > grid_b.max() + tol:
            return {"status": "above_range", "alpha_conditional": None}
        return {"status": "within_range",
                "alpha_conditional": float(np.interp(b, grid_b, grid_a))}

    with open(PROC + "/summary_real_stats.json", encoding="utf-8") as fh:
        real = json.load(fh)
    with open(PROC + "/summary_round3_computations.json", encoding="utf-8") as fh:
        r3 = json.load(fh)
    real_curves = {
        "pooled_local_slope": float(real["local_exponent"]["mean"]),
        "column_mean_slope": float(r3["cross_estimator"]["b_mean"]),
    }
    scenario_deltas = {"no_shift": 0.0, "q25_observer_effect": quart_b[0],
                       "median_observer_effect": quart_b[1],
                       "q75_observer_effect": quart_b[2]}
    curve_scenarios = {}
    for name, b_observed in real_curves.items():
        scenarios = []
        for scenario, delta in scenario_deltas.items():
            shifted_b = float(b_observed - delta)
            record = {"scenario": scenario, "delta_b_applied": float(delta),
                      "b_observed": float(b_observed), "shifted_b": shifted_b}
            record.update(invert_b_conditional(shifted_b))
            scenarios.append(record)
        curve_scenarios[name] = scenarios

    tail_scenarios = []
    if quart_a is not None:
        alpha_tail_obs = float(r3["alpha_uncertainty"]["alpha_hat"])
        for scenario, delta in zip(
                ["q25_observer_effect", "median_observer_effect",
                 "q75_observer_effect"], quart_a):
            tail_scenarios.append({
                "scenario": scenario, "delta_alpha_applied": float(delta),
                "alpha_observed": alpha_tail_obs,
                "shifted_alpha_conditional": float(alpha_tail_obs - delta),
            })

    out = {
        "generated_utc": t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seeds": {"cases": list(CASE_SEEDS)},
        "n_runs_total": int(len(df)),
        "n_runs_valid_b": int(delta_b.size),
        "n_runs_valid_alpha_pair": int(delta_a.size),
        "analysis_status": "model_conditional_sensitivity_not_a_bias_correction",
        "note": "b_obs uses the surviving column and the any-part-represented "
                "convention; b_gen uses contiguous true deposition intervals. "
                "Tail exponents are withheld unless n_tail >= 50 and the fitted "
                "range spans at least one decade.",
        "delta_b_median": float(np.median(delta_b)),
        "delta_b_iqr": [float(quart_b[0]), float(quart_b[2])],
        "delta_b_min": float(delta_b.min()),
        "delta_b_max": float(delta_b.max()),
        "fraction_flattened": float(np.mean(delta_b > 0)),
        "delta_alpha_median": float(np.median(delta_a)) if delta_a.size else None,
        "delta_alpha_iqr": ([float(quart_a[0]), float(quart_a[2])]
                            if quart_a is not None else None),
        "calibration_grid": {
            "%.2f" % a: float(b)
            for a, b in zip(calibration["alpha"], calibration["b"])},
        "calibration_diagnostics": {
            "monotone_in_alpha_order": calibration["monotone"],
            "b_range": [float(calibration["b_sorted"].min()),
                        float(calibration["b_sorted"].max())],
            "alpha_range": [float(calibration["alpha"].min()),
                            float(calibration["alpha"].max())],
            "out_of_range_rule": "return status and null; never clamp to an endpoint",
        },
        "real_data_sensitivity": {
            "warning": "Scenario shifts are not uncertainty intervals, independent "
                       "validation, or estimates of the real generating process.",
            "curve_scenarios": curve_scenarios,
            "tail_scenarios": tail_scenarios,
        },
    }
    print()
    print("delta_b: median %+.3f, IQR [%+.3f, %+.3f], positive in %.0f%% "
          "of %d valid runs"
          % (out["delta_b_median"], out["delta_b_iqr"][0],
             out["delta_b_iqr"][1], 100 * out["fraction_flattened"],
             delta_b.size))
    if delta_a.size:
        print("delta_alpha (screened pairs): median %+.3f over %d runs"
              % (out["delta_alpha_median"], delta_a.size))
    for name, scenarios in curve_scenarios.items():
        print(name + ": " + "; ".join(
            "%s %s" % (s["scenario"], s["status"] if s["alpha_conditional"] is None
                       else "alpha=%.3f" % s["alpha_conditional"])
            for s in scenarios))

    out["elapsed_s"] = (datetime.now(timezone.utc) - t0).total_seconds()
    with open(SYN + "/summary_round3_bias_on_b.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    print("written summary_round3_bias_on_b.json (%.1f s)" % out["elapsed_s"])


if __name__ == "__main__":
    main()
