# -*- coding: utf-8 -*-
"""Case Study I: statistical analysis of the processed Macrostrat dataset.

Every quantity written to ``summary_real_stats.json`` is computed here and
quoted from there in the manuscript.

Run:
    python paper_project/src/analysis_real.py
"""
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
PROC = ROOT + "/data/processed"
sys.path.insert(0, ROOT + "/src")

from stats_tools import (analyse_durations, fit_incompleteness_curve,  # noqa: E402
                         powerlaw_gof_pvalue)

SEED = 20261020
N_BOOT = 300
GOF_MC = 300


def analyse_group(name, values, out, n_boot=N_BOOT):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < 40:
        out[name] = {"n": int(vals.size), "status": "insufficient sample"}
        print("%-42s n=%d  INSUFFICIENT" % (name, vals.size))
        return out[name]
    res = analyse_durations(vals, n_boot=n_boot, seed=SEED)
    gof = powerlaw_gof_pvalue(vals, n_mc=GOF_MC, seed=SEED,
                              return_details=True)
    entry = {
        "n": int(vals.size),
        "decades": float(np.log10(vals.max() / vals.min())),
        "min_Myr": float(vals.min()), "median_Myr": float(np.median(vals)),
        "max_Myr": float(vals.max()),
        "alpha": res.powerlaw["alpha"], "xmin_Myr": res.powerlaw["xmin"],
        "n_tail": res.powerlaw["n_tail"], "ks": res.powerlaw["ks"],
        "decades_tail": res.powerlaw["decades_tail"],
        "gof_p": gof["p_value"],
        "gof_mc_valid": gof["n_mc_valid"],
        "gof_mc_floor": gof["mc_floor"],
        "gof_method": gof["method"],
        "alpha_ci95": list(res.alpha_ci) if res.alpha_ci else None,
        "lognormal_xmin": res.lognormal["xmin"] if res.lognormal else None,
        "exponential_loc": res.exponential["loc"] if res.exponential else None,
        "loglognormal": res.lognormal["mu"] if res.lognormal else None,
        "sigma_lognormal": res.lognormal["sigma"] if res.lognormal else None,
        "genpareto_shape": (res.generalised_pareto["shape_c"]
                            if res.generalised_pareto else None),
        "genpareto_tail_index": (res.generalised_pareto["tail_index"]
                                 if res.generalised_pareto else None),
        "vuong_powerlaw_vs_lognormal": res.comparisons.get("powerlaw_vs_lognormal"),
        "vuong_powerlaw_vs_exponential": res.comparisons.get("powerlaw_vs_exponential"),
    }
    out[name] = entry
    print("%-42s n=%5d alpha=%.2f xmin=%6.3f KS=%.3f GOF p=%.3f CI=%s"
          % (name, entry["n"], entry["alpha"], entry["xmin_Myr"], entry["ks"],
             entry["gof_p"],
             ("[%.2f, %.2f]" % tuple(entry["alpha_ci95"])) if entry["alpha_ci95"]
             else "n/a"))
    return entry


def main():
    gaps = pd.read_csv(PROC + "/hiatus_gaps.csv")
    comp = pd.read_csv(PROC + "/completeness_curves.csv")
    units = pd.read_csv(PROC + "/units_processed.csv", low_memory=False)
    rates = pd.read_csv(PROC + "/units_with_rates.csv", low_memory=False)
    with open(PROC + "/summary_real.json", encoding="utf-8") as fh:
        base = json.load(fh)

    out = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "seeds": {"bootstrap": SEED, "gof": SEED},
           "replicates": {"bootstrap": N_BOOT, "gof_monte_carlo": GOF_MC},
           "input_rows": {"gaps": int(len(gaps)), "completeness": int(len(comp)),
                          "units": int(len(units))}}

    # ------------------------------------------------------------ group filter
    # primary subset: as in data_processing, columns holding at least
    # MIN_UNITS_PRIMARY units (note: this is a unit count, not a gap count)
    per_col = gaps.groupby("col_id").agg(n_units=("n_units_column", "first"),
                                         col_group=("col_group", "first"))
    primary_ids = per_col.index[per_col["n_units"] >= 15]
    prim = gaps[gaps["col_id"].isin(primary_ids)].copy()
    prim = prim[prim["hiatus_duration_myr"] > 0]
    print("== hiatus duration distributions ==")
    print("primary subset: %d columns, %d positive gaps"
          % (primary_ids.size, len(prim)))
    dist = {}
    analyse_group("primary_all", prim["hiatus_duration_myr"], dist)
    for thr in [0.05, 0.25, 1.0, 2.0]:
        sel = prim.loc[prim["hiatus_duration_myr"] > thr, "hiatus_duration_myr"]
        analyse_group("primary_gt_%.2f_Myr" % thr, sel, dist)
    # regional subsets: the four largest Macrostrat column groups
    for grp in per_col.loc[primary_ids, "col_group"].value_counts().head(4).index:
        sel = prim.loc[prim["col_group"] == grp, "hiatus_duration_myr"]
        analyse_group("region_%s" % grp.replace(" ", "_").replace(",", ""), sel, dist)
    out["duration_distributions"] = dist

    # ----------------------------------------------------- completeness curves
    print()
    print("== completeness at cell size tau ==")
    agg = comp.groupby("tau_Myr").agg(
        n=("completeness", "size"),
        mean_completeness=("completeness", "mean"),
        median_completeness=("completeness", "median"),
        mean_predicted=("predicted_completeness", "mean"),
        mean_residual=("residual", "mean"),
        mean_coverage=("covered_fraction", "mean"),
    ).reset_index()
    agg["mean_incompleteness"] = 1.0 - agg["mean_completeness"]
    print(agg.round(4).to_string(index=False))
    out["completeness_pooled"] = agg.round(6).to_dict(orient="records")

    print()
    print("== fine-resolution limit check (count-based -> coverage-based) ==")
    finest = agg.iloc[0]
    out["fine_resolution_limit"] = {
        "tau_Myr": float(finest["tau_Myr"]),
        "mean_completeness": float(finest["mean_completeness"]),
        "mean_coverage_fraction": float(finest["mean_coverage"]),
        "absolute_difference": float(abs(finest["mean_completeness"]
                                         - finest["mean_coverage"])),
    }
    print(out["fine_resolution_limit"])

    # per-column fit of the capped analytic form to the incompleteness curve
    print()
    print("== alpha from completeness curves (capped analytic fit) ==")
    fits = []
    for col_id, g in comp.groupby("col_id"):
        g = g.sort_values("tau_Myr")
        g = g[(g["tau_Myr"] >= 0.05) & (g["completeness"] < 0.999)]
        inc = 1.0 - g["completeness"].values
        if len(g) < 4 or np.any(inc <= 0):
            continue
        t_cap = float(g["record_length_Myr"].iloc[0])
        try:
            f = fit_incompleteness_curve(g["tau_Myr"].values, inc, t_cap=t_cap)
        except Exception:  # noqa: BLE001 - keep going, record the failure
            continue
        f["col_id"] = int(col_id)
        f["n_tau"] = int(len(g))
        fits.append(f)
    fits_df = pd.DataFrame(fits)
    out["completeness_alpha"] = {
        "n_columns_fitted": int(len(fits_df)),
        "median_alpha": float(fits_df["alpha"].median()) if len(fits_df) else None,
        "alpha_q25": float(fits_df["alpha"].quantile(0.25)) if len(fits_df) else None,
        "alpha_q75": float(fits_df["alpha"].quantile(0.75)) if len(fits_df) else None,
        "median_rmse_log": float(fits_df["rmse_log"].median()) if len(fits_df) else None,
        "n_at_lower_bound": int(fits_df["at_lower_bound"].sum()) if len(fits_df) else 0,
        "n_at_upper_bound": int(fits_df["at_upper_bound"].sum()) if len(fits_df) else 0,
        "interpretation": "exploratory capped-shape fit; boundary optima and residual misfit preclude a stand-alone tail-exponent estimate",
    }
    print(out["completeness_alpha"])
    if len(fits_df):
        fits_df.to_csv(PROC + "/completeness_alpha_fits.csv", index=False)

    # pooled fit
    if len(agg) >= 5:
        pooled = agg[(agg["tau_Myr"] >= 0.05) & (agg["mean_incompleteness"] > 0)]
        f_pool = fit_incompleteness_curve(pooled["tau_Myr"].values,
                                          pooled["mean_incompleteness"].values,
                                          t_cap=float(comp["record_length_Myr"].median()))
        out["completeness_alpha_pooled"] = {
            "alpha": f_pool["alpha"], "rmse_log": f_pool["rmse_log"],
            "alpha_bounds": f_pool["alpha_bounds"],
            "at_lower_bound": f_pool["at_lower_bound"],
            "at_upper_bound": f_pool["at_upper_bound"],
            "interpretation": "descriptive fit only; not an unbiased estimate of the generating tail exponent"}
        print("pooled curve alpha = %.3f (rmse %.3f)"
              % (f_pool["alpha"], f_pool["rmse_log"]))

    # model-free local exponent b = d ln(incompleteness) / d ln(tau)
    tau_v = agg["tau_Myr"].values.astype(float)
    inc_v = agg["mean_incompleteness"].values.astype(float)
    ok = np.isfinite(inc_v) & (inc_v > 0)
    b_local = np.diff(np.log(inc_v[ok])) / np.diff(np.log(tau_v[ok]))
    out["local_exponent"] = {
        "definition": "b = d ln(1-completeness) / d ln(tau) between adjacent tau",
        "values": [float(x) for x in b_local],
        "mean": float(np.mean(b_local)),
        "min": float(np.min(b_local)), "max": float(np.max(b_local)),
        "implied_alpha_if_above_two": float(2.0 - np.mean(b_local)),
    }
    print("local exponent b mean %.3f range [%.3f, %.3f] -> implied alpha %.2f"
          % (out["local_exponent"]["mean"], out["local_exponent"]["min"],
             out["local_exponent"]["max"],
             out["local_exponent"]["implied_alpha_if_above_two"]))

    # ------------------------------------------------- Sadler rate-span scaling
    print()
    print("== Sadler-type accumulation rate vs measured time span ==")
    r = rates.dropna(subset=["rate_m_per_myr"])
    r = r[(r["duration_myr"] > 0) & (r["rate_m_per_myr"] > 0)]
    sadler = {}

    def fit_slope(frame, label):
        if len(frame) < 200:
            sadler[label] = {"n": int(len(frame)), "status": "too few"}
            return sadler[label]
        edges = np.logspace(np.log10(frame["duration_myr"].quantile(0.01)),
                            np.log10(frame["duration_myr"].quantile(0.99)), 12)
        frame = frame.assign(bin=np.digitize(frame["duration_myr"], edges))
        grp = frame.groupby("bin").agg(
            n=("rate_m_per_myr", "size"), med_dur=("duration_myr", "median"),
            med_rate=("rate_m_per_myr", "median")).reset_index()
        grp = grp[grp["n"] >= 30]
        if len(grp) < 5:
            sadler[label] = {"n": int(len(frame)), "status": "too few bins"}
            return sadler[label]
        slope, _ = np.polyfit(np.log10(grp["med_dur"]), np.log10(grp["med_rate"]), 1)
        sadler[label] = {"n_units": int(len(frame)), "n_bins": int(len(grp)),
                         "slope": float(slope), "median_rate": float(frame["rate_m_per_myr"].median())}
        print("%-34s n=%6d bins=%2d slope=%+.3f" % (label, len(frame), len(grp), slope))
        return sadler[label]

    fit_slope(r, "all_reported_thickness")
    phan = r[r["b_age"] <= 541.0]
    fit_slope(phan, "phanerozoic_only")
    fit_slope(r[r["max_thick"] > 0], "positive_max_thick")
    fit_slope(r[r["duration_myr"] > 0.05], "duration_gt_0.05Myr")
    fit_slope(r[(r["duration_myr"] > 0.05) & (r["max_thick"] > 0)],
              "duration_gt_0.05Myr_and_thickness")
    out["sadler_slopes"] = sadler

    # bootstrap the slope over columns so that spatial clustering is respected
    rng = np.random.default_rng(SEED)
    ids = r["col_id"].unique()
    slopes = []
    for _ in range(60):
        pick = rng.choice(ids, size=len(ids), replace=True)
        frame = pd.concat([r[r["col_id"] == i] for i in pick])
        if len(frame) < 500:
            continue
        try:
            s = fit_slope_quiet(frame)
        except Exception:  # noqa: BLE001
            continue
        if s is not None:
            slopes.append(s)
    if slopes:
        out["sadler_slope_column_bootstrap"] = {
            "n_replicates": len(slopes),
            "median": float(np.median(slopes)),
            "ci95": [float(np.quantile(slopes, 0.025)),
                     float(np.quantile(slopes, 0.975))]}
        print("column bootstrap over the all-units slope: median %.3f CI %s"
              % (out["sadler_slope_column_bootstrap"]["median"],
                 out["sadler_slope_column_bootstrap"]["ci95"]))

    out["provenance"] = base.get("parameters", {})
    out["n_units_total"] = base.get("n_units_total")
    out["n_units_phanerozoic"] = base.get("n_units_phanerozoic")
    out["fraction_boundaries_on_interval_boundary"] = base.get(
        "fraction_unit_boundaries_exactly_on_an_interval_boundary")

    with open(PROC + "/summary_real_stats.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print()
    print("wrote", PROC + "/summary_real_stats.json")


def fit_slope_quiet(frame):
    """Slope fit helper used inside bootstrap replicates (no printing)."""
    if len(frame) < 300:
        return None
    edges = np.logspace(np.log10(frame["duration_myr"].quantile(0.01)),
                        np.log10(frame["duration_myr"].quantile(0.99)), 12)
    frame = frame.assign(bin=np.digitize(frame["duration_myr"], edges))
    grp = frame.groupby("bin").agg(
        n=("rate_m_per_myr", "size"), med_dur=("duration_myr", "median"),
        med_rate=("rate_m_per_myr", "median")).reset_index()
    grp = grp[grp["n"] >= 30]
    if len(grp) < 5:
        return None
    slope, _ = np.polyfit(np.log10(grp["med_dur"]), np.log10(grp["med_rate"]), 1)
    return float(slope)


if __name__ == "__main__":
    main()
