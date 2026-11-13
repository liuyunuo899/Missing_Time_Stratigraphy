# -*- coding: utf-8 -*-
"""Case Study I: build the analysis dataset from the raw Macrostrat export.

Processing decisions are recorded verbatim in the returned summary JSON so that
every number quoted in the manuscript can be traced to a reproducible step.

Run (from the repository root):
    python paper_project/src/data_processing.py
Outputs:
    data/processed/units_processed.csv
    data/processed/hiatus_gaps.csv
    data/processed/completeness_curves.csv
    data/processed/summary_real.json
"""
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
RAW = ROOT + "/data/raw"
PROC = ROOT + "/data/processed"
sys.path.insert(0, ROOT + "/src")

from stats_tools import (completeness_at_resolution, covered_fraction,  # noqa: E402
                         extraction_gaps, predicted_incompleteness)

MIN_UNITS_PRIMARY = 15          # columns retained for the primary analysis
PHANEROZOIC_MAX_AGE = 541.0     # Ma, base Cambrian
TAU_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
MIN_TAU_MULTIPLE = 10.0         # require record length >= 10 * tau


def load_raw():
    units = pd.read_csv(RAW + "/macrostrat_unit_boundaries_all.csv", low_memory=False)
    cols = pd.read_csv(RAW + "/macrostrat_columns_all.csv", low_memory=False)
    intervals = pd.read_csv(RAW + "/macrostrat_intervals_all.csv", low_memory=False)
    return units, cols, intervals


def quality_screen(units, log):
    """Return the screened unit table and a dictionary of rejection counts."""
    n0 = len(units)
    u = units.copy()
    for c in ["t_age", "b_age", "max_thick", "min_thick", "col_id", "project_id"]:
        u[c] = pd.to_numeric(u[c], errors="coerce")
    u = u.drop_duplicates(subset=["unit_id"])
    log["n_after_dedup_units"] = int(len(u))

    bad_order = (u["b_age"] <= u["t_age"]) | u["b_age"].isna() | u["t_age"].isna()
    u = u[~bad_order]
    log["n_rejected_age_order_or_missing"] = int(bad_order.sum())

    u["duration_myr"] = u["b_age"] - u["t_age"]
    u["is_phanerozoic"] = u["b_age"] <= PHANEROZOIC_MAX_AGE
    # max_thick == 0 is Macrostrat's placeholder for "thickness not reported",
    # so it must not be treated as a measured zero thickness.
    u["thickness_m"] = u["max_thick"].replace(0.0, np.nan)
    u.loc[u["thickness_m"] < 0, "thickness_m"] = np.nan
    log["n_units_total"] = int(n0)
    log["n_units_retained"] = int(len(u))
    log["n_units_thickness_reported"] = int(u["thickness_m"].notna().sum())
    log["n_units_phanerozoic"] = int(u["is_phanerozoic"].sum())
    return u


def main():
    log = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "source": "Macrostrat v2 bulk CSV export, CC-BY 4.0",
           "parameters": {"MIN_UNITS_PRIMARY": MIN_UNITS_PRIMARY,
                          "PHANEROZOIC_MAX_AGE_Ma": PHANEROZOIC_MAX_AGE,
                          "TAU_GRID_Myr": TAU_GRID,
                          "MIN_TAU_MULTIPLE": MIN_TAU_MULTIPLE}}
    units, cols, intervals = load_raw()
    u = quality_screen(units, log)

    ph = u[u["is_phanerozoic"]].copy()

    # ---------------------------------------------------------- gap catalogue
    gaps_rows = []
    for col_id, grp in ph.groupby("col_id"):
        if len(grp) < 2:
            continue
        g = grp.sort_values("t_age")
        iv = list(zip(g["t_age"].values, g["b_age"].values))
        gaps, overlaps = extraction_gaps(iv)
        start, end = float(g["t_age"].min()), float(g["b_age"].max())
        cinfo = cols[cols["col_id"] == col_id]
        for duration in gaps:
            gaps_rows.append({
                "col_id": int(col_id),
                "col_name": str(cinfo["col_name"].iloc[0]) if len(cinfo) else "",
                "col_group": str(cinfo["col_group"].iloc[0]) if len(cinfo) else "",
                "lat": float(cinfo["lat"].iloc[0]) if len(cinfo) else np.nan,
                "lng": float(cinfo["lng"].iloc[0]) if len(cinfo) else np.nan,
                "col_area_km2": float(cinfo["col_area"].iloc[0]) if len(cinfo) else np.nan,
                "n_units_column": int(len(g)),
                "record_length_myr": end - start,
                "hiatus_duration_myr": float(duration),
                "n_overlaps": int(overlaps),
            })
    gaps_df = pd.DataFrame(gaps_rows)
    log["n_gaps_all_columns_phanerozoic"] = int(len(gaps_df))
    log["n_columns_with_gaps"] = int(gaps_df["col_id"].nunique()) if len(gaps_df) else 0
    # overlaps are a per-column statistic; sum them once per column, not per row
    if len(gaps_df):
        per_col = gaps_df.groupby("col_id")["n_overlaps"].first()
        log["overlaps_total"] = int(per_col.sum())
        log["overlaps_per_column_median"] = float(per_col.median())
        log["col_level_note"] = (
            "a Macrostrat column aggregates several sections, so unit age ranges "
            "may overlap laterally; gaps are computed on the union of coverage, "
            "which is why the sum of unit durations exceeds the record length")

    # primary subset: well-populated columns
    counts = ph.groupby("col_id").size()
    primary_ids = counts[counts >= MIN_UNITS_PRIMARY].index
    primary = gaps_df[gaps_df["col_id"].isin(primary_ids)].copy()
    log["n_columns_primary"] = int(len(primary_ids))
    log["n_gaps_primary"] = int(len(primary))
    log["primary_positive_gaps"] = int((primary["hiatus_duration_myr"] > 0).sum()) \
        if len(primary) else 0

    if len(primary):
        h = primary["hiatus_duration_myr"].values
        h = h[h > 0]
        log["hiatus_primary_summary_Myr"] = {
            "n": int(h.size),
            "min": float(np.min(h)), "median": float(np.median(h)),
            "mean": float(np.mean(h)),
            "p95": float(np.quantile(h, 0.95)), "max": float(np.max(h)),
            "decades_spanned": float(np.log10(np.max(h) / np.min(h))),
        }
        # filtering below resolution of the compilation is a real bias source
        for thr in [1e-9, 0.001, 0.01, 0.05, 0.1, 0.25]:
            log.setdefault("n_above_threshold", {})["gt_%.3g" % thr] = int((h > thr).sum())

    # ---------------------------------------------- completeness per column
    comp_rows = []
    for col_id in primary_ids:
        grp = ph[ph["col_id"] == col_id].sort_values("t_age")
        iv = list(zip(grp["t_age"].values, grp["b_age"].values))
        start, end = float(grp["t_age"].min()), float(grp["b_age"].max())
        L = end - start
        gaps_i, _ = extraction_gaps(iv)
        dur = (grp["b_age"] - grp["t_age"]).values
        for tau in TAU_GRID:
            if L < MIN_TAU_MULTIPLE * tau or tau >= L:
                continue
            c = completeness_at_resolution(iv, tau, start, end)
            pred_records = predicted_incompleteness(gaps_i, dur, tau,
                                                    total_length=L)
            pred_renewal = predicted_incompleteness(gaps_i, dur, tau)
            comp_rows.append({"col_id": int(col_id), "tau_Myr": tau,
                              "record_length_Myr": L,
                              "completeness": c,
                              "predicted_completeness": 1.0 - pred_records,
                              "residual": c - (1.0 - pred_records),
                              "predicted_completeness_renewal_denominator":
                                  1.0 - pred_renewal,
                              "residual_renewal": c - (1.0 - pred_renewal),
                              "covered_fraction": covered_fraction(iv, start, end)})
    comp = pd.DataFrame(comp_rows)
    log["n_completeness_records"] = int(len(comp))
    if len(comp):
        agg = comp.groupby("tau_Myr").agg(
            n=("completeness", "size"),
            mean_completeness=("completeness", "mean"),
            median_completeness=("completeness", "median"),
            mean_predicted=("predicted_completeness", "mean"),
        ).reset_index()
        agg["mean_residual"] = agg["mean_completeness"] - agg["mean_predicted"]
        comp_agg = agg
        log["pooled_completeness"] = agg.round(4).to_dict(orient="records")
    else:  # pragma: no cover
        comp_agg = pd.DataFrame()

    # --------------------------------------------- Sadler-type rate vs span
    rated = ph.dropna(subset=["thickness_m"]).copy()
    rated = rated[(rated["duration_myr"] > 0) & (rated["thickness_m"] > 0)]
    rated["rate_m_per_myr"] = rated["thickness_m"] / rated["duration_myr"]
    log["n_units_for_rate_analysis"] = int(len(rated))
    if len(rated) > 100:
        edges = np.logspace(np.log10(max(1e-4, rated["duration_myr"].quantile(0.005))),
                            np.log10(rated["duration_myr"].quantile(0.995)), 14)
        rated["bin"] = np.digitize(rated["duration_myr"], edges)
        grp = rated.groupby("bin").agg(
            n=("rate_m_per_myr", "size"),
            med_duration=("duration_myr", "median"),
            med_rate=("rate_m_per_myr", "median"),
            p25=("rate_m_per_myr", lambda s: np.quantile(s, 0.25)),
            p75=("rate_m_per_myr", lambda s: np.quantile(s, 0.75)),
        ).reset_index()
        grp = grp[grp["n"] >= 25]
        slope, intercept = np.polyfit(np.log10(grp["med_duration"]),
                                      np.log10(grp["med_rate"]), 1)
        log["sadler_rate_span_slope"] = float(slope)
        log["sadler_bins"] = grp.round(5).to_dict(orient="records")
        rate_table = grp
    else:  # pragma: no cover
        rate_table = pd.DataFrame()

    # ------------------------------------------------------------- age pinning
    iv_ages = np.unique(np.concatenate([
        pd.to_numeric(intervals[c], errors="coerce").dropna().to_numpy(dtype=float)
        for c in ("t_age", "b_age") if c in intervals.columns
    ]))
    all_ages = np.concatenate([u["t_age"].values, u["b_age"].values])
    all_ages = all_ages[np.isfinite(all_ages)]
    on_boundary = np.isclose(all_ages[:, None], iv_ages[None, :], rtol=0, atol=1e-9).any(axis=1)
    log["fraction_unit_boundaries_exactly_on_an_interval_boundary"] = float(on_boundary.mean())
    log["n_interval_boundary_ages"] = int(len(iv_ages))

    # --------------------------------------------------------------- outputs
    import os
    os.makedirs(PROC, exist_ok=True)
    u.to_csv(PROC + "/units_processed.csv", index=False)
    gaps_df.to_csv(PROC + "/hiatus_gaps.csv", index=False)
    comp.to_csv(PROC + "/completeness_curves.csv", index=False)
    if len(comp_agg):
        comp_agg.to_csv(PROC + "/completeness_pooled.csv", index=False)
    if len(rate_table):
        rate_table.to_csv(PROC + "/rate_span_bins.csv", index=False)
    rated.to_csv(PROC + "/units_with_rates.csv", index=False)
    with open(PROC + "/summary_real.json", "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=1)

    print(json.dumps({k: v for k, v in log.items()
                      if k not in ("sadler_bins", "pooled_completeness")}, indent=1))
    if len(comp_agg):
        print("\npooled completeness:")
        print(comp_agg.round(4).to_string(index=False))
    if len(rate_table):
        print("\nSadler-type slope dlog10(rate)/dlog10(span) = %.3f"
              % log["sadler_rate_span_slope"])


if __name__ == "__main__":
    main()
