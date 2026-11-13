# -*- coding: utf-8 -*-
"""Decompose the measured rate-span (Sadler-type) decline into a within-column
and an among-column component.

Motivation: pooled and within-column rate–span slopes answer different
questions. The capped stationary calibration supplies one contrasting model,
not a general proof that within-column slopes must vanish. This script measures
the observational components directly so that pooling, thickness–duration
coupling, and age assignment can be discussed without assigning a mechanism
from the pooled slope alone.

Run:
    python paper_project/src/sadler_decomposition.py
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

SEED = 20261020
MIN_UNITS = 20


def within_column_slopes(df, min_units=MIN_UNITS):
    """Slope of log10(rate) against log10(duration) fitted inside each column."""
    out = []
    for col_id, g in df.groupby("col_id"):
        if len(g) < min_units:
            continue
        x = np.log10(g["duration_myr"].values)
        y = np.log10(g["rate_m_per_myr"].values)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_units or np.std(x[ok]) == 0:
            continue
        slope = np.polyfit(x[ok], y[ok], 1)[0]
        out.append({"col_id": int(col_id), "n_units": int(ok.sum()),
                    "slope": float(slope),
                    "span_decades": float(np.ptp(x[ok]))})
    return pd.DataFrame(out)


def among_column_slope(df, n_bins=12, min_per_bin=30):
    """Slope obtained by pooling all units (the usual compilation procedure)."""
    edges = np.logspace(np.log10(df["duration_myr"].quantile(0.01)),
                        np.log10(df["duration_myr"].quantile(0.99)), n_bins)
    b = np.digitize(df["duration_myr"].values, edges)
    rows = []
    for k in np.unique(b):
        sel = df.iloc[np.flatnonzero(b == k)]
        if len(sel) < min_per_bin:
            continue
        rows.append({"n": len(sel), "med_dur": float(sel["duration_myr"].median()),
                     "med_rate": float(sel["rate_m_per_myr"].median())})
    t = pd.DataFrame(rows)
    if len(t) < 5:
        return np.nan, t
    return float(np.polyfit(np.log10(t["med_dur"]), np.log10(t["med_rate"]), 1)[0]), t


def main():
    df = pd.read_csv(PROC + "/units_with_rates.csv")
    df = df.dropna(subset=["rate_m_per_myr", "duration_myr"])
    df = df[(df["duration_myr"] > 0) & (df["rate_m_per_myr"] > 0)
            & (df["b_age"] <= 541.0)]
    print("units with rate and duration (Phanerozoic):", len(df))

    pooled, bins = among_column_slope(df)
    print("pooled (among-column) slope: %.3f" % pooled)

    wc = within_column_slopes(df)
    print("columns analysed within: %d" % len(wc))
    rng = np.random.default_rng(SEED)
    reps = []
    for _ in range(400):
        samp = rng.choice(wc["slope"].values, size=len(wc), replace=True)
        reps.append(np.median(samp))
    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_units": int(len(df)),
        "n_columns_within": int(len(wc)),
        "pooled_slope": float(pooled),
        "within_column_slope_median": float(np.median(wc["slope"])),
        "within_column_slope_mean": float(np.mean(wc["slope"])),
        "within_column_slope_iqr": [float(np.quantile(wc["slope"], 0.25)),
                                    float(np.quantile(wc["slope"], 0.75))],
        "within_column_slope_median_ci95": [float(np.quantile(reps, 0.025)),
                                            float(np.quantile(reps, 0.975))],
        "fraction_columns_with_negative_slope": float((wc["slope"] < 0).mean()),
        "median_span_decades_per_column": float(wc["span_decades"].median()),
    }
    print()
    print(json.dumps(summary, indent=1))
    wc.to_csv(PROC + "/sadler_within_column_slopes.csv", index=False)
    bins.to_csv(PROC + "/sadler_pooled_bins.csv", index=False)
    with open(PROC + "/summary_sadler_decomposition.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("wrote", PROC + "/summary_sadler_decomposition.json")


if __name__ == "__main__":
    main()
