# -*- coding: utf-8 -*-
"""Calibration of the rate-span (Sadler) exponent against the hiatus tail exponent.

For a stationary alternating process with constant deposition rate and a finite
mean cycle, random-phase window placement makes expected coverage independent of
window length. An uncapped heavy-tailed gap model with alpha <= 2 is sometimes
assigned the asymptotic scaling R(s) ~ s^{-(2-alpha)/(alpha-1)}, but that regime
is not the same as the finite, capped process simulated here.

This script therefore reports both the uncapped asymptotic reference and the
capped stationary prediction (zero slope). A rate-span slope is not treated as
an independent estimator of alpha: it shares the age model with the other
summaries and also responds to within-column and among-column heterogeneity.

Run:
    python paper_project/src/calibration_sadler.py
"""
import json
import sys

import numpy as np
import pandas as pd

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
SYN = ROOT + "/data/synthetic"
PROC = ROOT + "/data/processed"
sys.path.insert(0, ROOT + "/src")

SEED = 20261020


def sample_pareto(alpha, xmin, n, rng):
    """Continuous Pareto with pdf exponent alpha (survival exponent alpha-1)."""
    u = rng.random(n)
    return xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))


def build_record(alpha, n_cycles, rng, dep_rate=1.0, dt=0.01, cap=400.0):
    gaps = sample_pareto(alpha, 0.05, n_cycles, rng)
    gaps = np.clip(gaps, 2 * dt, cap)
    deps = rng.exponential(0.4, n_cycles)
    starts, t = [], 0.0
    for d, g in zip(deps, gaps):
        starts.append((t, t + d))
        t += d + g
    return np.array(starts), t


def covered_in_window(starts, window_len, total_len, n_windows, rng):
    """Mean fraction of a randomly placed window covered by deposition."""
    starts = starts[(starts[:, 1] <= total_len)]
    lo = rng.uniform(0.0, total_len - window_len, n_windows)
    frac = np.empty(n_windows)
    for k, a in enumerate(lo):
        b = a + window_len
        sel = starts[(starts[:, 1] > a) & (starts[:, 0] < b)]
        cov = np.minimum(sel[:, 1], b) - np.maximum(sel[:, 0], a)
        frac[k] = np.clip(cov.sum(), 0.0, window_len) / window_len
    return float(frac.mean())


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    spans = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    for alpha in [1.4, 1.6, 1.8, 1.9, 2.1, 2.5, 3.0]:
        for rep in range(3):
            starts, total = build_record(alpha, 120000, rng)
            rates = np.array([covered_in_window(starts, s, total, 400, rng)
                              for s in spans])
            slope = np.polyfit(np.log(spans), np.log(rates), 1)[0]
            uncapped = (-(2.0 - alpha) / (alpha - 1.0)) if alpha < 2 else 0.0
            capped_stationary = 0.0
            rows.append({"alpha_true": alpha, "replicate": rep,
                         "slope_measured": float(slope),
                         "slope_uncapped_asymptotic_reference": float(uncapped),
                         "slope_capped_stationary_prediction": capped_stationary,
                         "gap_cap_Myr": 400.0,
                         "rate_ratio_0.5_to_64": float(rates[0] / rates[-1]),
                         "record_length_Myr": float(total)})
            print("alpha=%.2f rep=%d  measured=%+.3f  capped prediction=%+.3f  "
                  "uncapped reference=%+.3f"
                  % (alpha, rep, slope, capped_stationary, uncapped))
    df = pd.DataFrame(rows)
    df.to_csv(SYN + "/calibration_sadler.csv", index=False)
    agg = df.groupby("alpha_true").agg(
        slope_measured=("slope_measured", "mean"),
        slope_uncapped_asymptotic_reference=(
            "slope_uncapped_asymptotic_reference", "mean"),
        slope_capped_stationary_prediction=(
            "slope_capped_stationary_prediction", "mean")).round(3).to_dict(orient="index")
    out = {"generated_utc": __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED, "by_alpha": agg,
        "note": "measured slope is d log(rate)/d log(window length); gaps are "
                "capped at 400 Myr, so the finite-mean stationary prediction is "
                "zero and the uncapped asymptote is only a contrasting reference"}
    print()
    print(df.groupby("alpha_true")[["slope_measured",
                                    "slope_capped_stationary_prediction",
                                    "slope_uncapped_asymptotic_reference"]].mean()
          .round(3).to_string())
    # Compare ranges only. Do not silently clamp an out-of-range real slope to
    # an endpoint or label a simulator-specific inversion as independent truth.
    with open(PROC + "/summary_sadler_decomposition.json", encoding="utf-8") as fh:
        real = json.load(fh)
    real_slope = float(real["pooled_slope"])
    measured_range = [float(df["slope_measured"].min()),
                      float(df["slope_measured"].max())]
    out["real_slope_context"] = {
        "pooled_slope": real_slope,
        "calibration_measured_slope_range": measured_range,
        "within_measured_range": bool(measured_range[0] <= real_slope
                                      <= measured_range[1]),
        "note": "descriptive comparison only; no alpha point estimate is returned "
                "because the calibration is generator-specific and the real slope "
                "also contains within- and among-column components",
    }
    print("\nobserved pooled slope %.3f; inside calibration range=%s; no alpha inversion"
          % (real_slope, out["real_slope_context"]["within_measured_range"]))
    with open(SYN + "/summary_sadler_calibration.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
