# -*- coding: utf-8 -*-
"""Negative controls for Case Study I.

Two questions are asked and answered separately.

1. Does interval pinning drive the tail estimate?  Ages are dequantised by
   adding uniform jitter of +-delta to every unit boundary age, which destroys
   the exact coincidences with chronostratigraphic interval boundaries without
   changing the large-scale age model.  The tail fit is repeated on the
   jittered gaps.  If alpha is stable, quantisation is not the explanation.

2. Does interval pinning drive the completeness result?  The completeness
   exponent b is recomputed on the jittered columns with the identical
   completeness-at-resolution function.

Also reported: the gap-duration fit restricted to gaps whose bounding ages are
NOT pinned to an interval boundary, which is the most direct exclusion test.

Everything is seeded; the script writes
data/processed/summary_negative_controls.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stats_tools import (fit_powerlaw, powerlaw_gof_pvalue,                     # noqa: E402
                         extraction_gaps, completeness_at_resolution,
                         predicted_incompleteness)

ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed")
RAW = os.path.join(ROOT, "data", "raw")

SEED = 20261020
MIN_UNITS_PRIMARY = 15
TAU_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
MIN_TAU_MULTIPLE = 10.0
GOF_REPLICATES = 150


def primary_columns(units):
    counts = units.groupby("col_id").size()
    return counts[counts >= MIN_UNITS_PRIMARY].index


def gaps_for_column(sub, jitter=0.0, rng=None):
    """Coverage intervals for one column -> gap durations.

    If jitter > 0, every boundary age is displaced by U(-jitter, +jitter)
    before the intervals are formed, which breaks exact ties with
    chronostratigraphic boundaries.
    """
    t = sub["t_age"].to_numpy(dtype=float).copy()
    b = sub["b_age"].to_numpy(dtype=float).copy()
    if jitter > 0.0:
        t = t + rng.uniform(-jitter, jitter, t.size)
        b = b + rng.uniform(-jitter, jitter, b.size)
    # a unit must have positive duration after jittering
    ok = b > t
    t, b = t[ok], b[ok]
    order = np.argsort(t)
    iv = list(zip(t[order], b[order]))
    gaps, _ = extraction_gaps(iv)
    return np.asarray(gaps, dtype=float)


def gap_bounding_ages(sub):
    """Replicate the interval-merging rule of extraction_gaps, but also return
    the two ages that bound each gap: the bottom age that terminated the previous
    coverage run, and the top age of the unit that starts the next one."""
    t = sub["t_age"].to_numpy(dtype=float)
    b = sub["b_age"].to_numpy(dtype=float)
    ok = b > t
    t, b = t[ok], b[ok]
    order = np.argsort(t)
    t, b = t[order], b[order]
    gaps, ends, starts = [], [], []
    covered_end = None
    source_bottom = None
    for ti, bi in zip(t, b):
        if covered_end is None:
            covered_end, source_bottom = bi, bi
            continue
        if ti > covered_end:
            gaps.append(ti - covered_end)
            ends.append(source_bottom)
            starts.append(ti)
            covered_end, source_bottom = bi, bi
        else:
            covered_end = max(covered_end, bi)
            if bi >= covered_end:
                source_bottom = bi
    return np.array(gaps), np.array(ends), np.array(starts)


def gap_pinned_flags(sub, iv_ages, atol=1e-9):
    """Whether each gap has at least one bounding age sitting exactly on a
    chronostratigraphic interval boundary."""
    gaps, ends, starts = gap_bounding_ages(sub)
    if gaps.size == 0:
        return gaps, np.array([], dtype=bool)
    pin_e = np.isclose(ends[:, None], iv_ages[None, :], rtol=0, atol=atol).any(axis=1)
    pin_s = np.isclose(starts[:, None], iv_ages[None, :], rtol=0, atol=atol).any(axis=1)
    return gaps, pin_e | pin_s


def fit_and_report(x, label, replicates=GOF_REPLICATES):
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    if x.size < 100:
        return {"label": label, "n": int(x.size), "valid": False}
    fit = fit_powerlaw(x)
    if not fit:
        return {"label": label, "n": int(x.size), "valid": False}
    decades = float(np.log10(x.max() / fit["xmin"])) if fit["xmin"] > 0 else np.nan
    valid = bool(decades >= 1.0 and fit["n_tail"] >= 50)
    out = {"label": label, "n": int(x.size), "valid": bool(valid),
           "alpha": float(fit["alpha"]), "xmin_Myr": float(fit["xmin"]),
           "n_tail": int(fit["n_tail"]), "ks": float(fit["ks"]),
           "decades_tail": float(decades),
           "median_Myr": float(np.median(x))}
    if valid:
        p = powerlaw_gof_pvalue(x, n_mc=replicates, seed=SEED)
        out["gof_p"] = float(p)
    return out


def b_slope(taus, inc):
    ok = np.isfinite(inc) & (inc > 0)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(np.asarray(taus)[ok]),
                            np.log(np.asarray(inc)[ok]), 1)[0])


def main():
    units = pd.read_csv(os.path.join(PROC, "units_processed.csv"))
    units = units[units["is_phanerozoic"]] if "is_phanerozoic" in units else units
    ivs = pd.read_csv(os.path.join(RAW, "macrostrat_intervals_all.csv"))
    # a chronostratigraphic boundary is either the top or the bottom age of a
    # Macrostrat interval; both are used, as in data_processing.py
    cols = [c for c in ("t_age", "b_age") if c in ivs.columns]
    iv_ages = np.unique(np.concatenate(
        [ivs[c].dropna().to_numpy(dtype=float) for c in cols]))

    ids = primary_columns(units)
    print("primary columns: %d" % ids.size)

    results = {"seed": SEED, "jitter_Myr": [], "unpinned_subset": {},
               "completeness": {}}

    # ---------------------------------------------------------- baseline
    base_gaps = []
    unpinned_gaps = []
    for cid, sub in units[units["col_id"].isin(ids)].groupby("col_id"):
        g = gaps_for_column(sub)
        base_gaps.extend(g.tolist())
        g2, pinned = gap_pinned_flags(sub, iv_ages)
        unpinned_gaps.extend(g2[~pinned].tolist())
    base_gaps = np.array(base_gaps)
    unpinned_gaps = np.array(unpinned_gaps)
    np.savetxt(os.path.join(PROC, "gaps_unpinned_subset.csv"), unpinned_gaps,
               delimiter=",", header="hiatus_duration_myr", comments="")
    results["baseline"] = fit_and_report(base_gaps, "baseline_no_jitter")
    print("baseline: n=%d alpha=%.3f"
          % (base_gaps.size, results["baseline"].get("alpha", np.nan)))

    results["unpinned_subset"] = fit_and_report(
        unpinned_gaps, "gaps_with_no_pinned_bounding_age")
    results["unpinned_subset"]["n_excluded"] = int(base_gaps.size
                                                   - unpinned_gaps.size)
    print("unpinned-only: n=%d alpha=%.3f"
          % (unpinned_gaps.size, results["unpinned_subset"].get("alpha", np.nan)))

    # ---------------------------------------------------------- jitter sweep
    for jitter in (0.5, 2.0, 5.0):
        rng = np.random.default_rng(SEED + int(jitter * 100))
        gaps = []
        for cid, sub in units[units["col_id"].isin(ids)].groupby("col_id"):
            gaps.extend(gaps_for_column(sub, jitter=jitter, rng=rng).tolist())
        gaps = np.array(gaps)
        rep = fit_and_report(gaps, "jitter_%.1f_Myr" % jitter)
        results["jitter_Myr"].append(rep)
        print("jitter %.1f Myr: n=%d alpha=%.3f"
              % (jitter, gaps.size, rep.get("alpha", np.nan)))

    # ------------------------------------------- completeness under jitter
    for jitter in (0.0, 2.0):
        rng = np.random.default_rng(SEED + 777)
        rows = []
        for cid, sub in units[units["col_id"].isin(ids)].groupby("col_id"):
            t = sub["t_age"].to_numpy(dtype=float).copy()
            b = sub["b_age"].to_numpy(dtype=float).copy()
            if jitter > 0:
                t = t + rng.uniform(-jitter, jitter, t.size)
                b = b + rng.uniform(-jitter, jitter, b.size)
            ok = b > t
            t, b = t[ok], b[ok]
            if t.size < 3:
                continue
            o = np.argsort(t)
            t, b = t[o], b[o]
            start, end = float(t.min()), float(b.max())
            L = end - start
            iv = list(zip(t, b))
            gaps_i, _ = extraction_gaps(iv)
            dur = (b - t)
            comp, taus = [], []
            for tau in TAU_GRID:
                if L < MIN_TAU_MULTIPLE * tau or tau >= L:
                    continue
                comp.append(completeness_at_resolution(iv, tau, start, end))
                taus.append(tau)
            if len(comp) < 4:
                continue
            inc = 1.0 - np.asarray(comp)
            rows.append({"col_id": int(cid), "b": b_slope(taus, inc),
                         "c_fine": float(1.0 - inc[0]),
                         "c_coarse": float(1.0 - inc[-1])})
        df = pd.DataFrame(rows)
        key = "jitter_%.1f" % jitter
        results["completeness"][key] = {
            "n_columns": int(len(df)),
            "b_median": float(df["b"].median()),
            "b_iqr": [float(df["b"].quantile(0.25)), float(df["b"].quantile(0.75))],
            "completeness_fine_median": float(df["c_fine"].median()),
            "completeness_coarse_median": float(df["c_coarse"].median()),
            "delta_over_grid": float(df["c_coarse"].median() - df["c_fine"].median()),
        }
        print("completeness %s: n=%d b_median=%.3f C(%.2f)=%.3f C(%.0f)=%.3f"
              % (key, len(df), df["b"].median(), TAU_GRID[0],
                 df["c_fine"].median(), TAU_GRID[-1], df["c_coarse"].median()))

    out = os.path.join(PROC, "summary_negative_controls.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("written", out)


if __name__ == "__main__":
    main()
