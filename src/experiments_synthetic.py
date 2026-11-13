# -*- coding: utf-8 -*-
"""Case Study II: reproducible experiments on the synthetic archive model.

Sections
  A  mechanism comparison (equifinality) over four forcing families
  B  calibration: how does the completeness exponent behave either side of
     alpha = 2 in this generator?  The calibration is generator-specific and is
     not a universal correction for real data.
  C  variational inversion: identifiability of deposition and erosion histories
  D  preservation budget: truth versus what an observer could infer

Run:
    python paper_project/src/experiments_synthetic.py
Outputs:
    data/synthetic/mechanism_sweep.csv
    data/synthetic/calibration_alpha.csv
    data/synthetic/inversion_ensemble.csv
    data/synthetic/preservation.csv
    data/synthetic/summary_synthetic.json
"""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
SYN = ROOT + "/data/synthetic"
PROC = ROOT + "/data/processed"
sys.path.insert(0, ROOT + "/src")

from inversion import (InverseProblem, ensemble, noisy_data_ensemble,  # noqa: E402
                       recover_metrics, solve_fista, solve_lbfgsb)
from simulation import (completeness_from_survival, observed_gaps,  # noqa: E402
                        preservation_metrics, run_case,
                        true_nondeposition_gaps)
from stats_tools import (analyse_durations, fit_powerlaw,  # noqa: E402
                         powerlaw_gof_pvalue, predicted_incompleteness)

SEED = 20261020
TAUS = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
N_STEPS = 20000
DT = 0.01          # Myr -> 200 Myr of simulated history
GOF_MC = 150


def _b_slope(taus, inc):
    """Local log-log slope with degenerate values removed."""
    inc = np.asarray(inc, dtype=float)
    ok = np.isfinite(inc) & (inc > 1e-12)
    taus = np.asarray(taus, dtype=float)[ok]
    inc = inc[ok]
    if taus.size < 4:
        return np.nan
    return float(np.polyfit(np.log(taus), np.log(inc), 1)[0])


def summarise_durations(x, label, store):
    """Fit the tail, but only report alpha when the scaling range is credible.

    Following the caution of Clauset et al. (2009), a fit is marked valid only if
    the fitted range spans at least one decade (xmax/xmin >= 10) and the tail
    holds at least 50 observations. Without this screen, near-degenerate samples
    return numerically absurd exponents that would otherwise be quoted.
    """
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    row = {"label": label, "n": int(x.size)}
    if x.size < 40:
        row.update({"alpha": np.nan, "xmin": np.nan, "n_tail": 0, "ks": np.nan,
                    "gof_p": np.nan, "decades_tail": np.nan, "valid": False})
        store.append(row)
        return row
    res = analyse_durations(x, n_boot=0, seed=SEED)
    if res.powerlaw is None:  # pragma: no cover
        row.update({"valid": False})
        store.append(row)
        return row
    gof = powerlaw_gof_pvalue(x, n_mc=GOF_MC, seed=SEED)
    decades = float(np.log10(x.max() / res.powerlaw["xmin"]))
    valid = bool(decades >= 1.0 and res.powerlaw["n_tail"] >= 50)
    row.update({"alpha": res.powerlaw["alpha"], "xmin": res.powerlaw["xmin"],
                "n_tail": res.powerlaw["n_tail"], "ks": res.powerlaw["ks"],
                "gof_p": gof, "decades_tail": decades, "valid": valid})
    if not valid:
        row["alpha"] = np.nan      # withheld rather than reported as absurd
    store.append(row)
    return row


# ---------------------------------------------------------------------- part A
def part_a_mechanisms(rows_dist, rows_record):
    """Mechanism sweep over long records, so that each realisation contains
    enough threshold crossings for a credible tail fit."""
    print("== Part A: mechanism sweep ==")
    n_a, dt_a = 30000, 0.02          # 600 Myr of simulated history
    cases = {
        # thresholds are expressed in standard deviations of the standardised
        # forcing, so all families are treated alike
        "periodic": [dict(period=p, noise=n)
                     for p in (3.0, 20.0) for n in (0.15, 0.45)],
        "multiscale": [dict(beta=b, n_scales=k)
                       for b in (0.7, 1.4) for k in (4, 8)],
        "ar1": [dict(rho=r, sigma=1.0) for r in (0.99, 0.997)],
        "critical": [dict(rho=r, drift_period=p)
                     for r in (0.99, 0.997) for p in (20.0, 80.0)],
    }
    for mechanism, kwlist in cases.items():
        for kw in kwlist:
            for seed in (1, 2, 3):
                for theta_dep in (-0.3, 0.3):
                    res = run_case(mechanism, n=n_a, dt=dt_a, seed=seed,
                                   forcing_kwargs=kw, theta_dep=theta_dep)
                    true_gaps = true_nondeposition_gaps(res.depositing, res.dt)
                    obs_gaps = observed_gaps(res.survive, res.dt)
                    comp = completeness_from_survival(res.survive, res.dt, TAUS)
                    metrics = preservation_metrics(res)
                    tag = "%s_%s_th%.1f_s%d" % (
                        mechanism,
                        "_".join("%s%s" % (k, v) for k, v in kw.items()),
                        theta_dep, seed)
                    r_true = summarise_durations(true_gaps, tag + "_true", rows_dist)
                    r_obs = summarise_durations(obs_gaps, tag + "_observed", rows_dist)
                    inc = 1.0 - comp
                    rows_record.append({
                        "tag": tag, "mechanism": mechanism, "seed": seed,
                        "span_Myr": float(n_a * dt_a),
                        "theta_dep": theta_dep,
                        "alpha_true": r_true.get("alpha", np.nan),
                        "alpha_true_valid": bool(r_true.get("valid", False)),
                        "alpha_observed_valid": bool(r_obs.get("valid", False)),
                        "alpha_observed_decades": r_obs.get("decades_tail", np.nan),
                        "duty_cycle": float(res.depositing.mean()),
                        "erosion_fraction": float(res.eroding.mean()),
                        "n_true_gaps": int(true_gaps.size),
                        "n_observed_gaps": int(obs_gaps.size),
                        "median_true_gap_Myr": float(np.median(true_gaps))
                        if true_gaps.size else np.nan,
                        "median_observed_gap_Myr": float(np.median(obs_gaps))
                        if obs_gaps.size else np.nan,
                        "alpha_observed": r_obs.get("alpha", np.nan),
                        "completeness_tau0.05": float(comp[0]),
                        "completeness_tau5": float(comp[-1]),
                        "local_exponent_b": _b_slope(TAUS, inc),
                        "P_gross": metrics["P_gross"],
                        "gross_deposited_m": metrics["gross_deposited_m"],
                        "final_preserved_m": metrics["final_preserved_m"],
                        "time_preserved_fraction": metrics["time_preserved_fraction_true"],
                        "time_represented_any_part_fraction": metrics["time_represented_any_part_fraction"],
                        "layer_preservation_residual_m": metrics["layer_preservation_residual_m"],
                        "mass_balance_residual_m": metrics["mass_balance_residual_m"],
                    })
    print("cases recorded:", len(rows_record))


# ---------------------------------------------------------------------- part B
def part_b_calibration(rows_calib):
    """Composite generated with prescribed alpha; measure the completeness exponent."""
    print("== Part B: calibration either side of alpha = 2 ==")
    rng = np.random.default_rng(SEED)
    for alpha_true in [1.5, 1.75, 1.9, 2.1, 2.5, 3.0]:
        for rep in range(3):
            gaps = []
            total = 0.0
            while len(gaps) < 20000:
                u = rng.random(40000)
                g = 0.05 * (1.0 - u) ** (-1.0 / (alpha_true - 1.0))
                g = g[(g > DT * 2) & (g < 400.0)]
                gaps.extend(g.tolist())
            gaps = np.array(gaps[:20000])
            deps = rng.exponential(0.4, gaps.size)
            iv, tcur = [], 0.0
            for gd, dd in zip(gaps, deps):
                iv.append((tcur, tcur + dd))
                tcur += dd + gd
            taus_local = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
            from stats_tools import completeness_at_resolution
            inc = np.array([1.0 - completeness_at_resolution(iv, tau, 0.0, tcur)
                            for tau in taus_local])
            inc2 = np.array([predicted_incompleteness(gaps, deps, tau,
                                                      total_length=tcur)
                             for tau in taus_local])
            b = _b_slope(taus_local, inc)
            b2 = _b_slope(taus_local, inc2)
            fit = fit_powerlaw(gaps)
            rows_calib.append({
                "alpha_true": alpha_true, "replicate": rep,
                "alpha_recovered": fit["alpha"] if fit else np.nan,
                "local_exponent_b_measured": float(b),
                "local_exponent_b_predicted": float(b2),
                "two_minus_alpha": float(2.0 - alpha_true),
                "completeness_tau0.1": float(1.0 - inc[0]),
                "completeness_tau5": float(1.0 - inc[-1]),
                "record_length_Myr": float(tcur),
            })
            print("   alpha_true=%.2f  b_measured=%+.3f  b_predicted=%+.3f  "
                  "2-alpha=%+.2f" % (alpha_true, b, b2, 2 - alpha_true))


# ---------------------------------------------------------------------- part C
def part_c_inversion(inv_rows, summary):
    print("== Part C: variational inversion and identifiability ==")
    res = run_case("multiscale", n=N_STEPS, dt=DT, seed=11,
                   forcing_kwargs=dict(beta=1.2, n_scales=6), theta_dep=0.0,
                   deposit_rate=25.0, erosion_rate=40.0)
    # Use the actual basement-constrained increments as ground truth. Reintegrating
    # the prescribed rates without the basement would define a different history.
    inc_true = np.diff(np.concatenate([[0.0], res.h]))
    d_true = np.maximum(inc_true, 0.0) / res.dt
    e_true = np.maximum(-inc_true, 0.0) / res.dt
    h_true = res.h.copy()
    n = N_STEPS
    rng = np.random.default_rng(SEED + 5)
    # observation design: 12 dated horizons inside the preserved column
    idx = np.sort(rng.choice(np.arange(1, n), size=12, replace=False))
    sigma_z = max(5.0, 0.02 * float(np.abs(h_true[idx]).max() + 1.0))
    obs_z = h_true[idx] + rng.normal(0.0, sigma_z, idx.size)
    ip = InverseProblem(n=n, dt=res.dt, obs_idx=idx, obs_z=obs_z,
                        obs_sigma=np.full(idx.size, sigma_z),
                        truth_d=d_true, truth_e=e_true)
    summary["inversion_observation_design"] = {
        "n_horizons": int(idx.size), "sigma_z_m": float(sigma_z),
        "grid_steps": int(n), "dt_Myr": float(res.dt),
        "true_gross_deposition_m": float(d_true.sum() * res.dt),
        "true_erosion_m": float(e_true.sum() * res.dt),
        "true_P_gross": float(1.0 - (e_true.sum() * res.dt) / (d_true.sum() * res.dt)),
    }
    print("   true gross deposition %.0f m, true P_gross %.3f"
          % (summary["inversion_observation_design"]["true_gross_deposition_m"],
             summary["inversion_observation_design"]["true_P_gross"]))

    # cross-check of the two solvers at a representative regularisation
    sol_f = solve_fista(ip, 1e-4, 1e-4, max_iter=6000)
    sol_s = solve_lbfgsb(ip, 1e-4, 1e-4)
    rel = abs(sol_f["objective"] - sol_s["objective"]) / max(1e-12,
                                                             abs(sol_s["objective"]))
    summary["solver_crosscheck"] = {
        "fista_objective": sol_f["objective"], "lbfgsb_objective": sol_s["objective"],
        "independent_solver": "scipy L-BFGS-B with non-negativity bounds",
        "relative_difference": float(rel),
        "fista_stationarity": sol_f["stationarity"],
        "fista_max_residual_sigma": sol_f["residual"],
        "acceptable_tolerance": 1e-3,
        "passed": bool(rel < 1e-3),
    }
    print("   solver cross-check: rel.obj.diff=%.2e stationarity=%.2e -> %s"
          % (rel, sol_f["stationarity"],
             "PASS" if summary["solver_crosscheck"]["passed"] else "FAIL"))

    lam_grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    ens = ensemble(ip, lam_grid, lam_grid, max_iter=4000)
    ratios = np.array([e["metrics"]["gross_deposition_ratio"] for e in ens])
    pres = np.array([e["metrics"]["preservation_recovered"] for e in ens])
    summary["lambda_ensemble"] = {
        "n_solutions": len(ens),
        "gross_ratio_min": float(np.nanmin(ratios)),
        "gross_ratio_max": float(np.nanmax(ratios)),
        "gross_ratio_median": float(np.nanmedian(ratios)),
        "preservation_min": float(np.nanmin(pres)),
        "preservation_max": float(np.nanmax(pres)),
        "max_constraint_residual_sigma": float(max(e["residual"] for e in ens)),
        "max_stationarity": float(max(e["stationarity"] for e in ens)),
    }
    print("   lambda ensemble: gross-deposition ratio spans %.3f-%.3f (%d solutions)"
          % (summary["lambda_ensemble"]["gross_ratio_min"],
             summary["lambda_ensemble"]["gross_ratio_max"], len(ens)))
    for e in ens:
        inv_rows.append({
            "experiment": "lambda_sweep", "lam_d": e["lam_d"], "lam_e": e["lam_e"],
            "objective": e["objective"], "stationarity": e["stationarity"],
            "residual_sigma": e["residual"],
            "gross_ratio": e["metrics"]["gross_deposition_ratio"],
            "preservation": e["metrics"]["preservation_recovered"],
            "d_rmse": e["metrics"]["d_rmse"], "e_rmse": e["metrics"]["e_rmse"],
            "d_corr": e["metrics"]["d_corr"],
        })
    np.save(SYN + "/inversion_true_history.npy",
            np.vstack([d_true, e_true, h_true]))

    # effect of extra chronological information
    rng2 = np.random.default_rng(SEED + 99)
    for extra in (0, 12, 36):
        st = noisy_data_ensemble(ip, sigma_frac=0.03, n_rep=12, rng=rng2,
                                 lam_d=1e-3, lam_e=1e-3, extra_rows=extra,
                                 max_iter=4000)
        st["experiment"] = "extra_horizons"
        inv_rows.append({k: st.get(k) for k in
                         ["experiment", "extra_horizons", "sigma_fraction",
                          "gross_ratio_median", "gross_ratio_iqr",
                          "gross_ratio_range", "preservation_median"]})
        summary.setdefault("extra_horizon_test", []).append(
            {"extra_horizons": extra,
             "gross_ratio_iqr_width": float(st["gross_ratio_iqr"][1]
                                            - st["gross_ratio_iqr"][0]),
             "gross_ratio_median": st["gross_ratio_median"],
             "gross_ratio_range": st["gross_ratio_range"]})
        print("   extra horizons=%2d -> gross-ratio IQR width %.3f (median %.3f)"
              % (extra, st["gross_ratio_iqr"][1] - st["gross_ratio_iqr"][0],
                 st["gross_ratio_median"]))


# ---------------------------------------------------------------------- part D
def part_d_preservation(rows_pres):
    print("== Part D: preservation budget truth vs observer inference ==")
    for mechanism, kw in [("multiscale", dict(beta=1.2, n_scales=6)),
                          ("periodic", dict(period=8.0, noise=0.3)),
                          ("ar1", dict(rho=0.999)),
                          ("critical", dict(rho=0.999, drift_period=40.0))]:
        for erosion_rate in (0.0, 10.0, 25.0, 50.0, 100.0):
            for seed in (1, 2, 3):
                res = run_case(mechanism, n=N_STEPS, dt=DT, seed=seed,
                               forcing_kwargs=kw, theta_dep=0.0,
                               deposit_rate=25.0, erosion_rate=erosion_rate)
                m = preservation_metrics(res)
                obs_g = observed_gaps(res.survive, res.dt)
                represented = float(res.survive.sum()) * res.dt
                apparent_rate = (m["final_preserved_m"] / represented
                                 if represented > 0 else np.nan)
                # naive observer estimate: assume the preserved rate also applied
                # during every observed gap
                missing_time = float(np.sum(obs_g))
                naive_gross = m["final_preserved_m"] + apparent_rate * missing_time
                naive_P = (m["final_preserved_m"] / naive_gross
                           if naive_gross > 0 else np.nan)
                rows_pres.append({
                    "mechanism": mechanism, "erosion_rate_m_per_Myr": erosion_rate,
                    "seed": seed, "P_gross_true": m["P_gross"],
                    "P_naive_observer": naive_P,
                    "abs_error": (naive_P - m["P_gross"]),
                    "gross_deposited_m": m["gross_deposited_m"],
                    "eroded_m": m["eroded_m"],
                    "final_preserved_m": m["final_preserved_m"],
                    "represented_time_Myr": represented,
                    "observed_missing_time_Myr": missing_time,
                    "time_preserved_fraction": m["time_preserved_fraction_true"],
                    "mass_balance_residual_m": m["mass_balance_residual_m"],
                })


def main():
    os.makedirs(SYN, exist_ok=True)
    rows_dist, rows_record, rows_calib, inv_rows, rows_pres = [], [], [], [], []
    part_a_mechanisms(rows_dist, rows_record)
    part_b_calibration(rows_calib)
    part_c_inversion(inv_rows, summary := {"generated_utc":
                                           datetime.now(timezone.utc).strftime(
                                               "%Y-%m-%dT%H:%M:%SZ"),
                                           "seed_master": SEED,
                                           "n_steps": N_STEPS, "dt_Myr": DT})
    part_d_preservation(rows_pres)

    pd.DataFrame(rows_dist).to_csv(SYN + "/duration_fits.csv", index=False)
    pd.DataFrame(rows_record).to_csv(SYN + "/mechanism_sweep.csv", index=False)
    pd.DataFrame(rows_calib).to_csv(SYN + "/calibration_alpha.csv", index=False)
    pd.DataFrame(inv_rows).to_csv(SYN + "/inversion_ensemble.csv", index=False)
    pd.DataFrame(rows_pres).to_csv(SYN + "/preservation.csv", index=False)

    mech = pd.DataFrame(rows_record)
    valid = pd.DataFrame(rows_dist)
    valid = valid[valid.get("valid", False)]
    summary["part_a_equifinality"] = {
        "n_cases": int(len(mech)),
        "n_valid_tail_fits": int(len(valid)) if len(valid) else 0,
        "n_withheld_degenerate_fits": int(len(rows_dist) - len(valid)),
        "alpha_observed_range_valid": [float(np.nanmin(mech["alpha_observed"])),
                                       float(np.nanmax(mech["alpha_observed"]))],
        "P_gross_range": [float(mech["P_gross"].min()), float(mech["P_gross"].max())],
        "max_mass_balance_residual_m": float(mech["mass_balance_residual_m"].abs().max()),
        "max_layer_preservation_residual_m": float(mech["layer_preservation_residual_m"].abs().max()),
        "alpha_observed_by_mechanism": mech.groupby("mechanism")["alpha_observed"]
        .agg(["count", "median", "min", "max"]).round(3).to_dict(),
    }
    print()
    valid_range = [v for v in mech["alpha_observed"].values if np.isfinite(v)]
    if valid_range:
        print("Part A: alpha_observed (valid fits only) spans %.2f - %.2f over %d cases"
              % (min(valid_range), max(valid_range), len(valid_range)))
    print(mech.groupby("mechanism")["alpha_observed"].describe()[["count", "50%", "min", "max"]].round(2).to_string())
    calib = pd.DataFrame(rows_calib)
    summary["part_b_calibration"] = calib.groupby("alpha_true").agg(
        b_measured=("local_exponent_b_measured", "mean"),
        b_predicted=("local_exponent_b_predicted", "mean"),
        two_minus_alpha=("two_minus_alpha", "mean")).round(3).to_dict(orient="index")
    pres = pd.DataFrame(rows_pres)
    summary["part_d_preservation"] = {
        "mean_abs_error_naive": float(pres["abs_error"].abs().mean()),
        "max_abs_error_naive": float(pres["abs_error"].abs().max()),
        "by_erosion_rate": pres.groupby("erosion_rate_m_per_Myr").agg(
            mean_true=("P_gross_true", "mean"),
            mean_naive=("P_naive_observer", "mean"),
            mean_abs_error=("abs_error", lambda s: float(s.abs().mean()))
        ).round(3).to_dict(orient="index"),
        "max_mass_balance_residual_m": float(pres["mass_balance_residual_m"].abs().max()),
    }
    print()
    print(pres.groupby("erosion_rate_m_per_Myr").agg(
        mean_true=("P_gross_true", "mean"), mean_naive=("P_naive_observer", "mean"),
        mean_abs_error=("abs_error", lambda s: float(s.abs().mean()))).round(3).to_string())

    if os.path.exists(SYN + "/calibration_alpha.csv"):
        calib_prev = pd.read_csv(SYN + "/calibration_alpha.csv")
        curve = calib_prev.groupby("alpha_true").agg(
            b=("local_exponent_b_measured", "mean")).reset_index()
        curve = curve.dropna().sort_values("alpha_true")
        if len(curve) >= 4:
            # This is a simulator-specific mapping, not an independent estimator
            # of the real generating exponent. Conditional inversion and explicit
            # range diagnostics are handled in round3_bias_on_b.py.
            summary["calibration_mapping"] = {
                "alpha_grid": curve["alpha_true"].tolist(),
                "b_grid": curve["b"].tolist(),
                "b_range": [float(curve["b"].min()), float(curve["b"].max())],
                "note": "descriptive finite-record calibration for this generator; "
                        "not a universal relation and not a validated correction"}
            real_b = None
            try:
                with open(PROC + "/summary_real_stats.json", encoding="utf-8") as fh:
                    real_b = json.load(fh)["local_exponent"]["mean"]
            except Exception:  # noqa: BLE001
                pass
            if real_b is not None:
                in_range = bool(curve["b"].min() <= real_b <= curve["b"].max())
                summary["calibration_mapping"]["observed_b_real"] = float(real_b)
                summary["calibration_mapping"]["observed_b_within_range"] = in_range
                print("observed real b=%.3f; inside simulator calibration range=%s"
                      % (real_b, in_range))

    with open(SYN + "/summary_synthetic.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print()
    print("wrote", SYN + "/summary_synthetic.json")


if __name__ == "__main__":
    main()
