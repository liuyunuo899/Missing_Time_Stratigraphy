# -*- coding: utf-8 -*-
"""Deterministic verification of statistical and archive-model ground truth.

Run directly with Python; pytest is optional rather than required. Every random
operation has a local, recorded seed. The tests include the failure modes found
during review: the empirical-CDF left limit, declared-support MLEs, full-sample
GOF threshold reselection, grid-phase effects, and partial-layer preservation.
"""
import numpy as np

from stats_tools import (analyse_durations, completeness_at_resolution,
                         continuous_powerlaw_mle, covered_fraction,
                         extraction_gaps, fit_incompleteness_curve, fit_powerlaw,
                         incompleteness_shape, ks_distance,
                         powerlaw_gof_pvalue, predicted_incompleteness)
from simulation import preservation_metrics, run_case

SEED = 20261020


def sample_powerlaw(alpha, xmin, n, rng):
    u = rng.random(n)
    return xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))


def brute_force_completeness(intervals, tau, t0, t1, n_sub=40):
    """Independent fine-subcell implementation used as a numerical reference."""
    sub = tau / n_sub
    edge = np.arange(int((t1 - t0) / sub) + 1) * sub + t0
    hit = np.zeros(len(edge) - 1, dtype=bool)
    for a, b in intervals:
        lo = int(np.clip(np.floor((a - t0) / sub), 0, len(edge) - 2))
        hi = int(np.clip(np.ceil((b - t0) / sub), 1, len(edge) - 1))
        if hi > lo:
            hit[lo:hi] = True
    n_cells = int(np.floor((t1 - t0) / tau))
    ok = sum(hit[k * n_sub:(k + 1) * n_sub].any() for k in range(n_cells))
    return ok / n_cells if n_cells else np.nan


def main():
    print("=" * 78)
    print("TEST 1  declared-support MLE and the full two-sided KS statistic")
    mle = continuous_powerlaw_mle([2.0, 4.0, 8.0], xmin=1.0)
    expected = 1.0 + 3.0 / np.log(64.0)
    print("   MLE=%.9f expected=%.9f" % (mle, expected))
    assert abs(mle - expected) < 1e-12
    ks = ks_distance([10.0, 20.0, 30.0], alpha=2.0, xmin=1.0)
    print("   KS=%.6f expected=0.900000 (left-limit term is required)" % ks)
    assert abs(ks - 0.9) < 1e-12
    print("   PASS")

    print("\nTEST 2  power-law recovery on two known Pareto samples")
    for i, (true_a, xmin) in enumerate([(1.8, 0.1), (2.5, 0.05)]):
        rng = np.random.default_rng(SEED + i)
        x = sample_powerlaw(true_a, xmin, 30000, rng)
        fit = fit_powerlaw(x)
        print("   true=%.2f recovered=%.4f xmin=%.4f KS=%.4f"
              % (true_a, fit["alpha"], fit["xmin"], fit["ks"]))
        assert abs(fit["alpha"] - true_a) < 0.06
        assert fit["n_tail"] == int(np.sum(x >= fit["xmin"]))
    print("   PASS")

    print("\nTEST 3  alternative models are proper distributions on common support")
    rng = np.random.default_rng(SEED + 10)
    x = sample_powerlaw(2.2, 0.2, 5000, rng)
    res = analyse_durations(x)
    xmin = res.powerlaw["xmin"]
    print("   xmin=%.4f; lognormal truncated=%s; exponential loc=%.4f; GPD threshold=%.4f"
          % (xmin, res.lognormal["truncated"], res.exponential["loc"],
             res.generalised_pareto["xmin"]))
    assert res.lognormal["truncated"] and res.lognormal["xmin"] == xmin
    assert res.exponential["loc"] == xmin
    assert res.generalised_pareto["xmin"] == xmin
    for model in (res.lognormal, res.exponential, res.generalised_pareto):
        assert np.isfinite(model["loglike"])
        assert len(model["log_contrib"]) == res.powerlaw["n_tail"]
    assert np.isfinite(res.comparisons["powerlaw_vs_lognormal"]["LR"])
    print("   PASS")

    print("\nTEST 4  full-sample semiparametric GOF rejects a lognormal sample")
    rng = np.random.default_rng(SEED + 20)
    xl = np.exp(rng.normal(0.0, 1.6, 10000))
    gof_ln = powerlaw_gof_pvalue(xl, n_mc=199, seed=SEED + 20,
                                 return_details=True)
    print("   p=%.5f, valid MC=%d, floor=%.5f"
          % (gof_ln["p_value"], gof_ln["n_mc_valid"], gof_ln["mc_floor"]))
    assert gof_ln["p_value"] < 0.1
    assert "xmin and alpha re-estimated" in gof_ln["method"]
    print("   PASS")

    print("\nTEST 5  null calibration: Pareto retention and false-rejection rate")
    rng = np.random.default_rng(SEED + 30)
    xpl = sample_powerlaw(1.9, 0.1, 6000, rng)
    gof_pl = powerlaw_gof_pvalue(xpl, n_mc=199, seed=SEED + 30)
    print("   single Pareto sample p=%.5f" % gof_pl)
    assert gof_pl > 0.1
    null_p = []
    for k in range(24):
        rk = np.random.default_rng(SEED + 100 + k)
        xk = sample_powerlaw(2.1, 0.2, 800, rk)
        null_p.append(powerlaw_gof_pvalue(xk, n_mc=99, seed=SEED + 200 + k,
                                          n_grid=60, min_tail=25))
    rejection = float(np.mean(np.asarray(null_p) < 0.1))
    print("   24 null datasets: rejection rate at 0.10 = %.3f" % rejection)
    print("   p-values:", np.round(null_p, 3))
    assert rejection <= 0.25, "GOF pipeline is badly anti-conservative under the null"
    print("   PASS (small-sample calibration screen, not a formal size proof)")

    print("\nTEST 6  gap extraction and overlap handling")
    stack = [(0.0, 1.0), (1.4, 2.0), (3.5, 4.0)]
    gaps, _ = extraction_gaps(stack)
    assert np.allclose(gaps, [0.4, 1.5])
    gaps2, overlaps = extraction_gaps([(0.0, 1.5), (1.0, 2.0)])
    assert len(gaps2) == 0 and overlaps == 1
    print("   gaps=%s; overlapping stack yields no negative gap" % gaps)
    print("   PASS")

    print("\nTEST 7  exact cell counting and coverage against an independent method")
    intervals = [(0.0, 1.0), (1.4, 2.0), (3.5, 4.0)]
    worst = 0.0
    for tau in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
        fast = completeness_at_resolution(intervals, tau, 0.0, 4.0)
        slow = brute_force_completeness(intervals, tau, 0.0, 4.0)
        worst = max(worst, abs(fast - slow))
    cov = covered_fraction(intervals, 0.0, 4.0)
    print("   maximum cell-count difference=%.2e; coverage=%.6f" % (worst, cov))
    assert worst < 1e-9 and abs(cov - 0.525) < 1e-9
    print("   PASS")

    print("\nTEST 8  random-phase mean-excess relation versus fixed-grid phase")
    # Alternating D=1, G=3, tau=1. At phase zero three cells per cycle are empty;
    # the random-phase expectation is only (G-tau)/(D+G)=0.5.
    d, g, tau, period = 1.0, 3.0, 1.0, 4.0
    n_cycles = 50
    intervals_periodic = [(k * period, k * period + d)
                          for k in range(-1, n_cycles + 1)]
    fixed = 1.0 - completeness_at_resolution(intervals_periodic, tau, 0.0,
                                             n_cycles * period)
    rng = np.random.default_rng(SEED + 40)
    phases = rng.uniform(0.0, period, 60)
    missing = []
    for phi in phases:
        c = completeness_at_resolution(intervals_periodic, tau, -phi,
                                       n_cycles * period - phi)
        missing.append(1.0 - c)
    phase_mean = float(np.mean(missing))
    print("   fixed-grid M=%.3f; random-phase mean=%.3f; theoretical=%.3f"
          % (fixed, phase_mean, (g - tau) / period))
    assert abs(fixed - 0.75) < 1e-12
    assert abs(phase_mean - 0.5) < 0.015
    assert abs(fixed - phase_mean) > 0.2
    print("   PASS (the identity is not a phase-free finite-grid identity)")

    print("\nTEST 9  non-parametric prediction on a long renewal record")
    rng = np.random.default_rng(SEED + 50)
    gaps_draw = sample_powerlaw(2.5, 0.05, 180000, rng)
    gaps_draw = gaps_draw[gaps_draw < 60.0]
    dep = rng.exponential(0.5, gaps_draw.size)
    tcur, intervals, gaps_used, deps_used = 0.0, [], [], []
    for gap, duration in zip(gaps_draw, dep):
        intervals.append((tcur, tcur + duration))
        gaps_used.append(gap)
        deps_used.append(duration)
        tcur += duration + gap
        if tcur > 2.0e6:
            break
    t_end = intervals[-1][1]
    gaps_used = np.asarray(gaps_used[:-1])
    deps_used = np.asarray(deps_used)
    taus = np.array([0.2, 0.4, 0.8, 1.6, 3.2])
    measured = np.array([1.0 - completeness_at_resolution(intervals, tau, 0.0, t_end)
                         for tau in taus])
    predicted = np.array([predicted_incompleteness(gaps_used, deps_used, tau)
                          for tau in taus])
    rel = np.max(np.abs(measured - predicted) / np.maximum(predicted, 3e-3))
    abs_dev = np.max(np.abs(measured - predicted))
    print("   max relative error=%.4f; max absolute deviation=%.6f" % (rel, abs_dev))
    assert rel < 0.05 and abs_dev < 2e-3
    print("   PASS")

    print("\nTEST 10 capped survival integral, alpha=2 limit and curve fitting")
    from scipy.integrate import quad
    g0, cap = 0.05, 100.0
    for alpha in (1.5, 2.0, 2.5, 4.0):
        tau = 2.0
        denom = g0 ** (1.0 - alpha) - cap ** (1.0 - alpha)
        numerical = quad(lambda z: ((z ** (1.0 - alpha) - cap ** (1.0 - alpha)) / denom),
                         tau, cap, epsabs=1e-10)[0]
        analytic = incompleteness_shape(np.array([tau]), cap, alpha)[0] / denom
        print("   alpha=%.1f quadrature=%.9f analytic=%.9f" % (alpha, numerical, analytic))
        assert abs(numerical - analytic) < 1e-7 * max(1.0, abs(numerical))
        taus = np.geomspace(0.1, 20.0, 12)
        inc = 0.02 * incompleteness_shape(taus, cap, alpha)
        fit = fit_incompleteness_curve(taus, inc, cap)
        assert abs(fit["alpha"] - alpha) < 1e-3
    print("   PASS")

    print("\nTEST 11 random seeds reproduce the complete GOF result")
    rng = np.random.default_rng(SEED + 60)
    x = sample_powerlaw(2.0, 0.1, 1200, rng)
    p1 = powerlaw_gof_pvalue(x, n_mc=49, seed=777, n_grid=60)
    p2 = powerlaw_gof_pvalue(x, n_mc=49, seed=777, n_grid=60)
    print("   repeated p-values: %.6f, %.6f" % (p1, p2))
    assert p1 == p2
    print("   PASS")

    print("\nTEST 12 basement, partial-layer preservation and exact mass balance")
    res = run_case("multiscale", n=12000, dt=0.01, seed=11,
                   forcing_kwargs={"beta": 1.2, "n_scales": 6},
                   deposit_rate=25.0, erosion_rate=40.0)
    met = preservation_metrics(res)
    inc = np.diff(np.concatenate([[0.0], res.h]))
    assert np.all(res.h >= -1e-12)
    assert np.all((res.survival_fraction >= 0) & (res.survival_fraction <= 1))
    assert abs(met["mass_balance_residual_m"]) < 1e-8
    assert abs(met["layer_preservation_residual_m"]) < 1e-8
    assert np.allclose(np.cumsum(inc), res.h)
    print("   final=%.4f m; gross=%.4f m; layer residual=%.2e m; mass residual=%.2e m"
          % (met["final_preserved_m"], met["gross_deposited_m"],
             met["layer_preservation_residual_m"], met["mass_balance_residual_m"]))
    same = run_case("multiscale", n=12000, dt=0.01, seed=11,
                    forcing_kwargs={"beta": 1.2, "n_scales": 6},
                    deposit_rate=25.0, erosion_rate=40.0)
    assert np.array_equal(res.u, same.u) and np.array_equal(res.h, same.h)
    print("   PASS (including same-seed forward-model identity)")

    print("\nALL 12 STATISTICAL AND ARCHIVE-MODEL TEST GROUPS PASSED")


if __name__ == "__main__":
    main()
