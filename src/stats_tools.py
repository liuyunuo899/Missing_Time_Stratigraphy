# -*- coding: utf-8 -*-
"""Distribution fitting utilities shared by Case Study I (real data) and
Case Study II (synthetic experiments).

The same estimator is applied to observations and to simulated output so that any
difference in the recovered exponents cannot be attributed to the estimator.

Methods follow Clauset, Shalizi & Newman (2009), DOI 10.1137/070710111, with the
discrete estimator replaced by the continuous version because the data (durations
in Myr) are effectively continuous; rounding is negligible relative to the fitted
scaling range. Model comparison uses the normalised log-likelihood ratio test
(Vuong 1989, DOI 10.2307/1912557) for non-nested hypotheses as recommended in the
same reference.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats


# --------------------------------------------------------------------- helpers
def _tail(x, xmin):
    x = np.asarray(x, dtype=float)
    return x[x >= xmin]


def continuous_powerlaw_mle(x_tail, xmin=None):
    """Continuous Pareto MLE on the declared support ``x >= xmin``.

    ``xmin`` is explicit throughout the fitting pipeline. Inferring it from the
    minimum of a resampled tail would change the support and bias the exponent.
    """
    x_tail = np.asarray(x_tail, dtype=float)
    x_tail = x_tail[np.isfinite(x_tail) & (x_tail > 0)]
    if x_tail.size == 0:
        return float("nan")
    xmin = float(np.min(x_tail) if xmin is None else xmin)
    tail = x_tail[x_tail >= xmin]
    denom = float(np.sum(np.log(tail / xmin)))
    if tail.size == 0 or xmin <= 0 or denom <= 0:
        return float("inf")
    return float(1.0 + tail.size / denom)


def powerlaw_loglike(x_tail, alpha, xmin):
    """Continuous power-law log-likelihood including the normalisation constant."""
    return np.sum(np.log((alpha - 1.0) / xmin * (x_tail / xmin) ** (-alpha)))


# ------------------------------------------------------------ candidate models
def fit_powerlaw(x, xmins=None, n_grid=120, min_tail=30):
    """Fit a continuous Pareto tail by KS-minimising ``xmin``.

    Candidate thresholds are observed quantiles rather than arbitrary rounded
    values. For every candidate, the MLE, likelihood and two-sided KS distance
    use exactly the same support. This is the estimator subsequently reapplied
    to every Monte Carlo replicate in :func:`powerlaw_gof_pvalue`.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < max(30, min_tail):
        return None
    if xmins is None:
        probs = np.linspace(0.05, 0.90, n_grid)
        xmins = np.unique(np.quantile(x, probs))
    best = None
    for xmin in np.asarray(xmins, dtype=float):
        if not np.isfinite(xmin) or xmin <= 0:
            continue
        tail = _tail(x, xmin)
        if len(tail) < min_tail:
            continue
        alpha = continuous_powerlaw_mle(tail, xmin=float(xmin))
        if not np.isfinite(alpha) or alpha <= 1:
            continue
        ks = ks_distance(tail, alpha, float(xmin))
        if best is None or ks < best["ks"] - 1e-15:
            best = {"xmin": float(xmin), "alpha": float(alpha), "ks": float(ks),
                    "n_tail": int(len(tail)), "n": int(x.size)}
    if best is None:
        return None
    tail = _tail(x, best["xmin"])
    best["loglike"] = float(powerlaw_loglike(tail, best["alpha"], best["xmin"]))
    best["xmax"] = float(np.max(x))
    best["decades_tail"] = float(np.log10(best["xmax"] / best["xmin"]))
    return best


def ks_distance(x_tail, alpha, xmin):
    """Two-sided KS statistic, including both empirical-CDF limits.

    At each ordered observation the supremum must compare the fitted CDF with
    both ``i/n`` and ``(i-1)/n``. Omitting the latter underestimates KS whenever
    the model CDF lies above the empirical step function.
    """
    x_tail = np.sort(np.asarray(x_tail, dtype=float))
    x_tail = x_tail[np.isfinite(x_tail) & (x_tail >= xmin)]
    n = x_tail.size
    if n == 0 or not np.isfinite(alpha) or alpha <= 1 or xmin <= 0:
        return float("nan")
    model_cdf = 1.0 - (x_tail / xmin) ** (-(alpha - 1.0))
    upper = np.arange(1, n + 1) / n
    lower = np.arange(0, n) / n
    return float(max(np.max(np.abs(upper - model_cdf)),
                     np.max(np.abs(lower - model_cdf))))


def fit_lognormal(x, xmin=None):
    """Lognormal fit, optionally conditioned on the common support x >= xmin.

    With a threshold, the likelihood includes the truncation normalisation and
    ``mu``/``sigma`` are re-optimised; simply truncating an untruncated fit is not
    the conditional MLE.
    """
    from scipy.optimize import minimize

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 3:
        return None
    sigma0, _, scale0 = stats.lognorm.fit(x, floc=0)
    mu0 = float(np.log(scale0))
    if xmin is None:
        return {"mu": mu0, "sigma": float(sigma0), "xmin": None,
                "loglike": float(np.sum(stats.lognorm.logpdf(x, sigma0, scale=scale0))),
                "truncated": False}
    xmin = float(xmin)
    tail = x[x >= xmin]
    if tail.size < 3:
        return None
    log_survival_xmin = stats.lognorm.logsf(xmin, sigma0, scale=np.exp(mu0))

    def objective(theta):
        mu, log_sigma = theta
        sigma = float(np.exp(log_sigma))
        log_z = stats.lognorm.logsf(xmin, sigma, scale=np.exp(mu))
        ll = stats.lognorm.logpdf(tail, sigma, scale=np.exp(mu)) - log_z
        if not np.all(np.isfinite(ll)):
            return 1e100
        return float(-np.sum(ll))

    opt = minimize(objective, [mu0, np.log(sigma0)], method="L-BFGS-B",
                   bounds=[(-50.0, 50.0), (-12.0, 5.0)])
    theta = opt.x if opt.success else np.array([mu0, np.log(sigma0)])
    mu, sigma = float(theta[0]), float(np.exp(theta[1]))
    log_z = float(stats.lognorm.logsf(xmin, sigma, scale=np.exp(mu)))
    log_contrib = stats.lognorm.logpdf(tail, sigma, scale=np.exp(mu)) - log_z
    return {"mu": mu, "sigma": sigma, "xmin": xmin, "log_z": log_z,
            "loglike": float(np.sum(log_contrib)), "log_contrib": log_contrib,
            "truncated": True, "optimiser_success": bool(opt.success)}


def fit_exponential(x, xmin=None):
    """Exponential or shifted-exponential fit on the declared common support."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    loc = 0.0 if xmin is None else float(xmin)
    tail = x[x >= loc]
    if tail.size < 3:
        return None
    scale = float(np.mean(tail - loc))
    if scale <= 0:
        return None
    log_contrib = stats.expon.logpdf(tail, loc=loc, scale=scale)
    return {"lambda": float(1.0 / scale), "scale": scale, "loc": loc,
            "xmin": loc if xmin is not None else None,
            "loglike": float(np.sum(log_contrib)), "log_contrib": log_contrib,
            "shifted": xmin is not None}


def fit_generalised_pareto(x, xmin=None):
    """Generalised Pareto fit to excesses over the declared threshold."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    threshold = 0.0 if xmin is None else float(xmin)
    excess = x[x >= threshold] - threshold
    if excess.size < 30:
        return None
    c, _, scale = stats.genpareto.fit(excess, floc=0)
    log_contrib = stats.genpareto.logpdf(excess, c, loc=0, scale=scale)
    return {"shape_c": float(c), "scale": float(scale), "xmin": threshold,
            "tail_index": float(1.0 / c) if c != 0 else float("inf"),
            "loglike": float(np.sum(log_contrib)), "log_contrib": log_contrib}


def powerlaw_gof_pvalue(x, alpha=None, xmin=None, n_mc=300, seed=12345,
                        n_grid=120, min_tail=30, return_details=False):
    """Clauset-type semiparametric KS goodness-of-fit test on a full sample.

    Each replicate contains a fitted Pareto tail and a nonparametric bootstrap
    sample below ``xmin``. Crucially, the complete fitting pipeline is reapplied
    to each replicate, including threshold selection, exponent estimation and the
    two-sided KS statistic. The plus-one correction avoids a zero Monte Carlo
    p-value. ``1 / (n_mc + 1)`` is the reporting floor, not an exact p-value.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    observed = fit_powerlaw(x, xmins=None if xmin is None else [xmin],
                            n_grid=n_grid, min_tail=min_tail)
    if observed is None:
        raise ValueError("power-law fit failed on the observed full sample")
    if alpha is not None and xmin is not None:
        # Preserve the caller's declared observed fit, while all replicates still
        # undergo free threshold selection. New analyses normally omit both.
        observed["alpha"] = float(alpha)
        observed["ks"] = ks_distance(_tail(x, xmin), alpha, xmin)
    a0, x0 = observed["alpha"], observed["xmin"]
    observed_ks = float(observed["ks"])
    n = x.size
    n_tail = int(np.sum(x >= x0))
    below = x[x < x0]
    rng = np.random.default_rng(seed)
    distances = np.empty(n_mc, dtype=float)
    failures = 0
    for k in range(n_mc):
        u = rng.random(n_tail)
        synth_tail = x0 * (1.0 - u) ** (-1.0 / (a0 - 1.0))
        if below.size:
            synth_below = rng.choice(below, size=n - n_tail, replace=True)
            synth = np.concatenate([synth_tail, synth_below])
        else:
            synth = synth_tail
        fit = fit_powerlaw(synth, n_grid=n_grid, min_tail=min_tail)
        if fit is None:
            failures += 1
            distances[k] = np.nan
        else:
            distances[k] = fit["ks"]
    valid = np.isfinite(distances)
    if not np.any(valid):
        raise RuntimeError("all Monte Carlo power-law fits failed")
    count = int(np.sum(distances[valid] >= observed_ks))
    p = float((count + 1) / (int(np.sum(valid)) + 1))
    if return_details:
        return {"p_value": p, "n_mc_requested": int(n_mc),
                "n_mc_valid": int(np.sum(valid)), "n_mc_failed": failures,
                "mc_floor": float(1.0 / (int(np.sum(valid)) + 1)),
                "observed_fit": observed, "synthetic_ks": distances,
                "method": "semiparametric full-sample KS; xmin and alpha re-estimated"}
    return p


def predicted_incompleteness(gaps, depositions, tau, total_length=None):
    """Non-parametric prediction of the missing fraction at cell size tau.

    Derived from M(tau) = E[(G - tau)_+] / E[cycle] for a renewal alternation of
    gaps G and deposition intervals D, with E[cycle] approximated either by the
    summed unit durations plus gaps (renewal geometry, default) or by the length
    of the record actually examined (``total_length``), which is the correct
    denominator when units overlap so that their durations double-count time.
    No functional form for the gap distribution is assumed, so a mismatch with
    the measured incompleteness directly indicates that the counting
    assumptions fail.
    """
    gaps = np.asarray(gaps, dtype=float)
    depositions = np.asarray(depositions, dtype=float)
    if total_length is not None:
        denom = float(total_length)
    else:
        denom = float(gaps.sum() + depositions.sum())
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.maximum(gaps - tau, 0.0)) / denom)


def incompleteness_shape(taus, t_cap, alpha):
    """Integral of the normalised truncated-Pareto survival, up to a constant.

    The returned shape is ``integral_tau^T S_G(g) dg`` without the constant
    denominator of ``S_G``. It includes the finite-cap term omitted by a pure
    power law and has the correct logarithmic limit at alpha = 2.
    """
    taus = np.asarray(taus, dtype=float)
    if np.any(taus <= 0) or t_cap <= 0 or np.any(taus >= t_cap):
        raise ValueError("require 0 < tau < T")
    if abs(alpha - 2.0) < 1e-7:
        return np.log(t_cap / taus) - 1.0 + taus / t_cap
    return ((t_cap ** (2.0 - alpha) - taus ** (2.0 - alpha)) / (2.0 - alpha)
            - t_cap ** (1.0 - alpha) * (t_cap - taus))


def fit_incompleteness_curve(taus, incomplete, t_cap, alpha_bounds=(1.05, 6.0)):
    """Fit a capped-Pareto incompleteness shape with a free amplitude.

    The fit is exploratory unless its residuals and boundary diagnostics are
    reported. Alpha is allowed on both sides of two; an optimum at a bound is
    flagged rather than interpreted as evidence for that numerical value.
    """
    from scipy.optimize import minimize_scalar

    taus = np.asarray(taus, dtype=float)
    inc = np.asarray(incomplete, dtype=float)
    ok = np.isfinite(inc) & (inc > 0) & np.isfinite(taus) & (taus > 0) & (taus < t_cap)
    taus, inc = taus[ok], inc[ok]
    if taus.size < 4:
        raise ValueError("at least four positive incompleteness points are required")
    ly = np.log(inc)

    def base(alpha):
        return np.maximum(incompleteness_shape(taus, t_cap, alpha), 1e-300)

    def best_at(alpha):
        log_a = float(np.mean(ly - np.log(base(alpha))))
        resid = log_a + np.log(base(alpha)) - ly
        return float(np.sum(resid ** 2)), log_a

    grid = np.linspace(alpha_bounds[0], alpha_bounds[1], 300)
    vals = np.array([best_at(a)[0] for a in grid])
    j = int(np.argmin(vals))
    lo = grid[max(j - 1, 0)]
    hi = grid[min(j + 1, len(grid) - 1)]
    if hi > lo:
        res_a = minimize_scalar(lambda a: best_at(a)[0], bounds=(lo, hi),
                                method="bounded", options={"xatol": 1e-7})
        alpha = float(res_a.x)
    else:
        alpha = float(grid[j])
    sse, log_a = best_at(alpha)
    pred = np.exp(log_a + np.log(base(alpha)))
    return {"alpha": alpha, "log_A": log_a, "sse": float(sse),
            "rmse_log": float(np.sqrt(np.mean((np.log(pred) - ly) ** 2))),
            "alpha_bounds": [float(alpha_bounds[0]), float(alpha_bounds[1])],
            "at_lower_bound": bool(abs(alpha - alpha_bounds[0]) < 1e-4),
            "at_upper_bound": bool(abs(alpha - alpha_bounds[1]) < 1e-4),
            "n_points": int(taus.size)}


def upper_incomplete_gamma(s, x):
    from scipy.special import gammaincc, gamma
    return float(gamma(s) * gammaincc(s, x))


# ------------------------------------------------------------ model comparison
def vuong_statistic(loglike_a, loglike_b, n, contributions_a=None, contributions_b=None):
    """Normalised log-likelihood ratio; positive favours A.

    With per-observation contributions available we use the standard scaling
    sigma^2 = var(d)/n; otherwise the user must supply them.
    """
    if contributions_a is None or contributions_b is None:
        return None
    d = np.asarray(contributions_a) - np.asarray(contributions_b)
    sigma = np.std(d, ddof=1)
    if sigma == 0 or not np.isfinite(sigma):
        return None
    n = len(d)
    r = np.sum(d)
    lr = r / (sigma * np.sqrt(n))
    # standard normal two-sided p-value for |LR|
    p = 2.0 * (1.0 - stats.norm.cdf(abs(lr)))
    return {"LR": float(lr), "p_value": float(p), "preferred": "A" if lr > 0 else "B"}


@dataclass
class ScalingResult:
    """Container returned by :func:`analyse_durations`."""
    x: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    powerlaw: Optional[dict] = None
    lognormal: Optional[dict] = None
    exponential: Optional[dict] = None
    generalised_pareto: Optional[dict] = None
    comparisons: dict = field(default_factory=dict)
    alpha_ci: Optional[tuple] = None
    notes: list = field(default_factory=list)


def analyse_durations(x, min_n=30, xmins=None, n_boot=0, seed=None):
    """Full distribution analysis for a sample of durations.

    Returns tail exponent with KS-based lower bound, alternative models, and
    pairwise normalised likelihood-ratio comparisons made on the *common support*
    (observations >= max(xmin candidates)) so that log-likelihoods are comparable
    in the sense discussed by Clauset et al. (2009).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    res = ScalingResult(x=x)
    if len(x) < min_n:
        res.notes.append("sample too small (n=%d)" % len(x))
        return res
    res.powerlaw = fit_powerlaw(x, xmins=xmins)
    if res.powerlaw is None:  # pragma: no cover
        res.notes.append("power-law fit failed")
        return res

    # All candidates are proper distributions on the same support x >= xmin.
    tail = _tail(x, res.powerlaw["xmin"])
    xmin = res.powerlaw["xmin"]
    res.lognormal = fit_lognormal(x, xmin=xmin)
    res.exponential = fit_exponential(x, xmin=xmin)
    res.generalised_pareto = fit_generalised_pareto(x, xmin=xmin)

    # Per-observation contributions for Vuong tests on that common support.
    a = res.powerlaw["alpha"]
    contrib_pl = ((a - 1.0) * np.log(xmin) - a * np.log(tail)
                  + np.log(a - 1.0))
    contrib_ln = res.lognormal["log_contrib"]
    contrib_ex = res.exponential["log_contrib"]
    res.comparisons["powerlaw_vs_lognormal"] = vuong_statistic(
        res.powerlaw["loglike"], res.lognormal["loglike"], len(tail),
        contrib_pl, contrib_ln)
    res.comparisons["powerlaw_vs_exponential"] = vuong_statistic(
        res.powerlaw["loglike"], res.exponential["loglike"], len(tail),
        contrib_pl, contrib_ex)

    if n_boot:
        rng = np.random.default_rng(seed)
        alphas = []
        for _ in range(n_boot):
            samp = rng.choice(x, size=len(x), replace=True)
            fit = fit_powerlaw(samp, xmins=xmins)
            if fit:
                alphas.append(fit["alpha"])
        if alphas:
            alphas = np.array(alphas)
            res.alpha_ci = (float(np.quantile(alphas, 0.025)),
                            float(np.quantile(alphas, 0.975)))
    return res


# --------------------------------------------------- completeness computations
def covered_fraction(intervals, t0, t1):
    """Fraction of [t0, t1] covered by a list of (top_age, bottom_age) intervals."""
    iv = sorted(np.asarray(intervals, dtype=float), key=lambda p: p[0])
    covered = 0.0
    cur = t0
    for a, b in iv:
        if b <= cur:
            continue
        if a > cur:
            cur = a
        if b > cur:
            covered += b - cur
            cur = b
        if cur >= t1:
            break
    return float(np.clip(covered / (t1 - t0), 0.0, 1.0))


def completeness_at_resolution(intervals, tau, t0, t1):
    """Fraction of cells of width tau intersected by at least one age interval.

    Definition follows the counting convention used in the completeness
    literature: a time cell counts as recorded if any part of it is represented
    by rock. Cells cannot overlap.
    """
    n_cells = int(np.floor((t1 - t0) / tau))
    if n_cells < 1:
        return np.nan
    start = np.arange(n_cells) * tau + t0
    stop = start + tau
    # a cell counts as recorded if any part of it intersects a represented
    # interval; margin-contact (measure zero) also counts, so floating-point
    # equality is absorbed by tol.
    tol = 1e-9 * max(float(t1 - t0), 1.0)
    mark = np.zeros(n_cells, dtype=bool)
    inv_tau = 1.0 / tau
    for a, b in intervals:
        if b <= a:
            continue
        # index range that can possibly overlap (cells touch [a,b])
        k_min = max(0, int(np.ceil((a - t0) * inv_tau)) - 1)
        k_max = min(n_cells - 1, int(np.ceil((b - t0) * inv_tau)) - 1)
        if k_max < k_min:
            continue
        start = np.arange(k_min, k_max + 1) * tau + t0
        stop = start + tau
        # positive-length overlap only; margin contact (zero length) is not
        # counted, which also absorbs floating-point boundary artefacts.
        inter = np.minimum(b, stop) - np.maximum(a, start)
        mark[k_min:k_max + 1] |= inter > tol
    return float(mark.mean())


def extraction_gaps(units_sorted):
    """Gaps between stacked age intervals.

    units_sorted: iterable of (top_age, bottom_age) ordered top-down.
    A positive value is unrepresented time between two successive units.
    """
    gaps = []
    covered_end = None
    overlap_count = 0
    for t, b in units_sorted:
        if covered_end is None:
            covered_end = b
            continue
        if t > covered_end:
            gaps.append(t - covered_end)
            covered_end = b
        else:
            overlap_count += 1
            covered_end = max(covered_end, b)
    return np.array(gaps, dtype=float), overlap_count


def empirical_ccdf(x):
    """(tail values, P(X > x)) sorted descending."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    surv = 1.0 - np.arange(1, n + 1) / n
    return x, surv
