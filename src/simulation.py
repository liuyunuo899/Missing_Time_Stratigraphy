# -*- coding: utf-8 -*-
"""Case Study II: forward model of deposition, stasis and erosion with exact
ground truth, plus the variational inverse used later for identifiability tests.

The model realises the state equation proposed in the source conversation,
    dh/dt = d(t) - e(t),   d, e >= 0,
with the extra structure that the source asked for: an external environmental
driver u(t), a threshold rule that switches the system between deposition,
stasis, and erosion, and a control volume in which every increment can be
followed. Because deposition adds material on top and erosion removes material
from the top, the fate of each increment is known exactly: an increment
deposited at time t survives if and only if the column surface never falls below
the level it reached when that increment was laid down.

All randomness is drawn from numpy.random.default_rng(seed); every experiment
records the seeds it used.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# --------------------------------------------------------------------- forcing
def forcing_periodic(n, dt, rng, period=5.0, amplitude=1.0, noise=0.15):
    """M1: single-timescale forcing plus white noise."""
    t = np.arange(n) * dt
    u = amplitude * np.sin(2.0 * np.pi * t / period)
    u[1:] += rng.normal(0.0, noise, n - 1)
    return u


def forcing_multiscale(n, dt, rng, n_scales=6, beta=1.0, f_lo=None, f_hi=None,
                       amplitude=1.0, noise=0.05):
    """M2: superposition of log-spaced scales with power spectral slope beta."""
    t = np.arange(n) * dt
    if f_lo is None:
        f_lo = 1.0 / (0.4 * n * dt)
    if f_hi is None:
        f_hi = 1.0 / (6.0 * dt)
    freqs = np.geomspace(f_lo, f_hi, n_scales)
    amps = amplitude * freqs ** (-beta / 2.0)
    amps = amps / np.linalg.norm(amps)
    phases = rng.uniform(0.0, 2.0 * np.pi, n_scales)
    u = np.zeros(n)
    for f, a, p in zip(freqs, amps, phases):
        u += a * np.sin(2.0 * np.pi * f * t + p)
    u[1:] += rng.normal(0.0, noise, n - 1)
    return u


def forcing_ar1(n, dt, rng, rho=0.999, sigma=1.0):
    """M3: red noise with lag-one autocorrelation rho."""
    u = np.zeros(n)
    eps = rng.normal(0.0, sigma * np.sqrt(max(1.0 - rho ** 2, 1e-12)), n)
    for i in range(1, n):
        u[i] = rho * u[i - 1] + eps[i]
    return u


def forcing_critical(n, dt, rng, rho=0.999, sigma=1.0, drift_period=40.0,
                     drift_amp=0.6):
    """M4: red noise plus a slow deterministic drift that pushes the system
    repeatedly across the threshold (a state-and-rate-dependent control)."""
    base = forcing_ar1(n, dt, rng, rho=rho, sigma=sigma)
    t = np.arange(n) * dt
    return base + drift_amp * np.sin(2.0 * np.pi * t / drift_period)


FORCINGS: dict[str, Callable] = {
    "periodic": forcing_periodic,
    "multiscale": forcing_multiscale,
    "ar1": forcing_ar1,
    "critical": forcing_critical,
}


# --------------------------------------------------------------- forward model
@dataclass
class ArchiveResult:
    """Container returned by :func:`run_archive`.

    Quantities are either *true* (known because we generated them) or *observed*
    (computed from the surviving column, i.e. what a geologist could measure).
    The distinction is enforced by the field names.
    """
    dt: float = 0.02
    u: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    depositing: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    eroding: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    h: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    survive: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    survival_fraction: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    n: int = 0
    params: dict = field(default_factory=dict)


def run_archive(u, dt, deposit_rate=2.0, erosion_rate=3.0, theta_dep=0.0,
                theta_ero=-0.6, sigma_age=0.0, seed=None):
    """Integrate the archive model for a given forcing.

    Units: dt in Myr, rates in m/kyr expressed as m per Myr is avoided; here
    rates are given in metres per Myr so that thicknesses come out in metres.
    The qualitative results are independent of the absolute values; the numbers
    are chosen to be of the order of observed shelf-to-basin accumulation rates.

    Returns an :class:`ArchiveResult` with the true history (h, deposition and
    erosion masks) and the survival mask derived from the running-surface rule.
    """
    u = np.asarray(u, dtype=float)
    n = u.size
    depositing = u > theta_dep
    eroding = u < theta_ero
    # integrate with a basement (h >= 0) constraint: erosion cannot remove more
    # material than the column holds, which is what makes the resulting
    # preservation fraction stay within [0, 1]
    h = np.zeros(n + 1)
    inc = np.zeros(n)
    increment = np.zeros(n)
    increment[depositing] = deposit_rate * dt
    increment[eroding] = -erosion_rate * dt
    for i in range(n):
        step = increment[i]
        if step < 0 and h[i] + step < 0.0:
            step = -h[i]
        inc[i] = step
        h[i + 1] = h[i] + step
    surface = h[1:].copy()           # surface level immediately after step i
    base = h[:-1].copy()             # base of the increment deposited at step i
    depositing = inc > 0
    eroding = inc < 0
    # Reverse running minimum of the future surface. If that surface falls into
    # a depositional increment, only the part below it survives. Retaining this
    # fraction avoids an arbitrary whole-layer deletion rule under partial erosion.
    future_min = np.minimum.accumulate(surface[::-1])[::-1]
    survival_fraction = np.zeros(n, dtype=float)
    thickness = surface - base
    survival_fraction[depositing] = np.clip(
        (future_min[depositing] - base[depositing]) / thickness[depositing], 0.0, 1.0)
    survive = survival_fraction > 1e-12  # any-part representation, as for real units
    if sigma_age > 0:
        rng = np.random.default_rng(seed)
        # Dating error shifts the apparent position of preserved intervals.
        shift = rng.normal(0.0, sigma_age, n)
        idx = np.arange(n) + np.round(shift / dt).astype(int)
        shifted_fraction = np.zeros(n, dtype=float)
        for i, j in enumerate(idx):
            if 0 <= j < n:
                shifted_fraction[j] = max(shifted_fraction[j], survival_fraction[i])
        survival_observed_fraction = shifted_fraction
    else:
        survival_observed_fraction = survival_fraction.copy()
    survival_observed = survival_observed_fraction > 1e-12
    # h reported here is the post-step surface, so that
    #   sum(positive increments) - |sum(negative increments)| = h[-1]
    # closes the mass balance exactly (checked in test_stats_tools / reports).
    return ArchiveResult(dt=dt, u=u, depositing=depositing, eroding=eroding,
                         h=surface, survive=survival_observed,
                         survival_fraction=survival_observed_fraction, n=n,
                         params={"deposit_rate_m_per_Myr": deposit_rate,
                                 "erosion_rate_m_per_Myr": erosion_rate,
                                 "theta_dep": theta_dep, "theta_ero": theta_ero,
                                 "sigma_age_Myr": sigma_age, "seed": seed})


def runs_of_true(mask):
    """Lengths (in time steps) of consecutive False runs inside the window."""
    out, count = [], 0
    for m in mask:
        if not m:
            count += 1
        elif count:
            out.append(count)
            count = 0
    return np.array(out, dtype=float)


def observed_gaps(survive, dt):
    """Hiatus durations as they would be read off the surviving column.

    The observed record window spans the oldest and the youngest preserved
    increment; gaps are the runs of unpreserved time inside that window. Hence
    they mix genuine non-deposition with intervals whose deposit was later
    removed, which is exactly the ambiguity investigated here.
    """
    idx = np.flatnonzero(survive)
    if idx.size < 2:
        return np.array([], dtype=float)
    lo, hi = idx.min(), idx.max()
    window = survive[lo:hi + 1]
    return runs_of_true(window) * dt


def true_nondeposition_gaps(depositing, dt):
    """Hiatus durations of the generating process before any removal."""
    return runs_of_true(depositing) * dt


def preservation_metrics(res):
    """Exact mass budget of the control volume (all values in metres of
    thickness, which is proportional to mass for constant density)."""
    inc = np.diff(np.concatenate([[0.0], res.h]))
    deposited = np.sum(inc[inc > 0])
    eroded = -np.sum(inc[inc < 0])
    final = max(float(res.h[-1]), 0.0) if res.h.size else 0.0
    positive = np.maximum(inc, 0.0)
    layer_preserved = float(np.sum(res.survival_fraction * positive))
    return {
        "gross_deposited_m": float(deposited),
        "eroded_m": float(eroded),
        "final_preserved_m": float(final),
        "P_gross": float(final / deposited) if deposited > 0 else float("nan"),
        "time_preserved_fraction_true": float(res.survival_fraction.mean()),
        "time_represented_any_part_fraction": float(res.survive.mean()),
        "time_preserved_equivalent_Myr": float(res.dt * res.survival_fraction.sum()),
        "layer_preservation_residual_m": float(layer_preserved - final),
        "mass_balance_residual_m": float(deposited - eroded - final),
    }


def completeness_from_survival(survive, dt, taus, t_index_start=None,
                               t_index_end=None):
    """Completeness of the surviving record measured exactly like the real data.

    A cell of width tau counts as recorded if any *surviving* increment falls in
    it. This makes the synthetic and the observational estimates directly
    comparable.
    """
    from stats_tools import completeness_at_resolution
    idx = np.flatnonzero(survive)
    if idx.size < 2:
        return np.full(len(taus), np.nan)
    lo = int(t_index_start if t_index_start is not None else idx.min())
    hi = int(t_index_end if t_index_end is not None else idx.max())
    iv = [(i * dt, (i + 1) * dt) for i in range(lo, hi + 1) if survive[i]]
    return np.array([completeness_at_resolution(iv, tau, lo * dt, (hi + 1) * dt)
                     for tau in taus])


# ------------------------------------------------------------------ experiment
def standardise(u):
    """Zero mean, unit standard deviation, so that thresholds mean the same thing
    for every forcing family."""
    u = np.asarray(u, dtype=float)
    s = u.std()
    return (u - u.mean()) / s if s > 0 else u - u.mean()


def run_case(kind, n=8000, dt=0.02, seed=1, forcing_kwargs=None,
             standardise_forcing=True, **archive_kwargs):
    """Generate one realisation end to end."""
    rng_master = np.random.default_rng(seed)
    forcing_seed = int(rng_master.integers(0, 2 ** 31 - 1))
    fseed = int(np.random.default_rng(forcing_seed).integers(0, 2 ** 31 - 1))
    kwargs = forcing_kwargs or {}
    u = FORCINGS[kind](n, dt, np.random.default_rng(fseed), **kwargs)
    if standardise_forcing:
        u = standardise(u)
    res = run_archive(u, dt, seed=seed + 7919, **archive_kwargs)
    res.params.update({"forcing": kind, "forcing_seed": fseed,
                       "forcing_kwargs": {k: float(v) for k, v in kwargs.items()
                                          if isinstance(v, (int, float))},
                       "n_steps": n, "span_Myr": float(n * dt)})
    return res
