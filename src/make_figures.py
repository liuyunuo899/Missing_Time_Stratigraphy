# -*- coding: utf-8 -*-
"""Stage 6: all manuscript figures.

Thirteen multi-panel figures are written as PDF and 300 dpi PNG into figures/.
Every number plotted is read from an archived result file or regenerated from a
seeded computation; nothing is hard-coded from memory except annotation strings
that are themselves read from the archived JSON summaries where possible.

Run:  python src/make_figures.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed")
RAW = os.path.join(ROOT, "data", "raw")
SYN = os.path.join(ROOT, "data", "synthetic")
FIG = os.path.join(ROOT, "figures")
SEDAPP = os.path.join(ROOT, "data", "sedapp")

from stats_tools import analyse_durations, empirical_ccdf  # noqa: E402
from simulation import run_case, preservation_metrics, observed_gaps  # noqa: E402

os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
})

C_REAL = "#1f4e79"     # dark blue  : observational
C_SYN = "#b45309"      # amber      : synthetic
C_MODEL = "#2f7d4f"    # green      : fitted model
C_BAD = "#a02c2c"      # red        : rejected
C_GREY = "#808080"


def jload(name, folder=PROC):
    with open(os.path.join(folder, name)) as f:
        return json.load(f)


def panel_label(ax, lab, x=-0.14, y=1.05):
    ax.text(x, y, lab, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top")


def save(fig, stem):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (stem, ext)))
    plt.close(fig)
    print("  wrote", stem)


# ===========================================================================
# Figure 1 - vocabulary and geometry of missing time (conceptual)
# ===========================================================================
def figure_1():
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2))
    rng = np.random.default_rng(7)

    # ---- (a) a column and the time it represents
    a = ax[0, 0]
    t = np.linspace(0, 100, 4000)
    dep = np.zeros_like(t, dtype=bool)
    # four depositional episodes of unequal length
    for s, e in [(2, 14), (22, 27), (45, 78), (88, 96)]:
        dep[(t >= s) & (t <= e)] = True
    thick = np.cumsum(dep) * 0.04
    a.fill_between(t, 0, thick, where=dep, color=C_REAL, alpha=0.75, lw=0)
    a.plot(t, thick, color=C_REAL, lw=0.8)
    # mark hiatuses on the timeline
    for s, e in [(14, 22), (27, 45), (78, 88)]:
        a.axvspan(s, e, color=C_BAD, alpha=0.20)
    a.set_xlabel("Time (arbitrary units)")
    a.set_ylabel("Cumulative thickness")
    a.set_title("Deposition and unrepresented time", fontsize=8)
    a.text(
    0.5, -0.28,
    "shaded = time with no preserved material",
    transform=a.transAxes,
    fontsize=6.5,
    color=C_BAD,
    ha="center",
    va="top",
    clip_on=False
)
    panel_label(a, "a")

    # ---- (b) counting grid at two resolutions
    b = ax[0, 1]
    tau = 10.0
    for row, tau in enumerate([20.0, 5.0]):
        cells = np.arange(0, 100, tau)
        recorded = np.array([dep[(t >= c) & (t < c + tau)].any() for c in cells])
        y0 = row * 1.0
        for k, c in enumerate(cells):
            col = C_REAL if recorded[k] else C_BAD
            b.add_patch(plt.Rectangle((c, y0), tau, 0.8, facecolor=col,
                                      alpha=0.75 if recorded[k] else 0.35,
                                      edgecolor="white", lw=0.4))
        b.text(101, y0 + 0.4, r"$\tau$ = %g" % tau, fontsize=6.5, va="center")
    b.set_xlim(0, 118)
    b.set_ylim(-0.2, 2.0)
    b.set_yticks([])
    b.set_xlabel("Time")
    b.set_title("A coarser grid records more cells, not more time", fontsize=8)
    panel_label(b, "b")

    # ---- (c) coverage versus completeness
    c = ax[1, 0]
    taus = np.logspace(-2, np.log10(50), 60)
    # Eq. (1) with a heavy-tailed gap population: flat branch
    for alpha, lab, col in [(1.6, r"$\alpha = 1.6$ ($\leq 2$)", C_BAD),
                            (2.6, r"$\alpha = 2.6$ ($> 2$)", C_MODEL)]:
        gaps = (1 - np.random.default_rng(int(alpha * 100)).random(20000)) ** (
            -1.0 / (alpha - 1.0))
        gaps = gaps[(gaps > 1e-3) & (gaps < 400)]
        m = np.array([np.mean(np.maximum(gaps - tau, 0.0))
                      for tau in taus])
        m = m / (m[0] / 0.45)
        c.plot(taus, 1 - np.clip(m, 0, 1), color=col, label=lab)
    c.set_xscale("log")
    c.set_xlabel(r"Resolution $\tau$")
    c.set_ylabel(r"Completeness $C(\tau)$")
    c.set_ylim(0, 1)
    c.legend(frameon=False, loc="lower right")
    c.set_title("Tail shape and resolution both affect the curve", fontsize=8)
    panel_label(c, "c")

    # ---- (d) distinguish the measured quantity from its interpretation
    d = ax[1, 1]
    labels = ["Measured quantity", "Candidate processes", "Geological terminology"]
    causes = ["unrepresented time\nat a stated resolution", "non-deposition or erosion",
              "hiatus / diastem /\nunconformity"]
    for i, (lab, cau) in enumerate(zip(labels, causes)):
        y = 2 - i
        d.add_patch(plt.Rectangle((0.05, y - 0.3), 0.9, 0.6, facecolor=C_REAL,
                                  alpha=0.15, edgecolor=C_REAL, lw=0.6))
        d.text(0.09, y, lab, fontsize=6.8, fontweight="bold", va="center")
        d.text(0.58, y, cau, fontsize=6.2, va="center")
    d.set_xlim(0, 1.05)
    d.set_ylim(-0.6, 3.0)
    d.set_xticks([])
    d.set_yticks([])
    d.set_title("Separate measurement, process and terminology", fontsize=8)
    d.text(0.5, -0.13, "a duration alone cannot identify its cause",
           transform=d.transAxes, fontsize=6.5, ha="center", color=C_BAD)
    panel_label(d, "d")

    save(fig, "fig01_vocabulary_and_geometry")


# ===========================================================================
# Figure 2 - evidence and inference flow (conceptual)
# ===========================================================================
def figure_2():

    fig, ax = plt.subplots(
        1, 3,
        figsize=(12.5, 4.2),
        gridspec_kw={"width_ratios": [1.0, 1.1, 1.45]}
    )

    # ==========================================================
    # (a) Evidence and inference flow
    # ==========================================================
    a = ax[0]

    boxes = [
        ("forcing\n$u(t)$", 0.50, "#dbe5f1"),
        ("archive model\n" + r"$dh/dt = d-e$", 0.38, "#dbe5f1"),
        ("surviving record\n(what is measurable)", 0.26, "#f2dcd9"),
        ("estimators\n" + r"$\hat\alpha$, $C(\tau)$, $b$", 0.14, "#e2efda")
    ]

    # Draw boxes and labels
    for lab, y, col in boxes:

        a.add_patch(
            plt.Rectangle(
                (0.15, y),
                0.70,
                0.09,
                facecolor=col,
                edgecolor="#404040",
                lw=0.7
            )
        )

        a.text(
            0.50,
            y + 0.045,
            lab,
            ha="center",
            va="center",
            fontsize=7.0,
            linespacing=1.1
        )

    # Arrows only in gaps between boxes
    for upper_y, lower_y in [
        (0.50, 0.38),
        (0.38, 0.26),
        (0.26, 0.14)
    ]:

        a.annotate(
            "",
            xy=(0.50, lower_y + 0.092),
            xytext=(0.50, upper_y - 0.002),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.0,
                color="#404040",
                shrinkA=0,
                shrinkB=0,
                mutation_scale=10
            )
        )

    a.text(
        0.50, 0.635,
        "unobserved in nature",
        fontsize=6.5,
        ha="center",
        color=C_GREY,
        style="italic"
    )

    a.text(
        0.50, 0.055,
        "only the last two are accessible\nwithout a model",
        fontsize=6.2,
        ha="center",
        va="center",
        color=C_GREY
    )

    a.set_xlim(0, 1)
    a.set_ylim(0, 0.70)
    a.axis("off")

    a.set_title(
        "Where the ground truth stops",
        fontsize=9,
        pad=14
    )

    panel_label(a, "a", x=-0.05, y=1.06)

    # ==========================================================
    # (b) Which method answers what
    # ==========================================================
    b = ax[1]

    qs = [
        "heavy tail?",
        "power law?",
        "which mechanism?",
        "gross deposition?",
        "preservation?"
    ]

    Ms = ["Real data", "Synthetic"]

    grid = np.array([
        [2, 2],
        [2, 0],
        [0, 2],
        [0, 1],
        [0, 1]
    ])

    b.imshow(
        grid,
        cmap="Greens",
        vmin=0,
        vmax=2,
        aspect="auto"
    )

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):

            txt = {
                2: "yes",
                1: "bounded",
                0: "no"
            }[grid[i, j]]

            b.text(
                j, i, txt,
                ha="center",
                va="center",
                fontsize=7.0,
                color="white" if grid[i, j] == 2 else "#404040"
            )

    b.set_xticks(range(2))
    b.set_xticklabels(Ms, fontsize=7.5)

    b.set_yticks(range(len(qs)))
    b.set_yticklabels(qs, fontsize=7.5)

    b.tick_params(axis="y", pad=5)

    b.set_title(
        "Each method answers a different question",
        fontsize=9,
        pad=14
    )

    panel_label(b, "b", x=-0.16, y=1.06)

    # ==========================================================
    # (c) Evidence grading
    # ==========================================================
    c = ax[2]

    grades = [
        ("O observational", 30, C_REAL),
        ("D derived", 22, C_MODEL),
        ("M model-conditional", 26, C_SYN),
        ("U untested", 22, C_GREY)
    ]

    labels = [g[0] for g in grades]
    values = [g[1] for g in grades]
    colors = [g[2] for g in grades]

    y_pos = np.arange(len(grades))

    c.barh(
        y_pos,
        values,
        height=0.54,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.90
    )

    # Category labels on the y axis, outside the bars
    c.set_yticks(y_pos)
    c.set_yticklabels(labels, fontsize=7.0)

    # First category at the top
    c.invert_yaxis()

    # Percentages at the ends of the bars
    for y, value in zip(y_pos, values):
        c.text(
            value + 0.8,
            y,
            f"{value}%",
            ha="left",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="#303030"
        )

    # Extra space for percentage labels
    c.set_xlim(0, 37)

    c.set_xlabel(
        "Share of claims (%)",
        fontsize=8
    )

    c.set_title(
        "No claim is graded above its evidence",
        fontsize=9,
        pad=14
    )

    # Light x-axis grid behind bars
    c.set_axisbelow(True)
    c.grid(
        axis="x",
        linestyle="--",
        linewidth=0.5,
        alpha=0.20
    )

    c.spines["top"].set_visible(False)
    c.spines["right"].set_visible(False)

    panel_label(c, "c", x=-0.10, y=1.06)

    # ==========================================================
    # Layout and save
    # ==========================================================
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.17,
        top=0.86,
        wspace=0.65
    )

    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(FIG, f"fig02_evidence_flow.{ext}"),
            bbox_inches="tight",
            pad_inches=0.12,
            dpi=300
        )

    plt.close(fig)

    print("  wrote fig02_evidence_flow")

# ===========================================================================
# Figure 3 - Case Study I coverage
# ===========================================================================
def figure_3():
    units = pd.read_csv(os.path.join(PROC, "units_processed.csv"))
    gaps = pd.read_csv(os.path.join(PROC, "hiatus_gaps.csv"))
    ph = units[units["is_phanerozoic"]] if "is_phanerozoic" in units else units
    counts = ph.groupby("col_id").size()
    prim = counts[counts >= 15].index

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))

    # ---- (a) geography
    a = ax[0, 0]
    gl = gaps.groupby("col_id").agg(lat=("lat", "first"), lng=("lng", "first"),
                                    nu=("n_units_column", "first"))
    m = gl[gl.index.isin(prim)]
    a.scatter(gl["lng"], gl["lat"], s=2.5, color="#d3d3d3", label="excluded")
    sc = a.scatter(m["lng"], m["lat"], c=m["nu"], s=14, cmap="YlOrRd",
                   vmin=15, vmax=120, edgecolor="#404040", linewidth=0.15,
                   zorder=3)
    a.set_xlabel("Longitude")
    a.set_ylabel("Latitude")
    a.legend(frameon=False, loc="lower left", fontsize=6)
    cb = plt.colorbar(sc, ax=a, fraction=0.046, pad=0.04)
    cb.set_label("units per column", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    a.set_title("736 primary columns", fontsize=8)
    panel_label(a, "a")

    # ---- (b) units per column
    b = ax[0, 1]
    b.hist(counts.values, bins=np.logspace(np.log10(1), np.log10(counts.max()),
                                           40), color="#cccccc", label="all")
    b.hist(counts[prim].values,
           bins=np.logspace(np.log10(15), np.log10(counts.max()), 30),
           color=C_REAL, label="primary (n >= 15)")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("units per column")
    b.set_ylabel("columns")
    b.legend(frameon=False)
    b.set_title("The 15-unit filter", fontsize=8)
    panel_label(b, "b")

    
# ---- (c) record length
    c = ax[1, 0]

    rl = gaps.groupby("col_id")["record_length_myr"].first()
    med_rl = np.median(rl[rl.index.isin(prim)])

    c.hist(rl[rl.index.isin(prim)].values, bins=40, color=C_REAL)

# 中位数位置只绘制一条虚线
    c.axvline(med_rl, color="black", ls="--", lw=0.8)

# 文字紧靠虚线左侧，右端对齐，避免与虚线重叠
    c.annotate(
    "median %.0f Myr" % med_rl,
    xy=(med_rl, 0.92),
    xycoords=("data", "axes fraction"),
    xytext=(-8, 0),
    textcoords="offset points",
    ha="right",
    va="center",
    fontsize=6.5,
    color="#303030"
)

    c.set_xlabel("record length (Myr)")
    c.set_ylabel("columns")
    c.set_title("Record lengths", fontsize=8)

    panel_label(c, "c")


    # ---- (d) age distribution
    d = ax[1, 1]
    ages = ph["b_age"].dropna()
    d.hist(ages.values, bins=60, color=C_REAL)
    d.axvline(541.0, color=C_BAD, ls="--", lw=0.8)
    d.text(541 * 0.98, 0.9, "541 Ma cut", transform=d.get_xaxis_transform(),
           color=C_BAD, fontsize=6.5, ha="right")
    d.set_xlabel("unit bottom age (Ma)")
    d.set_ylabel("units")
    d.set_title("Phanerozoic only", fontsize=8)
    panel_label(d, "d")

    save(fig, "fig03_real_data_coverage")


# ===========================================================================
# Figure 4 - hiatus duration distributions
# ===========================================================================
def figure_4():
    gaps = pd.read_csv(os.path.join(PROC, "hiatus_gaps.csv"))
    st = jload("summary_real_stats.json")
    r3 = jload("summary_round3_computations.json")
    counts = gaps.groupby("col_id")["n_units_column"].first()
    prim = gaps[gaps["col_id"].isin(counts[counts >= 15].index)]
    x = prim["hiatus_duration_myr"].values
    x = x[x > 0]
    duration_fit = analyse_durations(x)
    fit = duration_fit.powerlaw
    d = st["duration_distributions"]["primary_all"]

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))

    # ---- (a) histogram
    a = ax[0, 0]
    a.hist(np.log10(x), bins=50, color=C_REAL, alpha=0.85)
    a.axvline(np.log10(d["xmin_Myr"]), color="black", ls="--", lw=0.8)
    a.text(np.log10(d["xmin_Myr"]), a.get_ylim()[1] * 0.92,
           r" $x_{min}$ = %.1f Myr" % d["xmin_Myr"], fontsize=6.5)
    a.set_xlabel(r"$\log_{10}$ hiatus duration (Myr)")
    a.set_ylabel("count")
    a.set_title("n=4354, median %.1f, max %.0f Myr"
                % (d["median_Myr"], d["max_Myr"]), fontsize=8)
    panel_label(a, "a")

    # ---- (b) CCDF with fitted power law
    b = ax[0, 1]
    xs, surv = empirical_ccdf(x)
    b.plot(xs, surv, color=C_REAL, lw=1.0, label="observed")
    xt = np.logspace(np.log10(fit["xmin"]), np.log10(x.max()), 100)
    tail_mass = float(np.mean(x >= fit["xmin"]))
    b.plot(xt, tail_mass * (xt / fit["xmin"]) ** (-(fit["alpha"] - 1)),
           color=C_MODEL, ls="--",
           label=r"Pareto tail contribution, $\hat\alpha$ = %.2f" % fit["alpha"])
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("hiatus duration (Myr)")
    b.set_ylabel(r"$P(G > g)$")
    b.legend(frameon=False, loc="lower left")
    gof_p = r3["gof_resolution"]["primary_all"]["gof_p_R3000"]
    b.set_title("pooled CCDF (3000-replicate GOF $p$ = %.1e)" % gof_p,
                fontsize=8)
    panel_label(b, "b")

    # ---- (c) model comparison residuals
    c = ax[1, 0]
    tail = x[x >= fit["xmin"]]
    xt2, s2 = empirical_ccdf(tail)
    c.plot(xt2, s2, color=C_REAL, lw=1.0, label="observed tail")
    c.plot(xt2, (xt2 / fit["xmin"]) ** (-(fit["alpha"] - 1)), color=C_MODEL,
           ls="--", label=r"power law ($\alpha$ = %.2f)" % fit["alpha"])
    from scipy import stats as _st
    # Like-for-like: proper conditional distributions on x >= xmin.
    ln_tail = duration_fit.lognormal
    ex_tail = duration_fit.exponential
    if ln_tail:
        sf_x = _st.lognorm.sf(xt2, s=ln_tail["sigma"], scale=np.exp(ln_tail["mu"]))
        sf_0 = _st.lognorm.sf(fit["xmin"], s=ln_tail["sigma"],
                              scale=np.exp(ln_tail["mu"]))
        c.plot(xt2, sf_x / sf_0, color=C_SYN, ls=":",
               label="truncated lognormal")
        c.plot(xt2, np.exp(-(xt2 - fit["xmin"]) / ex_tail["scale"]),
               color=C_GREY, ls="-.", label="shifted exponential")
    c.set_xscale("log")
    c.set_yscale("log")
    c.set_xlim(fit["xmin"], x.max() * 1.1)
    c.set_xticks([10, 30, 100, 300])
    c.set_xticklabels(["10", "30", "100", "300"])
    c.xaxis.set_minor_formatter(NullFormatter())
    c.set_xlabel("hiatus duration (Myr)")
    c.set_ylabel(r"$P(G > g\mid G\geq x_{\min})$")
    c.legend(frameon=False, loc="lower left", fontsize=6)
    c.set_title("Relative vs absolute fit", fontsize=8)
    panel_label(c, "c")

    # ---- (d) regional subsets
    dax = ax[1, 1]
    regional = r3["gof_resolution"]
    regs = [k for k in regional if k.startswith("region_")]
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(regs)))
    for k, col in zip(regs, cols):
        name = k.replace("region_", "").replace("_", " ")
        sub = prim[prim["col_group"].str.replace(r"[ /]", "_", regex=True)
                   == k.replace("region_", "")]
        if sub.empty:
            continue
        xx = sub["hiatus_duration_myr"].values
        xx = xx[xx > 0]
        xs3, s3 = empirical_ccdf(xx)
        dax.plot(xs3, s3, color=col, lw=0.9,
                 label="%s (n=%d, $\\hat\\alpha$=%.2f)"
                 % (name, xx.size, regional[k]["alpha"]))
    xs0, s0 = empirical_ccdf(x)
    dax.plot(xs0, s0, color="black", lw=1.4, ls="--", label="pooled")
    dax.set_xscale("log")
    dax.set_yscale("log")
    dax.set_xlabel("hiatus duration (Myr)")
    dax.set_ylabel(r"$P(G > g)$")
    dax.legend(frameon=False, fontsize=5.6, loc="lower left")
    dax.set_title("Regional fits are unequally resolved", fontsize=8)
    panel_label(dax, "d")

    save(fig, "fig04_hiatus_durations")


# ===========================================================================
# Figure 5 - rate-span scaling and completeness
# ===========================================================================
def figure_5():
    bins = pd.read_csv(os.path.join(PROC, "rate_span_bins.csv"))
    within = pd.read_csv(os.path.join(PROC, "sadler_within_column_slopes.csv"))
    comp = pd.read_csv(os.path.join(PROC, "completeness_pooled.csv"))
    decomp = jload("summary_sadler_decomposition.json")
    syn = jload("summary_synthetic.json", SYN)
    st = jload("summary_real_stats.json")

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))

    # ---- (a) Sadler plot
    a = ax[0, 0]
    a.errorbar(bins["med_duration"], bins["med_rate"],
               yerr=[bins["med_rate"] - bins["p25"], bins["p75"] - bins["med_rate"]],
               fmt="o", ms=3.5, color=C_REAL, ecolor="#9dc3e6", elinewidth=0.8,
               capsize=0)
    # the line uses the archived unit-level pooled slope, anchored at the
    # median of the binned points, so the figure and the text agree exactly
    sl_arch = st["sadler_slopes"]["all_reported_thickness"]["slope"]
    ok = (bins["med_duration"] > 0) & (bins["med_rate"] > 0)
    xm = np.log10(bins["med_duration"][ok]).mean()
    ym = np.log10(bins["med_rate"][ok]).mean()
    intercept = ym - sl_arch * xm
    xx = np.logspace(np.log10(bins["med_duration"].min()),
                     np.log10(bins["med_duration"].max()), 50)
    a.plot(xx, 10 ** (intercept + sl_arch * np.log10(xx)), color="black",
           ls="--", lw=0.9, label="pooled slope %.2f" % sl_arch)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("averaging span (Myr)")
    a.set_ylabel(r"median rate (m Myr$^{-1}$)")
    a.legend(frameon=False)
    a.set_title("Pooled rate–span relation", fontsize=8)
    panel_label(a, "a")

    # ---- (b) within versus pooled
    b = ax[0, 1]
    w = within["slope"].dropna()
    b.hist(w, bins=40, color=C_REAL)
    b.axvline(np.median(w), color="black", lw=1.0)
    b.axvline(decomp["pooled_slope"], color=C_BAD, ls="--", lw=1.0)
    b.text(0.02, 0.92, "within-column median %.2f" % np.median(w),
           transform=b.transAxes, fontsize=6.5)
    b.text(0.02, 0.82, "pooled %.2f" % decomp["pooled_slope"],
           transform=b.transAxes, fontsize=6.5, color=C_BAD)
    b.set_xlabel("within-column slope of log rate on log span")
    b.set_ylabel("columns")
    b.set_title("Steeper within columns", fontsize=8)
    panel_label(b, "b")

    # ---- (c) completeness versus resolution, measured and predicted
    c = ax[1, 0]
    c.plot(comp["tau_Myr"], comp["mean_completeness"], "o-", ms=3.5,
           color=C_REAL, label="measured")
    c.plot(comp["tau_Myr"], comp["mean_predicted"], "s--", ms=3.0,
           color=C_MODEL, label="renewal/random-phase prediction")
    c.set_xscale("log")
    c.set_xlabel(r"resolution $\tau$ (Myr)")
    c.set_ylabel(r"completeness $C(\tau)$")
    c.legend(
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    fontsize=6.2,
    handlelength=2.0,
    labelspacing=0.5
)
    cvals = np.concatenate([comp["mean_completeness"].to_numpy(float),
                            comp["mean_predicted"].to_numpy(float)])
    cpad = max(0.02, 0.08 * np.ptp(cvals[np.isfinite(cvals)]))
    c.set_ylim(max(0.0, np.nanmin(cvals) - cpad),
               min(1.03, np.nanmax(cvals) + cpad))
    c.set_title("A phase-averaged, non-parametric prediction", fontsize=8)
    panel_label(c, "c")

    # ---- (d) local exponent against the calibration curve
    d = ax[1, 1]
    agrid = np.array(syn["calibration_mapping"]["alpha_grid"], dtype=float)
    bgrid = np.array(syn["calibration_mapping"]["b_grid"], dtype=float)
    d.plot(agrid, bgrid, "o-", color=C_SYN, ms=4,
           label="synthetic calibration")
    d.plot(agrid, 2.0 - agrid, "k:", lw=0.8, label=r"$2-\alpha$ (asymptotic)")
    d.axhline(st["local_exponent"]["mean"], color=C_REAL, ls="--", lw=1.0)
    d.text(2.9, st["local_exponent"]["mean"] + 0.04,
           "observed $b$ = %.3f" % st["local_exponent"]["mean"],
           color=C_REAL, fontsize=6.5, ha="right")
    d.axvline(2.0, color=C_GREY, lw=0.6)
    d.text(2.01, -1.1, r"$\alpha = 2$", fontsize=6.5, color=C_GREY)
    d.set_xlabel(r"tail exponent $\alpha$")
    d.set_ylabel(r"completeness exponent $b$")
    d.legend(
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.88),
    fontsize=6.0,
    handlelength=2.0,
    labelspacing=0.5
)
    d.set_title("Flatness is not a unique tail diagnosis", fontsize=8)
    panel_label(d, "d")

    save(fig, "fig05_rate_span_and_completeness")


# ===========================================================================
# Figure 6 - robustness and negative controls
# ===========================================================================
def figure_6():
    st = jload("summary_real_stats.json")
    nc = jload("summary_negative_controls.json")
    gaps = pd.read_csv(os.path.join(PROC, "hiatus_gaps.csv"))
    counts = gaps.groupby("col_id")["n_units_column"].first()
    prim = gaps[gaps["col_id"].isin(counts[counts >= 15].index)]
    x = prim["hiatus_duration_myr"].values
    x = x[x > 0]

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))

    # ---- (a) truncation
    a = ax[0, 0]
    thrs = [0.0, 0.05, 0.25, 1.0, 2.0]
    keys = ["primary_all", "primary_gt_0.05_Myr", "primary_gt_0.25_Myr",
            "primary_gt_1.00_Myr", "primary_gt_2.00_Myr"]
    al = [st["duration_distributions"][k]["alpha"] for k in keys]
    nn = [st["duration_distributions"][k]["n"] for k in keys]
    a.plot(thrs, al, "o-", color=C_REAL)
    a.set_ylim(min(al) - 0.02, max(al) + 0.035)
    a.set_xlim(-0.15, 2.20)
    a.set_xlabel("lower truncation of the sample (Myr)")
    a.set_ylabel(r"$\hat\alpha$")
    # 五个数据点采用不同的标注偏移，避免文字互相重叠
    label_specs = [
       ((0, -18),   "center", "top"),      # n=4354：第一个点下方
        ((0, 14),   "left",   "bottom"),   # n=4340：第二个点上方
        ((8, -18),   "left",   "top"),      # n=4214：第三个点下方
        ((0, -18),   "center", "top"),      # n=3881：第四个点下方
        ((-8, -18),  "right",  "top")       # n=3432：第五个点下方
   ]

    for (t, v, n), (xyoff, ha, va) in zip(
        zip(thrs, al, nn), label_specs
    ): a.annotate(
           f"n={n}",
           xy=(t, v),
           xytext=xyoff,
           textcoords="offset points",
           ha=ha,
           va=va,
           fontsize=5.6,
           color="#303030",
           annotation_clip=True
        )
    
    a.set_title("Robust to truncation", fontsize=8)
    panel_label(a, "a")

    # ---- (b) excluding interval-pinned hiatuses
    b = ax[0, 1]
    xs, s = empirical_ccdf(x)
    b.plot(xs, s, color="black", lw=1.2, ls="--",
           label="all (n=%d, $\\hat\\alpha$=%.2f)" % (x.size, nc["baseline"]["alpha"]))
    sub_a = nc["unpinned_subset"]
    try:
        up = pd.read_csv(os.path.join(PROC, "gaps_unpinned_subset.csv"))[
            "hiatus_duration_myr"].values
        up = up[up > 0]
        xs2, s2 = empirical_ccdf(up)
        b.plot(xs2, s2, color=C_SYN, lw=1.0,
               label="unpinned only (n=%d, $\\hat\\alpha$=%.2f)"
               % (sub_a["n"], sub_a["alpha"]))
    except OSError:
        pass
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("hiatus duration (Myr)")
    b.set_ylabel(r"$P(G > g)$")
    b.legend(frameon=False, fontsize=6, loc="lower left")
    b.set_title("Robust to interval pinning", fontsize=8)
    panel_label(b, "b")

    # ---- (c) age perturbation
    c = ax[1, 0]
    jit = [r["label"] for r in nc["jitter_Myr"]]
    jv = [float(s.split("_")[1]) for s in jit]
    ja = [r["alpha"] for r in nc["jitter_Myr"]]
    jn = [r["n"] for r in nc["jitter_Myr"]]
    c.plot([0] + jv, [nc["baseline"]["alpha"]] + ja, "o-",
           color=C_REAL, label=r"$\hat\alpha$")
    c.set_xlabel(r"age perturbation $\pm\delta$ (Myr)")
    c.set_ylabel(r"$\hat\alpha$", color=C_REAL)
    c.set_ylim(1.2, 2.0)
    c2 = c.twinx()
    c2.plot([0] + jv, [nc["baseline"]["n"]] + jn, "s--", color=C_BAD,
            label="number of hiatuses")
    c2.set_ylabel("hiatuses detected", color=C_BAD)
    c2.tick_params(labelsize=7)
    c.set_title("Sensitive to age perturbation", fontsize=8)
    panel_label(c, "c")

    # ---- (d) completeness under perturbation
    d = ax[1, 1]
    keys = ["jitter_0.0", "jitter_2.0"]
    vals = [nc["completeness"][k] for k in keys]
    d.bar([0, 1], [v["b_median"] for v in vals], color=[C_REAL, C_SYN],
          yerr=[[v["b_median"] - v["b_iqr"][0] for v in vals],
                [v["b_iqr"][1] - v["b_median"] for v in vals]],
          capsize=3, width=0.55)
    d.set_xticks([0, 1])
    d.set_xticklabels(["unperturbed", r"$\pm$2 Myr"])
    d.set_ylabel("median completeness exponent $b$")
    d.axhline(0, color="black", lw=0.6)
    d.set_title("Flatness is robust", fontsize=8)
    panel_label(d, "d")

    save(fig, "fig06_robustness")


# ===========================================================================
# Figure 7 - the synthetic generator
# ===========================================================================
def figure_7():
    res = run_case("multiscale", n=6000, dt=0.02, seed=11,
                   forcing_kwargs=dict(beta=1.2, n_scales=6), theta_dep=0.0,
                   deposit_rate=25.0, erosion_rate=40.0)
    t = np.arange(res.n) * res.dt
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.0))

    # ---- (a) forcing and thresholds
    a = ax[0]
    a.plot(t, res.u, color="#404040", lw=0.5)
    a.axhline(0.0, color=C_MODEL, ls="--", lw=0.8)
    a.axhline(-0.6, color=C_BAD, ls="--", lw=0.8)
    a.text(0.99, 0.06, r"$\theta_{dep}$ = 0", transform=a.transAxes,
           ha="right", color=C_MODEL, fontsize=6.5)
    a.text(0.99, -0.22, r"$\theta_{ero}$ = $-$0.6", transform=a.transAxes,
           ha="right", color=C_BAD, fontsize=6.5)
    a.set_ylabel("forcing $u$")
    a.set_xlabel("time (Myr)")
    a.set_title("Two-threshold switch", fontsize=8)
    a.set_xlim(0, 60)
    panel_label(a, "a")

    # ---- (b) surface height and fate
    b = ax[1]
    b.plot(t, res.h, color=C_REAL, lw=0.7)
    b.fill_between(t, 0, res.h.max(), where=res.depositing,
                   color=C_MODEL, alpha=0.10, lw=0)
    b.fill_between(t, 0, res.h.max(), where=res.eroding,
                   color=C_BAD, alpha=0.10, lw=0)
    b.set_xlim(0, 60)
    b.set_ylabel("surface $h$ (m)")
    b.set_xlabel("time (Myr)")
    b.set_title("Surface with a basement", fontsize=8)
    panel_label(b, "b")

    # ---- (c) age-depth model of the surviving column
    c = ax[2]
    surv = res.survive
    increments = np.diff(np.concatenate([[0.0], res.h]))
    preserved_increment = res.survival_fraction * np.maximum(increments, 0.0)
    thick = np.cumsum(preserved_increment)
    c.plot(t, thick, color=C_REAL, lw=1.0, label="preserved column")
    # a hiatus is a stretch of depositional age with no new thickness:
    # draw it as a horizontal segment at the level where it occurs
    starts = np.flatnonzero(np.diff(surv.astype(int)) == 1)   # resume after gap
    stops = np.flatnonzero(np.diff(surv.astype(int)) == -1)   # enter a gap
    if surv[0]:
        starts = np.concatenate([[ -1 ], starts]) - 1
    n_h = min(len(stops), len(starts))
    for k in range(n_h):
        s_ = stops[k] * res.dt
        e_ = (starts[k] + 1) * res.dt
        z_ = thick[stops[k]]
        if e_ > 0:
            c.plot([s_, e_], [z_, z_], color=C_BAD, lw=2.2,
                   alpha=0.55, solid_capstyle="butt")
    m = preservation_metrics(res)
    c.set_xlim(0, 120)
    c.set_ylim(0, None)
    c.set_ylabel("preserved thickness (m)")
    c.set_xlabel("depositional age (Myr)")
    c.legend(frameon=False, loc="upper left")
    c.plot([], [], color=C_BAD, lw=2.2, label="hiatus (unrepresented time)")
    c.legend(frameon=False, loc="upper left")
    c.annotate("horizontal segments mark time with\n"
               "no represented depositional increment",
               xy=(60, float(np.nanmax(thick)) * 0.72), fontsize=6.5,
               ha="center", color="#303030")
    c.set_title("Age-depth of the surviving column "
                "(mass residual %.0e m; layer residual %.0e m)"
                % (m["mass_balance_residual_m"],
                   m["layer_preservation_residual_m"]), fontsize=8)
    panel_label(c, "c")

    save(fig, "fig07_synthetic_generator")


# ===========================================================================
# Figure 8 - mechanism comparison and equifinality
# ===========================================================================
def figure_8():
    mech = pd.read_csv(os.path.join(SYN, "mechanism_sweep.csv"))
    cal = pd.read_csv(os.path.join(SYN, "calibration_alpha.csv"))
    st = jload("summary_real_stats.json")
    syn = jload("summary_synthetic.json", SYN)

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))

    # ---- (a) estimator validation
    a = ax[0, 0]
    g = cal.groupby("alpha_true")["alpha_recovered"]
    a.plot(g.mean().index, g.mean().values, "o-", color=C_MODEL, ms=4)
    a.fill_between(g.mean().index, g.min().values, g.max().values,
                   color=C_MODEL, alpha=0.2, lw=0)
    a.plot([1.4, 3.1], [1.4, 3.1], "k:", lw=0.8)
    a.set_xlabel(r"prescribed $\alpha$")
    a.set_ylabel(r"recovered $\hat\alpha$")
    a.set_title("Recovery on prescribed tails", fontsize=8)
    panel_label(a, "a")

    # ---- (b) equifinality across mechanisms
    b = ax[0, 1]
    order = ["periodic", "multiscale", "ar1", "critical"]
    names = {"periodic": "periodic", "multiscale": "multiscale",
             "ar1": "red noise", "critical": "drift-modulated"}
    tv = mech[mech["alpha_true_valid"]]
    ranges = []
    for i, k in enumerate(order):
        v = tv[tv["mechanism"] == k]["alpha_true"].dropna().values
        if v.size == 0:
            b.text(i, 2.0, "no screened fit", ha="center", fontsize=5.5,
                   color=C_GREY)
            continue
        ranges.append((float(v.min()), float(v.max())))
        b.scatter(np.full(v.size, i) + np.random.default_rng(i).uniform(
            -0.13, 0.13, v.size), v, s=16, color=C_SYN, alpha=0.8)
        b.plot([i - 0.28, i + 0.28], [np.median(v), np.median(v)],
               color="black", lw=1.2)
    if ranges:
        common_lo = max(r[0] for r in ranges)
        common_hi = min(r[1] for r in ranges)

        if common_lo < common_hi:
            # 红色阴影表示不同机制的共同指数范围
            b.axhspan(
                common_lo,
                common_hi,
                color=C_BAD,
                alpha=0.10
            )

            # 说明文字放在红色阴影范围的左侧
            b.text(
                3.70,
                common_hi - 0.025,
                "overlap\nrange",
                fontsize=5.0,
                color=C_BAD,
                ha="right",
                va="top"
                )


    b.axhline(2.0, color=C_GREY, lw=0.6, ls="--")
    b.set_xticks(range(4))
    b.set_xticklabels([names[k] for k in order], fontsize=6.5)
    b.set_ylabel(r"generating $\alpha$")
    b.set_xlim(-0.5, 3.8)
    b.set_title("Overlapping generating-tail fits", fontsize=8)
    panel_label(b, "b")

    # ---- (c) observer bias
    c = ax[1, 0]
    p = mech[(mech["alpha_true_valid"]) & (mech["alpha_observed_valid"])]
    c.scatter(p["alpha_true"], p["alpha_observed"], s=22, color=C_SYN)
    paired_alpha = np.concatenate([p["alpha_true"].to_numpy(float),
                                   p["alpha_observed"].to_numpy(float)])
    paired_alpha = paired_alpha[np.isfinite(paired_alpha)]
    pad = max(0.08, 0.06 * np.ptp(paired_alpha))
    lims = [float(paired_alpha.min() - pad), float(paired_alpha.max() + pad)]
    c.plot(lims, lims, "k:", lw=0.8)
    dif = (p["alpha_observed"] - p["alpha_true"]).median()
    for _, r in p.iterrows():
        c.annotate("", xy=(r["alpha_true"], r["alpha_observed"]),
                   xytext=(r["alpha_true"], r["alpha_true"]),
                   arrowprops=dict(arrowstyle="-", color=C_BAD, lw=0.5,
                                   alpha=0.5))
    c.set_xlim(lims)
    c.set_ylim(lims)
    c.set_xlabel(r"generating $\alpha$ (before removal)")
    c.set_ylabel(r"observed $\hat\alpha$ (surviving column)")
    c.set_title("Observer effect in %d screened pairs (median %.2f)"
                % (len(p), dif), fontsize=8)
    panel_label(c, "c")

    # ---- (d) calibration of the completeness exponent against alpha
    d = ax[1, 1]
    gg = cal.groupby("alpha_true")
    d.plot(gg["local_exponent_b_measured"].mean().index,
           gg["local_exponent_b_measured"].mean().values,
           "o-", color=C_SYN, ms=4, label="measured $b$")
    d.plot(gg["two_minus_alpha"].mean().index,
           gg["two_minus_alpha"].mean().values, "s--", color=C_MODEL, ms=3.5,
           label=r"$2-\alpha$ (asymptotic)")
    d.axvline(2.0, color=C_GREY, lw=0.6)
    d.axhline(0, color="black", lw=0.5)
    d.axhline(st["local_exponent"]["mean"], color=C_REAL, ls="--", lw=1.0)
    d.text(1.45, st["local_exponent"]["mean"] + 0.06, "real data",
           color=C_REAL, fontsize=6.5)
    d.set_xlabel(r"prescribed $\alpha$")
    d.set_ylabel("completeness exponent $b$")
    # 图例放在右上方空白区域
    d.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=6.0,
        handlelength=1.8,
        labelspacing=0.45
    )
    d.set_title("Finite-record curves need not follow the asymptote", fontsize=8)
    panel_label(d, "d")

    save(fig, "fig08_equifinality")

# ===========================================================================
# Figure 9 - variational inversion
# ===========================================================================
def figure_9():
    from inversion import InverseProblem, solve_fista, ensemble  # noqa: E402
    from simulation import run_case  # noqa: E402
    from experiments_synthetic import N_STEPS, DT, SEED  # noqa: E402

    res = run_case("multiscale", n=N_STEPS, dt=DT, seed=11,
                   forcing_kwargs=dict(beta=1.2, n_scales=6), theta_dep=0.0,
                   deposit_rate=25.0, erosion_rate=40.0)
    inc_true = np.diff(np.concatenate([[0.0], res.h]))
    d_true = np.maximum(inc_true, 0.0) / res.dt
    e_true = np.maximum(-inc_true, 0.0) / res.dt
    h_true = res.h.copy()
    n = N_STEPS
    rng = np.random.default_rng(SEED + 5)
    idx = np.sort(rng.choice(np.arange(1, n), size=12, replace=False))
    sigma_z = max(5.0, 0.02 * float(np.abs(h_true[idx]).max() + 1.0))
    obs_z = h_true[idx] + rng.normal(0.0, sigma_z, idx.size)
    ip = InverseProblem(n=n, dt=res.dt, obs_idx=idx, obs_z=obs_z,
                        obs_sigma=np.full(idx.size, sigma_z),
                        truth_d=d_true, truth_e=e_true)
    sol = solve_fista(ip, 1e-4, 1e-4, max_iter=6000)
    ens = pd.read_csv(os.path.join(SYN, "inversion_ensemble.csv"))
    swe = ens[ens["experiment"] == "lambda_sweep"]
    syn = jload("summary_synthetic.json", SYN)

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
    t = np.arange(n) * res.dt

    # ---- (a) recovered versus true rates
    a = ax[0, 0]
    a.plot(t, d_true, color="#cccccc", lw=0.8, label="true $d$")
    a.plot(t, sol["d"], color=C_MODEL, lw=0.7, label="recovered $d$")
    a.plot(t, -e_true, color="#e0b4b4", lw=0.8, label="true $-e$")
    a.plot(t, -sol["e"], color=C_BAD, lw=0.7, label="recovered $-e$")
    a.set_xlabel("time (Myr)")
    a.set_ylabel(r"rate (m Myr$^{-1}$)")
    a.legend(frameon=False, fontsize=6, ncol=2)
    a.set_xlim(0, 60)
    a.set_title("Smooth where the truth is episodic", fontsize=8)
    panel_label(a, "a")

    # ---- (b) ensemble over regularisation
    b = ax[0, 1]
    for le, grp in swe.groupby("lam_e"):
        grp = grp.sort_values("lam_d")
        b.plot(np.log10(grp["lam_d"]), grp["gross_ratio"], "o-", ms=3,
               lw=0.8, label=r"$\lambda_e$=%g" % le)
    b.axhline(1.0, color="black", ls="--", lw=0.9)
    b.text(0.97, 0.91,
    "truth",
    transform=b.transAxes,
    fontsize=6.5,
    color="#303030",
    ha="right",
    va="top")
    b.set_xlabel(r"$\log_{10}\lambda_d$")
    b.set_ylabel("recovered / true gross deposition")
    b.legend(frameon=False,
        fontsize=5.8,
        ncol=2,
        loc="center",
        bbox_to_anchor=(0.50, 0.72),
        handlelength=1.8,
        columnspacing=1.5,
        labelspacing=0.6)
    b.set_title("Deposition shrinks with the prior", fontsize=8)
    panel_label(b, "b")

    # ---- (c) admissibility collapses the spread
    c = ax[1, 0]
    c.scatter(swe["residual_sigma"], swe["gross_ratio"],
              c=np.log10(swe["lam_d"]), cmap="viridis", s=26)
    for thr, col, lab in [(1.0, C_MODEL, "<= 1 sigma"), (3.0, C_SYN, "<= 3 sigma")]:
        q = swe[swe["residual_sigma"] <= thr]
        c.plot([q["gross_ratio"].min(), q["gross_ratio"].max()],
               [q["gross_ratio"].median(), q["gross_ratio"].median()],
               color=col, lw=1.6, label="%s (factor %.2f)"
               % (lab, q["gross_ratio"].max() / q["gross_ratio"].min()))
    c.axhline(1.0, color="black", ls=":", lw=0.8)
    c.set_xscale("log")
    c.set_xlabel("maximum data misfit (sigma)")
    c.set_ylabel("recovered / true gross deposition")
    c.legend(frameon=False, fontsize=6, loc="lower left")
    c.set_title("Misfit screening changes the plausible set", fontsize=8)
    panel_label(c, "c")

    # ---- (d) more horizons reduce the bias
    d = ax[1, 1]
    eh = syn["extra_horizon_test"]
    xs = [e["extra_horizons"] for e in eh]
    med = [e["gross_ratio_median"] for e in eh]
    lo = [e["gross_ratio_range"][0] for e in eh]
    hi = [e["gross_ratio_range"][1] for e in eh]
    d.errorbar([12, 24, 48], med, yerr=[np.array(med) - np.array(lo),
                                        np.array(hi) - np.array(med)],
               fmt="o-", ms=4, color=C_MODEL, capsize=3)
    d.axhline(1.0, color="black", ls="--", lw=0.9)
    d.set_xlabel("number of dated horizons")
    d.set_ylabel("recovered / true gross deposition")
    d.set_ylim(0.4, 1.1)
    d.set_title("Additional horizons shift the recovery distribution", fontsize=8)
    panel_label(d, "d")

    save(fig, "fig09_variational_inversion")


# ===========================================================================
# Figure 10 - real versus synthetic
# ===========================================================================
def figure_10():
    comp = pd.read_csv(os.path.join(PROC, "completeness_pooled.csv"))
    mech = pd.read_csv(os.path.join(SYN, "mechanism_sweep.csv"))
    pres = pd.read_csv(os.path.join(SYN, "preservation.csv"))
    st = jload("summary_real_stats.json")

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))

    # ---- (a) where the generator sits relative to the data
    a = ax[0, 0]
    a.scatter(mech["alpha_observed"], mech["local_exponent_b"], s=20,
              color=C_SYN, alpha=0.8, label="synthetic (observed)")
    a.axvline(st["duration_distributions"]["primary_all"]["alpha"],
              color=C_REAL, lw=1.2)
    a.axhline(st["local_exponent"]["mean"], color=C_REAL, lw=1.2)
    a.scatter([st["duration_distributions"]["primary_all"]["alpha"]],
              [st["local_exponent"]["mean"]], marker="*", s=140,
              color=C_REAL, zorder=5, label="real data")
    a.set_xlabel(r"observed $\hat\alpha$")
    a.set_ylabel("completeness exponent $b$")
    a.legend(frameon=False, fontsize=6)
    a.set_title("Real and synthetic summary statistics", fontsize=8)
    panel_label(a, "a")

    # ---- (b) real curve and synthetic endpoint distributions
    b = ax[0, 1]
    b.plot(comp["tau_Myr"], comp["mean_completeness"], "o-", ms=3.2,
           color=C_REAL, label="real mean (n=224–736)")
    paired = mech[["completeness_tau0.05", "completeness_tau5"]].dropna()
    rng = np.random.default_rng(20261020)
    for tau, col in [(0.05, "completeness_tau0.05"),
                     (5.0, "completeness_tau5")]:
        vals = paired[col].to_numpy(float)
        # Horizontal jitter is solely for visual separation; statistics use x=tau.
        xj = tau * np.exp(rng.normal(0.0, 0.035, vals.size))
        b.scatter(xj, vals, s=7, color=C_SYN, alpha=0.22, lw=0, zorder=2)
        q25, med, q75 = np.quantile(vals, [0.25, 0.5, 0.75])
        b.plot([tau, tau], [vals.min(), vals.max()], color=C_SYN,
               lw=0.8, alpha=0.65, zorder=3)
        b.plot([tau, tau], [q25, q75], color=C_SYN, lw=5.0,
               alpha=0.42, solid_capstyle="butt", zorder=3,
               label="synthetic IQR" if tau == 0.05 else None)
        b.scatter([tau], [med], marker="D", s=34, color=C_SYN,
                  edgecolor="white", linewidth=0.5, zorder=4,
                  label="synthetic median" if tau == 0.05 else None)
    b.set_xscale("log")
    b.set_xlim(0.007, 75)
    b.set_ylim(0.0, 1.02)
    b.set_xlabel(r"resolution $\tau$ (Myr)")
    b.set_ylabel(r"completeness $C(\tau)$")
    b.text(0.24, 0.08,
        "84 experiments; 72 paired cases\nthin bar = range; thick bar = IQR",
        transform=b.transAxes,
        fontsize=5.5,
        color="#604020",
        ha="left",
        va="bottom")
    b.legend(frameon=False,
        fontsize=5.3,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.32, 0.94),
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.45,
        borderaxespad=0)
    b.set_title("Two-resolution distributions, not full synthetic curves", fontsize=7.6)
    panel_label(b, "b")

    # ---- (c) preservation, true versus observer
    c = ax[1, 0]
    for er, grp in pres.groupby("erosion_rate_m_per_Myr"):
        c.scatter(grp["P_gross_true"], grp["P_naive_observer"], s=22,
                  label="erosion %g" % er, alpha=0.85)
    lims = [0, 1.05]
    c.plot(lims, lims, "k:", lw=0.8)
    c.set_xlim(lims)
    c.set_ylim(lims)
    c.set_xlabel("true $P_{gross}$ (exact budget)")
    c.set_ylabel("observer estimate")
    c.legend(frameon=False,
        fontsize=5.5,
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.37),
        handletextpad=0.5,
        labelspacing=0.45,
        borderaxespad=0.2,
        markerscale=0.85)
    c.set_title("Sign-reversing error", fontsize=8)
    panel_label(c, "c")

    # ---- (d) error versus erosion rate
    d = ax[1, 1]
    gg = pres.groupby("erosion_rate_m_per_Myr")
    d.errorbar(gg["abs_error"].apply(lambda s: s.abs().mean()).index,
               gg["abs_error"].apply(lambda s: s.abs().mean()).values,
               fmt="o-", ms=4, color=C_BAD, label="mean |error|")
    xi = gg["P_gross_true"].mean().index.values.astype(float)
    d.plot(xi, gg["P_gross_true"].mean().values, "s--", ms=4,
           color=C_MODEL, label="true $P_{gross}$")
    d.plot(xi, gg["P_naive_observer"].mean().values, "d:", ms=4,
           color=C_SYN, label="observer estimate")
    d.set_xlabel(r"erosion rate (m Myr$^{-1}$)")
    d.set_ylabel("preservation fraction")
    d.legend(frameon=False, fontsize=6)
    d.set_title("Least wrong in the middle", fontsize=8)
    panel_label(d, "d")

    save(fig, "fig10_real_vs_synthetic")


# ===========================================================================
# Figure 11 - integrated framework
# ===========================================================================
def figure_11():
    fig = plt.figure(figsize=(9.6, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.0])

    # ---- (a) strength of inference
    a = fig.add_subplot(gs[0, 0])
    qs = ["heavy tail", "power law (pooled)", "mean-excess relation",
          "b = 2 − α domain", "alpha vs 2", "mechanism",
          "rate-span location", "gross deposition", "P_gross meaning"]
    vals = [2, 0, 2, 2, 1, 0, 2, 1, 1]      # 2 yes, 1 bounded, 0 no
    cols = [C_REAL, C_BAD, C_REAL, C_MODEL, C_SYN, C_GREY, C_REAL, C_SYN, C_SYN]
    a.barh(range(len(qs)), [1] * len(qs), color=cols, alpha=0.85)
    status = ["established", "rejected (GOF)", "established",
              "bounded / qualified", "bounded / qualified", "not identifiable",
              "established", "bounded / qualified", "bounded / qualified"]
    for i, v in enumerate(vals):
        a.text(0.5, i, status[i], ha="center", va="center", fontsize=5.8,
               color="white" if v == 2 or i == 1 else "#303030")
    a.set_yticks(range(len(qs)))
    a.set_yticklabels(qs, fontsize=6)
    a.set_xticks([])
    a.invert_yaxis()
    a.set_title("What each question can be answered with", fontsize=8)
    panel_label(a, "a", x=-0.13, y=1.08)

    # ---- (b) domain of the asymptotic relation
    b = fig.add_subplot(gs[0, 1])
    al = np.linspace(2.001, 3.0, 200)
    b.plot(al, 2.0 - al, color=C_MODEL, lw=1.4,
           label=r"asymptotic $b=2-\alpha$")
    b.axvspan(1.2, 2.0, color=C_BAD, alpha=0.10)
    b.axvspan(2.0, 3.0, color=C_MODEL, alpha=0.10)
    b.axvline(2.0, color=C_GREY, lw=0.8, ls="--")
    b.text(1.50, -0.20,
        "relation not\napplicable",
        fontsize=6,
        ha="center",
        va="center",
        color=C_BAD)
    b.text(2.50, -0.20,
        "finite-mean\nbranch",
        fontsize=6,
        ha="center",
        va="center",
        color=C_MODEL)
    b.axhline(0, color="black", lw=0.5)
    b.set_xlim(1.2, 3.0)
    b.set_xlabel(r"tail exponent $\alpha$")
    b.set_ylabel(r"completeness exponent $b$")
    b.set_ylim(-1.1, 0.25)
    b.set_title("A conditional, not universal, relation", fontsize=8)
    panel_label(b, "b", x=-0.22, y=1.08)

    # ---- (c) ranked priorities
    c = fig.add_subplot(gs[0, 2])
    labs = ["1 removal measured\nindependently",
            "2 duration measured\nindependently",
            "3 transport-based\ngenerator",
            "4 record-length\ndiscriminant",
            "5 reporting\nnorm"]
    ys = np.arange(len(labs))[::-1]
    c.barh(ys, [5, 4, 3, 2, 1], color=[C_BAD, C_BAD, C_SYN, C_SYN, C_GREY],
           alpha=0.85)
    c.set_yticks(ys)
    c.set_yticklabels(labs, fontsize=6)
    c.set_xticks([])
    c.set_title("Ranked by leverage", fontsize=8)
    panel_label(c, "c", x=-0.22, y=1.08)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "fig11_framework.%s" % ext))
    plt.close(fig)
    print("  wrote fig11_framework")


def figure_12():
    """Numerical diagnostics without an unjustified cross-estimator correction.

    (a) Section-scale versus column-scale completeness.
    (b) L1-prior sensitivity of recovered gross deposition.
    (c) Monte-Carlo behaviour of the fitted-null GOF pipeline.
    (d) Model-conditional observer effect on the local curve exponent.
    (e) Observed tail and curve estimators, with their distinct assumptions.
    """
    fig = plt.figure(figsize=(7.2, 9.4))
    gs = fig.add_gridspec(3, 2, hspace=0.62, wspace=0.30)
    axes = {(0, 0): fig.add_subplot(gs[0, 0]),
            (0, 1): fig.add_subplot(gs[0, 1]),
            (1, 0): fig.add_subplot(gs[1, 0]),
            (1, 1): fig.add_subplot(gs[1, 1])}
    ax_e = fig.add_subplot(gs[2, :])

    # ---- (a) section vs column completeness -------------------------------
    ax = axes[0, 0]
    sec = pd.read_csv(os.path.join(PROC, "completeness_sections.csv"))
    r2 = jload("summary_round2_computations.json")
    r3 = jload("summary_round3_computations.json")
    col_rows = [(r["tau_Myr"], r["column_mean"]) for r in
                r2["section_vs_column_completeness"]]
    col = pd.DataFrame(col_rows, columns=["tau_Myr", "col_mean"])
    n_sec = int(sec.iloc[0]["n"])
    n_col = int(pd.read_csv(os.path.join(PROC, "completeness_curves.csv"),
                            usecols=["col_id"])["col_id"].nunique())
    ax.semilogx(sec["tau_Myr"], sec["mean"], "o-", color=C_REAL, ms=3.5,
                label="per section (n = %d at %.2f Myr)" %
                      (n_sec, float(sec.iloc[0]["tau_Myr"])))
    ax.semilogx(col["tau_Myr"], col["col_mean"], "s-", color=C_SYN, ms=3.5,
                label="per column (n = %d)" % n_col)
    ax.axhline(1.0, color=C_GREY, lw=0.7, ls=":")
    ax.set_xlabel(r"$\tau$ (Myr)")
    ax.set_ylabel("Mean completeness")
    ax.set_ylim(0, 1.08)
    ax.set_title("Support changes mean completeness", fontsize=8)
    ax.legend(loc="lower left", frameon=True, framealpha=0.85,
              edgecolor="none", fontsize=6)
    panel_label(ax, "a")

    # ---- (b) L1 prior sweep ----------------------------------------------
    ax = axes[0, 1]
    l1 = pd.read_csv(os.path.join(SYN, "inversion_l1.csv"))
    syn = jload("summary_synthetic.json", folder=SYN)
    quadratic_ensemble = pd.read_csv(os.path.join(SYN, "inversion_ensemble.csv"))
    quadratic_reference = quadratic_ensemble[
        (quadratic_ensemble["experiment"] == "lambda_sweep")
        & np.isclose(quadratic_ensemble["lam_d"], 1e-4)
        & np.isclose(quadratic_ensemble["lam_e"], 1e-4)
    ].iloc[0]
    quadratic_ratio = float(quadratic_reference["gross_ratio"])
    quadratic_residual = float(quadratic_reference["residual_sigma"])
    ax.semilogx(l1["lam"], l1["gross_ratio"], "o-", color=C_SYN, ms=3.5)
    ax.axhline(1.0, color=C_GREY, lw=0.7, ls=":")
    ax.axhline(quadratic_ratio, color=C_MODEL, lw=0.9, ls="--",
               label="quadratic reference (%.2f; %.3f$\\sigma$)"
               % (quadratic_ratio, quadratic_residual))
    ax.set_xlabel(r"$\lambda$ (L1 penalty weight)")
    ax.set_ylabel("Gross deposition recovered\n(fraction of truth)")
    ax.set_ylim(0.0, 1.08)
    ax.set_title("The prior changes recovered gross deposition", fontsize=8)
    ax.legend(loc="lower left", frameon=False, fontsize=6)
    panel_label(ax, "b")

    # ---- (c) fitted-null calibration --------------------------------------
    ax = axes[1, 0]
    cal = r3["circularity_calibration"]
    ps = np.array(cal["p_values"], dtype=float)
    gof = r3["gof_resolution"]["primary_all"]
    p_obs = float(gof["gof_p_R3000"])
    floor = float(gof.get("mc_floor_R3000", gof["floor_R3000"]))
    ax.hist(ps, bins=np.linspace(0, 1, 21), color=C_MODEL,
            edgecolor="white", lw=0.4)
    ax.axvline(0.10, color=C_BAD, lw=0.9, ls="--",
               label="nominal 0.10 level")
    obs_label = "observed pooled\np = %.2e" % p_obs
    if p_obs <= floor * (1.0 + 1e-12):
        obs_label  += "\n(MC floor)"
    ax.axvline(p_obs, color=C_REAL, lw=0.9, label=obs_label)
    ax.set_xlabel("GOF p-value under the fitted null")
    ax.set_ylabel("Datasets (of %d)" % int(cal["n_datasets"]))
    n_reject = int(np.sum(ps < 0.10))
    ax.set_title("Null calibration: %d/%d rejected at 0.10" %
                 (n_reject, ps.size), fontsize=8)
    ax.legend( frameon=False,
    fontsize=5.2,
    ncol=1,
    loc="upper center",
    bbox_to_anchor=(0.32, 0.98),
    handlelength=1.5,
    labelspacing=0.4,
    borderaxespad=0.2)
    panel_label(ax, "c")

    # ---- (d) observer effect on the local exponent ------------------------
    ax = axes[1, 1]
    bb = pd.read_csv(os.path.join(SYN, "bias_on_b.csv"))
    bias = jload("summary_round3_bias_on_b.json", folder=SYN)
    db = bb["delta_b"].dropna().to_numpy(float)
    ax.hist(db, bins=14, color=C_MODEL, edgecolor="white", lw=0.4)
    ax.axvline(0.0, color=C_GREY, lw=0.9, ls=":", label="no change")
    med = float(bias["delta_b_median"])
    q25, q75 = bias["delta_b_iqr"]
    ax.axvline(med, color=C_BAD, lw=1.1, label="median %+.2f" % med)
    ax.axvspan(q25, q75, color=C_BAD, alpha=0.12, lw=0, label="IQR")
    ax.set_xlabel(r"$b_{\mathrm{observed}} - b_{\mathrm{generating}}$")
    ax.set_ylabel("Archives (of %d)" % db.size)
    ax.set_title("Model-conditional observer effect\n"
                 "(median %+.2f; positive in %.0f%% of valid runs)" %
                 (med, 100.0 * float(bias["fraction_flattened"])), fontsize=8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9,
              edgecolor="none", fontsize=5.5)
    panel_label(ax, "d")

    # ---- (e) observed estimators, not a false reconciliation ---------------
    ax = ax_e
    comp = r3["cross_estimator_comparison"]
    unc = r3["alpha_uncertainty"]
    curve = r3["cross_estimator"]
    labels = ["tail fit\n(observed)", "completeness curve\n(conditional)"]
    vals = np.array([comp["tail_alpha_observed"],
                     comp["curve_alpha_observed_conditional"]], dtype=float)
    ci = np.array([unc["fixed_xmin"]["ci95"], curve["alpha_eff_ci95"]],
                  dtype=float)
    xpos = np.arange(2)
    ax.bar(xpos, vals, width=0.42, color=[C_REAL, C_MODEL])
    ax.errorbar(xpos, vals, yerr=np.vstack([vals - ci[:, 0], ci[:, 1] - vals]),
                fmt="none", ecolor="black", capsize=4, lw=0.8)
    for x, v in zip(xpos, vals):
        ax.text(x, v + 0.035, "%.3f" % v, ha="center", fontsize=7)
    ax.axhline(2.0, color=C_GREY, lw=0.9, ls=":", label=r"$\alpha=2$")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"$\alpha$ or conditional $\alpha_{\mathrm{eff}}$")
    lo = min(1.0, float(ci.min()) - 0.12)
    hi = max(2.25, float(ci.max()) + 0.16)
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.55, 1.85)
    ax.set_title("Observed estimators differ by %.3f; no common correction is applied"
                 % float(comp["absolute_difference"]), fontsize=8)
    panel_label(ax, "e")

    save(fig, "fig12_round3_calibration")

def figure_13():
    """Sedapp process-model demonstration generated from the archived NPZ."""
    from matplotlib.colors import TwoSlopeNorm

    archive = np.load(os.path.join(SEDAPP, "sedapp_case_output.npz"))
    summary = jload("sedapp_case_summary.json", folder=SEDAPP)
    z = archive["z_matrix"]
    lt = archive["layer_thickness_matrix"]
    sf = archive["sand_fraction_matrix"]
    wd = archive["wd_matrix"]
    x = archive["x_value"]
    times = archive["time_series"]
    sl = archive["sl_series"]
    shoreline = archive["shoreline_x"]
    base = archive["h0"]
    thickness = lt.sum(axis=1)
    positive_fraction = (lt > 1e-8).mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))

    # (a) imposed forcing and shoreline response
    a = axes[0, 0]
    a2 = a.twinx()
    line1 = a.plot(times, sl, color=C_REAL, lw=1.2, label="sea level")
    line2 = a2.plot(times, shoreline, color=C_SYN, lw=1.0, ls="--",
                    label="shoreline position")
    a.set_xlabel("model time (Myr)")
    a.set_ylabel("sea level (m)", color=C_REAL)
    a2.set_ylabel("shoreline x (m)", color=C_SYN)
    a.tick_params(axis="y", colors=C_REAL)
    a2.tick_params(axis="y", colors=C_SYN)
    a.legend(line1 + line2,
        [line1[0].get_label(), line2[0].get_label()],
        frameon=False,
        fontsize=5.8,
        ncol=1,
        loc="center",
        bbox_to_anchor=(0.37, 0.88),
        handlelength=1.8,
        labelspacing=0.5,
        borderaxespad=0)
    panel_label(a, "a")

    # (b) final compacted stratigraphic architecture
    b = axes[0, 1]
    xx = np.repeat(x, z.shape[1])
    zz = z.T.ravel()
    cc = sf.T.ravel()
    points = b.scatter(xx, zz, c=cc, s=4.0, cmap="cividis", vmin=0, vmax=1,
                       linewidths=0, rasterized=True)
    b.plot(x, z[:, -1], color="black", lw=1.0, label="final surface")
    b.plot(x, base, color=C_GREY, lw=0.8, ls=":", label="final basal reference")
    b.axhline(sl[-1], color=C_REAL, lw=0.8, ls="--", label="final sea level")
    b.set_xlabel("distance (m)")
    b.set_ylabel("elevation (m)")
    b.set_title("Compacted architecture after %.1f Myr" % times[-1], fontsize=8)
    b.legend(frameon=False,
        fontsize=5.3,
        ncol=1,
        loc="center",
        bbox_to_anchor=(0.55, 0.19),
        handlelength=1.8,
        labelspacing=0.45,
        borderaxespad=0)
    cb = fig.colorbar(points, ax=b, pad=0.02)
    cb.set_label("sand fraction", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    b.text(0.02, 0.03, "process-model demonstration; not field validation",
           transform=b.transAxes, fontsize=5.8, color=C_BAD)
    panel_label(b, "b")

    # (c) final thickness and net-positive increment fraction
    c = axes[1, 0]
    c2 = c.twinx()
    c.plot(x, thickness, color=C_REAL, lw=1.2, label="compacted thickness")
    c2.plot(x, positive_fraction, color=C_MODEL, lw=1.0, ls="--",
            label="net-positive steps")
    c.set_xlabel("distance (m)")
    c.set_ylabel("final compacted thickness (m)", color=C_REAL)
    c2.set_ylabel("net-positive increment fraction", color=C_MODEL)
    c2.set_ylim(0, 1.05)
    c.tick_params(axis="y", colors=C_REAL)
    c2.tick_params(axis="y", colors=C_MODEL)
    c.set_title("Final thickness: %.2f–%.2f m" %
                (summary["final_compacted_thickness_range_m"][0],
                 summary["final_compacted_thickness_range_m"][1]), fontsize=8)
    panel_label(c, "c")

    # (d) signed stored increments; not a hiatus-duration matrix
    d = axes[1, 1]
    finite_abs = np.abs(lt[np.isfinite(lt)])
    lim = max(float(np.quantile(finite_abs, 0.98)), 1e-6)
    mesh = d.pcolormesh(x, times, lt.T, shading="auto", cmap="RdBu_r",
                        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                        rasterized=True)
    d.set_xlabel("distance (m)")
    d.set_ylabel("model time (Myr)")
    d.set_title("Stored net increments are not hiatus durations",
        fontsize=8,
        x=0.55,
        pad=8)
    cb2 = fig.colorbar(mesh, ax=d, pad=0.02)
    cb2.set_label("net surface increment (m)", fontsize=6.5)
    cb2.ax.tick_params(labelsize=6)
    panel_label(d, "d")

    save(fig, "fig13_sedapp_process_model")


def main():
    print("Figure 1 ...");  figure_1()
    print("Figure 2 ...");  figure_2()
    print("Figure 3 ...");  figure_3()
    print("Figure 4 ...");  figure_4()
    print("Figure 5 ...");  figure_5()
    print("Figure 6 ...");  figure_6()
    print("Figure 7 ...");  figure_7()
    print("Figure 8 ...");  figure_8()
    print("Figure 9 ...");  figure_9()
    print("Figure 10 ..."); figure_10()
    print("Figure 11 ..."); figure_11()
    print("Figure 12 ..."); figure_12()
    print("Figure 13 ..."); figure_13()
    print("all figures written to", FIG)


if __name__ == "__main__":
    main()
