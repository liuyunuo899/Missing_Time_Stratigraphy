# -*- coding: utf-8 -*-
"""Generate the top-level figure_captions.md deliverable.

Reads the captions from manuscript/13_figure_captions.md (the single source of
truth) and joins each with the figure's PNG/PDF filenames and a short title
(the caption's first sentence). Run:  python paper_project/src/build_figure_captions.py
"""
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "manuscript", "13_figure_captions.md")
FIGDIR = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "figure_captions.md")

SHORT = {
    1: "Missing time has one measurement under three names, and it is resolution-dependent by construction.",
    2: "The two cases answer different questions, and every claim is graded by the evidence that carries it.",
    3: "The observational sample is large but geographically and temporally structured.",
    4: "Hiatus durations are heavy tailed but a pooled power law is rejected, and the rejection is spatial.",
    5: "The completeness relation is exact, and the rate-span decline lives inside columns.",
    6: "The duration statistics are robust to the compilation's age conventions, but the count of hiatuses is not.",
    7: "The synthetic generator has an exact budget.",
    8: "The estimator is honest, the mechanisms are not distinguishable, and removal steepens the apparent tail.",
    9: "The variational inverse is limited by bias, not by non-uniqueness.",
    10: "The generator reproduces where the real record sits, and the preservation error reverses sign.",
    11: "What can be claimed, what is bounded, and what to do next.",
    12: "The pipeline is calibrated, the prior is not the cause of the bias, and the two estimators reconcile after correction.",
}


def main():
    txt = open(SRC, encoding="utf-8").read()
    caps = {}
    for m in re.finditer(r"\*\*Figure (\d+)\.\*\*(.*?)(?=\n\n\*\*Figure |\Z)",
                         txt, re.S):
        n = int(m.group(1))
        caps[n] = " ".join(m.group(2).split()).rstrip()

    def stem(n):
        png = glob.glob(os.path.join(FIGDIR, "fig%02d_*.png" % n))
        return os.path.basename(png[0]).replace(".png", "") if png else None

    lines = []
    lines.append("# Figure captions")
    lines.append("")
    lines.append("Standalone list of the twelve figures. Captions state the "
                 "message of each figure, not merely its content; panel letters "
                 "are shown in every figure. The same captions appear in "
                 "`manuscript/13_figure_captions.md` (the source of truth; this "
                 "file is regenerated from it by `src/build_figure_captions.py`). "
                 "Files are in `figures/`, each as 300 dpi PNG and vector PDF.")
    lines.append("")
    lines.append("| # | File stem | Short title |")
    lines.append("|---|---|---|")
    for n in sorted(caps):
        s = stem(n) or "**missing**"
        lines.append("| %d | `%s` (.png / .pdf) | %s |" % (n, s, SHORT.get(n, "")))
    lines.append("")
    lines.append("---")
    lines.append("")
    for n in sorted(caps):
        s = stem(n) or "MISSING"
        lines.append("## Figure %d — `%s`" % (n, s))
        lines.append("")
        lines.append("**Figure %d.** %s" % (n, caps[n]))
        lines.append("")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("wrote", OUT, "- %d figures" % len(caps))


if __name__ == "__main__":
    main()
