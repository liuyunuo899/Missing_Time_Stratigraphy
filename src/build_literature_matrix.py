# -*- coding: utf-8 -*-
"""Build literature_review.csv / .xlsx: the theme-evidence matrix.

One row per verified reference. Theme, role, evidence summary and the section
in which the work is used are assigned by hand from the manuscript; the
bibliographic fields come from references/verified_references.csv, so no
bibliographic detail is invented here.
"""
import os
import re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# label-prefix -> (theme, method_class, evidence/role in this review, cited)
MAP = {
 "Sadler 1981": ("rate-span scaling", "empirical compilation",
   "Decline of measured accumulation rate with averaging span; the phenomenon this review decomposes", True),
 "Sadler & Strauss 1990": ("completeness", "empirical method",
   "Completeness defined relative to a stated resolution; the convention adopted here", True),
 "Anders et al. 1987": ("rate-span scaling", "empirical compilation",
   "Re-examination of rate-span decline across environments", True),
 "Strauss & Sadler 1989": ("completeness", "stochastic model",
   "Stochastic models linking completeness and intermittency; background to the random-phase relation", True),
 "Tipper 1987": ("completeness", "method critique",
   "Estimator for stratigraphic completeness; the counting convention used", True),
 "Tipper 2016": ("rate-span scaling", "method critique",
   "Measured rates estimate an ill-specified quantity; motivation for the within-column decomposition", True),
 "Plotnick 1986": ("hiatus scaling", "model proposal",
   "Fractal hiatus model; the hypothesis tested in Section 7.3", True),
 "Plotnick 1988": ("hiatus scaling", "reply",
   "Published exchange on whether the data could distinguish a fractal from other heavy tails", True),
 "Plotnick 2017": ("hiatus scaling", "synthesis",
   "Later return to hierarchical patterns in the record", True),
 "Schumer & Jerolmack 2009": ("rate-span scaling", "stochastic model",
   "Hiatus statistics and apparent rate change; motivates separating within-column and pooled rate–span relations", True),
 "Jerolmack & Sadler 2007": ("rate-span scaling", "empirical + model",
   "Transience versus persistence; source of the among-setting explanation tested in Sec. 7.5", True),
 "Jerolmack & Paola 2010": ("signal shredding", "theory + experiment",
   "Transport filters forcing below a diffusive threshold; the process-school claim not tested here", True),
 "Paola et al. 2018": ("preservation", "synthesis",
   "Time not our time: physical framing of what the record preserves", True),
 "Paola et al. 2009": ("process stratigraphy", "experiment",
   "Effectiveness of experiments; supports reading bed statistics physically", True),
 "Straub et al. 2012": ("process stratigraphy", "experiment",
   "Bed thickness closer to exponential than power law; argues against reading tails as self-organisation", True),
 "Wilkinson et al. 2009": ("global compilation", "empirical",
   "Rock-cycling rates from area-age distributions", True),
 "Gombosi et al. 2012": ("global compilation", "empirical",
   "Tectonic diffusion of lithosomes; extends the rock-cycling logic", True),
 "Peters & Heim 2010": ("global compilation", "empirical",
   "Macrostratigraphic completeness and palaeontological sampling", True),
 "Heim & Peters 2010": ("global compilation", "empirical",
   "Covariation of macrostratigraphic and macroevolutionary patterns", True),
 "Peters 2021": ("global compilation", "empirical",
   "Igneous area-age relations in continental crust", True),
 "Tasistro-Hart et al. 2023": ("global compilation", "empirical",
   "Phanerozoic flooding and the Great Unconformity measured from Macrostrat", True),
 "Jones et al. 2024": ("global compilation", "software/data",
   "rmacrostrat programmatic access; related software documented alongside the Python API downloader used here", True),
 "Quinn et al. 2023": ("global compilation", "database",
   "Macrostrat construction; the dataset of Case Study I", True),
 "Wilkinson et al. 2023": ("rate-span scaling", "empirical",
   "Sadler effect biases in Holocene palaeofire records; a concrete consequence of the decline", True),
 "Kemp et al. 2020": ("non-stationarity", "empirical",
   "Human-driven change in erosion, transfer, and storage; context for non-stationary preservation budgets", True),
 "Molnar 2004": ("non-stationarity", "empirical",
   "Late Cenozoic increase in accumulation rates; candidate (iv) of Sec. 4.4", True),
 "Clauset et al. 2009": ("statistics", "method",
   "Tail estimation, xmin selection and bootstrap goodness-of-fit used throughout", True),
 "Virkar & Clauset 2014": ("statistics", "method",
   "Binned variants of tail fitting; supports the caution on estimator choice", True),
 "Vuong 1989": ("statistics", "method",
   "Likelihood-ratio model selection; reported as secondary to absolute fit", True),
 "Pilgrim": ("statistics", "method",
   "Bias in tail-exponent estimators; supports the validation in Sec. 8.3", True),
 "Alstott et al. 2014": ("statistics", "software",
   "Reference implementation of the power-law workflow; conceptual background", False),
 "Holbrook 2012": ("terminology", "synthesis",
   "Reappraisal of the sequence boundary in time and space; supports separating observations from terminology", True),
 "Burgess 2016": ("modelling", "perspective",
   "The future of sequence stratigraphy; the third dimension problem", True),
 "Burgess 2010": ("modelling", "comment",
   "Eustatic amplitude and shelf morphology; context for accommodation control", False),
 "Bornholdt 1999": ("inverse modelling", "method",
   "Inverse stratigraphic modelling with genetic algorithms; compared in Sec. 6.3", True),
 "Granjeon 2023": ("inverse modelling", "application",
   "Forward and inverse modelling in exploration; compared in Sec. 6.3", True),
 "Griffiths 2022": ("modelling", "review",
   "What forward and inverse stratigraphic models can deliver", True),
 "Patani et al. 2020": ("inverse modelling", "method",
   "Stochastic inverse modelling and parametric uncertainty; compared in Sec. 6.3", True),
 "Blaauw & Christen 2011": ("age-depth", "method",
   "Bacon gamma-process age-depth models; the neighbouring formulation of Sec. 6.3", True),
 "Haslett & Parnell 2008": ("age-depth", "method",
   "Monotone process for radiocarbon chronologies; the other age-depth route", True),
 "Liu et al. 2020": ("completeness", "process model",
   "Process-based completeness for carbonate platforms", True),
 "Shahkarami et al. 2020": ("completeness", "application",
   "Completeness at the Ediacaran-Cambrian boundary", False),
 "Yan et al. 2023": ("preservation", "empirical",
   "Preservation and accretion in fluvial meander belts", False),
 "Colombera et al. 2020": ("process stratigraphy", "meta-analysis",
   "Accommodation and sediment-supply controls on parasequences", False),
 "Green et al. 2022": ("missing section", "thermochronology",
   "Kilometre-scale burial and exhumation; detects removal stratigraphy cannot", True),
 "Hart 2024": ("missing section", "geochemistry",
   "Geochemical detection of breaks in a single basin", True),
 "Sadler 1994": ("process stratigraphy", "theory",
   "Expected duration of peritidal cycles; background to completeness", False),
 "Partin & Sadler 2016": ("non-stationarity", "empirical",
   "Snowball Earth accumulation; secular change in the record", False),
 "Voepel et al. 2013": ("preservation", "theory",
   "Residence-time distributions; the formalism behind the preservation discussion", True),
 "Salles et al. 2018": ("modelling", "software",
   "pyBadlands landscape model; the transport-based generator of Priority 3", False),
 "Salles 2016": ("modelling", "software",
   "Badlands basin and landscape model", False),
 "Miall 2014": ("preservation", "synthesis",
   "Stratigraphy as frozen accidents; the process reframing", True),
 "Dacey 1970": ("statistics", "method",
   "Markovian models in stratigraphic analysis; early formal treatment", False),
 "Hagen 2024": ("rate-span scaling", "perspective",
   "Accumulation rates and high-resolution age-depth models", True),
 "Wang et al. 2025": ("process stratigraphy", "empirical",
   "Alluvial response to increasing and variable supply", False),
 "Basquin et al. 2026": ("process stratigraphy", "empirical",
   "Accommodation versus sediment supply in estuaries", False),
 "Dosch et al. 2026": ("signal shredding", "empirical",
   "Fluvial transit and storage effects on signal preservation", False),
 "Schmelz et al. 2024": ("modelling", "numerical",
   "Sensitivity of modelled passive-margin stratigraphy", False),
 "Vaucher 2024": ("process stratigraphy", "empirical",
   "Sediment-supply control on delta sequences", False),
 "Li et al. 2021": ("modelling", "software",
   "Sedapp nonlinear-diffusion stratigraphic forward model; the process-model example executed in Section 8.7", True),
}


EXCLUSION_REASONS = {
    "10.1130/g38828.1": "Verified during backward tracing; not needed for the final fossil-locality-independent argument.",
    "10.1130/g49037.1": "Verified during compilation review; igneous area–age relations are outside the final sedimentary-record scope.",
    "10.1130/focus042016.1": "Verified perspective; excluded to avoid unsupported terminology and nonessential priority claims.",
    "10.1130/g30693c.1": "Verified comment; background only and not directly used in a final scientific claim.",
    "10.2110/sepmmisc.24.093": "Verified regional example; excluded because independent removal evidence is represented by Green et al. (2022).",
}


def cited_sections(doi, first_author, year, section_texts):
    surname = str(first_author).strip().split()[-1].casefold()
    year = str(int(year))
    hits = []
    for section, text in section_texts:
        for match in re.finditer(re.escape(surname), text.casefold()):
            if year in text[max(0, match.start() - 30):match.start() + 220]:
                hits.append(section)
                break
    return "; ".join(hits)


def main():
    import json
    import re
    from pathlib import Path
    from openpyxl import Workbook
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    df = pd.read_csv(os.path.join(ROOT, "references", "verified_references.csv"))
    keep = set(json.loads(Path(ROOT, "references", "final_reference_dois.json").read_text(encoding="utf-8")))
    section_texts = []
    for path in sorted(Path(ROOT, "manuscript").glob("*.md")):
        if path.name in ("14_references.md", "15_back_matter.md"):
            continue
        text = path.read_text(encoding="utf-8")
        heading = next((line[3:] for line in text.splitlines() if line.startswith("## ")), path.stem)
        section_texts.append((heading, text))
    rows = []
    for _, r in df.iterrows():
        label = str(r["label"])
        key = next((pref for pref in MAP if label.startswith(pref)), None)
        if key is None:
            key = next((p for p in MAP if p in label), None)
        if key is None:
            raise KeyError(f"No evidence role assigned for {label}")
        theme, method, evidence, _ = MAP[key]
        doi = str(r["doi"])
        cited = doi in keep
        sections = cited_sections(doi, r["crossref_first_author"], r["year"], section_texts) if cited else ""
        if cited and not sections:
            raise RuntimeError(f"Final reference has no detected manuscript use: {label}")
        rows.append({
            "Theme": theme,
            "Citation": "%s (%s)" % (str(r["crossref_first_author"]), int(r["year"])),
            "Title": r["title"], "Year": int(r["year"]), "Container": r["container"],
            "DOI": doi, "Verification Status": r["status"],
            "Method Class": method, "Role in Review": evidence,
            "Cited in Manuscript": "yes" if cited else "no",
            "Cited Sections": sections,
            "Exclusion or Non-citation Reason": "" if cited else EXCLUSION_REASONS[doi],
            "Source Name": "Crossref DOI metadata",
            "Source URL": "https://doi.org/" + doi.replace("<", "%3C").replace(">", "%3E"),
        })
    out = pd.DataFrame(rows).sort_values(["Cited in Manuscript", "Theme", "Year", "Citation"], ascending=[False, True, True, True])
    if len(out) != 60 or (out["Cited in Manuscript"] == "yes").sum() != 55:
        raise RuntimeError("Literature audit must contain 60 records with 55 cited references")
    out.to_csv(os.path.join(ROOT, "literature_review.csv"), index=False)

    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"
    summary.sheet_view.showGridLines = False
    summary["B2"] = "Literature verification and use audit"
    summary["B2"].font = Font(size=16, bold=True, color="1F4E5F")
    summary["B4"] = "Metric"; summary["C4"] = "Count"
    summary["B5"] = "Verified records"; summary["C5"] = "=COUNTA('Literature Audit'!A2:A61)"
    summary["B6"] = "Cited in final manuscript"; summary["C6"] = '=COUNTIF(\'Literature Audit\'!J2:J61,"yes")'
    summary["B7"] = "Verified but not cited"; summary["C7"] = '=COUNTIF(\'Literature Audit\'!J2:J61,"no")'
    summary["B9"] = "Scope note"
    summary["C9"] = "Structured critical review; metadata verification is not systematic-review completeness certification."
    summary["B10"] = "Evidence grades"
    summary["C10"] = "D derived; O observational; M model-conditional; L literature-based; U unresolved."
    summary.column_dimensions["B"].width = 30
    summary.column_dimensions["C"].width = 95
    for row in summary.iter_rows(min_row=4, max_row=10, min_col=2, max_col=3):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    for cell in summary[4]:
        if cell.column in (2, 3):
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E5F")

    ws = wb.create_sheet("Literature Audit")
    ws.append(list(out.columns))
    for row in out.itertuples(index=False, name=None):
        ws.append(list(row))
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.sheet_view.showGridLines = False
    widths = [23, 24, 62, 10, 34, 30, 20, 24, 68, 20, 40, 68, 26, 62]
    for i, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = width
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E5F")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[1].height = 34
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.conditional_formatting.add("A2:N61", FormulaRule(formula=['$J2="no"'], fill=PatternFill("solid", fgColor="F2F2F2")))
    table = Table(displayName="LiteratureAudit", ref="A1:N61")
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True, showFirstColumn=False, showLastColumn=False)
    ws.add_table(table)
    wb.calculation.fullCalcOnLoad = True
    wb.calculation.forceFullCalc = True
    wb.calculation.calcMode = "auto"
    wb.save(os.path.join(ROOT, "literature_review.xlsx"))
    print("rows:", len(out), "| cited:", (out["Cited in Manuscript"] == "yes").sum(), "| excluded:", (out["Cited in Manuscript"] == "no").sum())
    print(out.groupby("Theme").size().sort_values(ascending=False))


if __name__ == "__main__":
    main()
