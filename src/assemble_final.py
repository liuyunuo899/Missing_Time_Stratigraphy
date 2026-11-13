# -*- coding: utf-8 -*-
"""Assemble the authoritative Markdown manuscript and its editable Word version.

All mathematical source remains LaTeX in Markdown. Pandoc converts it to native
Word OMML; figures are embedded from the 300 dpi PNG files. The separate vector
PDF figures remain in figures/ and are not rasterised for that archive.
"""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
OUT_MD = ROOT / "manuscript_final.md"
OUT_DOCX = ROOT / "manuscript-final1.docx"
REFERENCE_DOC = Path("/app/.agents/plugins/kimi-word/skills/kimi-word/assets/reference.docx")
DOCX_TOOL = Path("/app/.agents/plugins/kimi-word/skills/kimi-word/scripts/docx.py")


def add_running_header_and_page_numbers(path):
    """Add submission-style running heads and native Word page fields."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    doc = Document(path)
    for section in doc.sections:
        header = section.header
        hp = header.paragraphs[0]
        hp.style = doc.styles["Normal"]
        hp.text = "MISSING TIME IN THE SEDIMENTARY RECORD"
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for run in hp.runs:
            run.font.size = Pt(8)
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.style = doc.styles["Normal"]
        fp.clear()
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fp.add_run("Page ")
        for field_name in ("PAGE", "NUMPAGES"):
            begin = OxmlElement("w:fldChar")
            begin.set(qn("w:fldCharType"), "begin")
            instruction = OxmlElement("w:instrText")
            instruction.set(qn("xml:space"), "preserve")
            instruction.text = f" {field_name} "
            separate = OxmlElement("w:fldChar")
            separate.set(qn("w:fldCharType"), "separate")
            value = OxmlElement("w:t")
            value.text = "1"
            end = OxmlElement("w:fldChar")
            end.set(qn("w:fldCharType"), "end")
            run = OxmlElement("w:r")
            for element in (begin, instruction, separate, value, end):
                run.append(element)
            fp._p.append(run)
            if field_name == "PAGE":
                fp.add_run(" of ")
        for run in fp.runs:
            run.font.size = Pt(9)
    doc.save(path)


ORDER = [
    "00_front.md", "01_introduction.md", "02_scope.md", "03_concepts.md",
    "04_theory.md", "05_synthesis.md", "06_variational.md", "07_case_real.md",
    "08_case_synthetic.md", "09_integration.md", "10_limitations.md",
    "11_future.md", "12_conclusions.md", "14_references.md", "15_back_matter.md",
]


def main():
    parts = [(MS / name).read_text(encoding="utf-8").strip() for name in ORDER]
    full = "\n\n".join(parts) + "\n"
    figures = [int(x) for x in re.findall(r"!\[\*\*Figure (\d+)\.", full)]
    if figures != list(range(1, 14)):
        raise RuntimeError(f"Figure sequence is not 1–13: {figures}")
    for rel in re.findall(r"!\[[^\n]*?\]\(([^)]+)\)", full):
        if not (ROOT / rel).is_file():
            raise FileNotFoundError(ROOT / rel)
    forbidden = [r"\bTODO\b", r"\bTBD\b", r"\[\^\d+\^\]", r"tautochron", r"a explicit"]
    for pattern in forbidden:
        if re.search(pattern, full, flags=re.IGNORECASE):
            raise RuntimeError(f"Forbidden final-manuscript residue: {pattern}")
    OUT_MD.write_text(full, encoding="utf-8")
    print(f"wrote {OUT_MD} ({len(full.split())} whitespace-delimited tokens)")
    command = [
        "pandoc", str(OUT_MD), "-o", str(OUT_DOCX),
        "--from=markdown", "--to=docx", "--standalone",
        f"--resource-path={ROOT}",
        f"--reference-doc={REFERENCE_DOC}",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"wrote {OUT_DOCX} with Pandoc LaTeX-to-OMML conversion")
    add_running_header_and_page_numbers(OUT_DOCX)
    from normalize_docx_omml import normalise
    print(f"normalised {normalise(OUT_DOCX)} OMML propertie(s)")
    subprocess.run(["python3", str(DOCX_TOOL), "validate", str(OUT_DOCX)], check=True)


if __name__ == "__main__":
    main()
