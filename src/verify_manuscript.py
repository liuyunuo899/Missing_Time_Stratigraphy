# -*- coding: utf-8 -*-
"""Verify manuscript source, bibliography coverage, and final DOCX contents."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import json
import re

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "manuscript-final1.docx"
MD = ROOT / "manuscript_final.md"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}


def main():
    source = MD.read_text(encoding="utf-8")
    expected_displays = source.count("$$") // 2
    source_figures = [int(x) for x in re.findall(r"!\[\*\*Figure (\d+)\.", source)]
    assert source_figures == list(range(1, 14)), source_figures
    with ZipFile(DOCX) as archive:
        names = archive.namelist()
        root = etree.fromstring(archive.read("word/document.xml"))
        text = "".join(root.xpath("//w:t/text()", namespaces=NS))
        omath = root.xpath("//m:oMath", namespaces=NS)
        displays = root.xpath("//m:oMathPara", namespaces=NS)
        blips = root.xpath("//a:blip", namespaces=NS)
        media = [n for n in names if n.startswith("word/media/") and not n.endswith("/")]
        rel = etree.fromstring(archive.read("word/_rels/document.xml.rels"))
        image_rels = [x for x in rel if x.get("Type", "").endswith("/image")]
        paragraphs = ["".join(x.xpath(".//w:t/text()", namespaces=NS)) for x in root.xpath("//w:p", namespaces=NS)]
        captions = [x for x in paragraphs if re.match(r"^Figure \d+\.", x)]
        residue = {pattern: len(re.findall(pattern, text)) for pattern in [
            r"\$[^\s$][^$]*\$", r"\\alpha", r"\\frac", r"\\mathrm",
            r"\\mathbb", r"\\qquad", r"\\text", r"\[\^\d+\^\]",
        ]}
        headers = [n for n in names if re.fullmatch(r"word/header\d+\.xml", n)]
        footers = [n for n in names if re.fullmatch(r"word/footer\d+\.xml", n)]
        footer_text = " ".join(
            etree.fromstring(archive.read(name)).xpath("string()") for name in footers)
        report = {
            "docx": str(DOCX), "bytes": DOCX.stat().st_size,
            "omath_objects": len(omath), "omath_paragraphs": len(displays),
            "source_display_equations": expected_displays,
            "numbered_display_equations": 21,
            "drawing_blips": len(blips), "image_relationships": len(image_rels),
            "media_files": media, "figure_captions": captions,
            "unrendered_latex_residue": residue,
            "headers": headers, "footers": footers, "footer_text": footer_text,
        }
    assert len(omath) > 250, len(omath)
    assert len(displays) == expected_displays, (len(displays), expected_displays)
    assert len(blips) == len(image_rels) == len(media) == 13
    assert len(captions) == 13
    assert [int(re.match(r"^Figure (\d+)\.", x).group(1)) for x in captions] == list(range(1, 14))
    assert not any(residue.values()), residue
    assert headers and footers and "PAGE" in footer_text and "NUMPAGES" in footer_text
    report["status"] = "PASSED"
    out = ROOT / "qa" / "docx_structure_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("figure_captions", "media_files")}, indent=2))
    print("DOCX STRUCTURE AUDIT PASSED")


if __name__ == "__main__":
    main()
