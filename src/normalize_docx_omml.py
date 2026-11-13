# -*- coding: utf-8 -*-
"""Normalise schema ordering in Pandoc/texmath OMML without changing its text.

The bundled md2docx utility handles run and matrix properties. This project also
uses delimiters whose property order differs from the strict CT_DPr sequence
(begChr, sepChr, endChr, grow, shp, ctrlPr). All edits are mechanical OOXML
property operations in the deliverable, never edits to the Word skill itself.
"""
from pathlib import Path
import importlib.util
import os
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree

SKILL_UTIL = Path("/app/.agents/plugins/kimi-word/skills/kimi-word/scripts/md2docx/docx_utils.py")
MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
DPR_ORDER = {"begChr": 0, "sepChr": 1, "endChr": 2, "grow": 3, "shp": 4, "ctrlPr": 5}


def _load_shared_normaliser():
    spec = importlib.util.spec_from_file_location("wb_docx_utils", SKILL_UTIL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_omml


def normalise(path):
    path = Path(path).resolve()
    fixed = _load_shared_normaliser()(str(path))
    with ZipFile(path, "r") as zin:
        entries = [(item, zin.read(item.filename)) for item in zin.infolist()]
    qn = lambda tag: f"{{{MATH_NS}}}{tag}"
    for i, (item, data) in enumerate(entries):
        if item.filename not in ("word/document.xml", "word/footnotes.xml", "word/endnotes.xml"):
            continue
        root = etree.fromstring(data)
        part_fixed = 0
        for props in root.iter(qn("dPr")):
            children = list(props)
            ordered = sorted(enumerate(children), key=lambda ic: (
                DPR_ORDER.get(etree.QName(ic[1]).localname, 99), ic[0]))
            new_children = [child for _, child in ordered]
            if [etree.QName(c).localname for c in new_children] != [etree.QName(c).localname for c in children]:
                for child in children:
                    props.remove(child)
                for child in new_children:
                    props.append(child)
                part_fixed += 1
        if part_fixed:
            entries[i] = (item, etree.tostring(root, encoding="utf-8", xml_declaration=True, standalone=True))
            fixed += part_fixed
    if fixed:
        fd, tmp = tempfile.mkstemp(suffix=".docx", dir=path.parent)
        os.close(fd)
        try:
            with ZipFile(tmp, "w", ZIP_DEFLATED) as zout:
                for item, data in entries:
                    zout.writestr(item, data)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return fixed


if __name__ == "__main__":
    import sys
    print(f"normalised {normalise(sys.argv[1])} OMML propertie(s)")
