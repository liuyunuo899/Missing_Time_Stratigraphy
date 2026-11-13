# -*- coding: utf-8 -*-
"""Optional helper: extract text from a user-supplied source PDF.

The source PDF is not required to reproduce the numerical results. Pass its path
explicitly; no machine-specific download directory is assumed.
"""
import argparse
import pathlib
import sys

from pypdf import PdfReader

ROOT = pathlib.Path(__file__).resolve().parents[1]
DST = ROOT / "data" / "raw" / "conversation_pdf_text.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=pathlib.Path, help="path to the source PDF")
    parser.add_argument("--output", type=pathlib.Path, default=DST)
    args = parser.parse_args()
    target = args.pdf.expanduser().resolve()
    if not target.is_file():
        print("ERROR target pdf missing:", target)
        return 1
    print("INFO target:", target.name, target.stat().st_size)
    reader = PdfReader(str(target))
    n = len(reader.pages)
    print("INFO pages:", n)
    parts = []
    for i, page in enumerate(reader.pages):
        parts.append("===== PAGE %d =====" % (i + 1))
        try:
            parts.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - defensive
            print("WARN page %d extraction failed: %s" % (i + 1, exc))
            parts.append("")
    out = "\n".join(parts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(out, encoding="utf-8")
    print("INFO chars:", len(out))
    print("INFO written:", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
