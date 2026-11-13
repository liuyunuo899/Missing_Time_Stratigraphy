# -*- coding: utf-8 -*-
"""Recalculate and verify the literature workbook with the Excel QA gate."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TOOL = Path("/app/.agents/plugins/kimi-excel/skills/kimi-excel/scripts/xlsx_tools.py")
BOOK = ROOT / "literature_review.xlsx"


def main():
    for command in ("recalc", "verify"):
        result = subprocess.run(["python3", str(TOOL), command, str(BOOK)],
                                check=True, capture_output=True, text=True)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")
        if command == "verify":
            payload = json.loads(result.stdout)
            if payload.get("status") != "pass":
                raise RuntimeError(f"Literature workbook verification did not pass: {payload}")


if __name__ == "__main__":
    main()
