# -*- coding: utf-8 -*-
"""Probe bulk-download options for the Macrostrat v2 API (real public-data source)."""
import json
import pathlib
import urllib.request

BASE = "https://macrostrat.org/api/v2"
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)


def head(url):
    req = urllib.request.Request(url, headers={"User-Agent": "earth-science-reviews-repro/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as fh:
            raw = fh.read()
        return fh.status, fh.headers.get("Content-Type", ""), len(raw), raw[:400]
    except Exception as exc:
        return "ERR", type(exc).__name__, 0, str(exc)[:300]


PROBES = [
    BASE + "/units?project_id=1&format=csv",
    BASE + "/units?all&format=csv",
    BASE + "/columns?project_id=1&format=csv",
    BASE + "/columns?all&format=csv",
    BASE + "/sections?project_id=1&format=csv",
]

for url in PROBES:
    status, ctype, n, snippet = head(url)
    print("=" * 70)
    print("URL:", url)
    print("STATUS:", status, "| CONTENT-TYPE:", ctype, "| BYTES:", n)
    print("HEAD:", snippet[:220])
