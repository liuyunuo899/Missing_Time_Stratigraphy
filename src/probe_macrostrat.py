# -*- coding: utf-8 -*-
"""Probe the Macrostrat v2 API structure to plan the real-data download."""
import json
import pathlib
import urllib.request

BASE = "https://macrostrat.org/api/v2"
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "earth-science-reviews-repro/1.0"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        return json.load(fh)


def main():
    unit = get(BASE + "/units?limit=1&response=long")
    data = unit.get("success", {}).get("data", [])
    keys = sorted(data[0].keys()) if data else []
    print("UNIT FIELDS (%d):" % len(keys))
    print(", ".join(keys))
    print("SAMPLE RECORD:")
    print(json.dumps(data[0], indent=1)[:3000])
    (OUT / "probe_unit.json").write_text(json.dumps(unit, indent=1), encoding="utf-8")

    cols = get(BASE + "/columns?lat=39&lng=-98&adjacent=true")
    cdata = cols.get("success", {}).get("data", [])
    print("\nCOLUMN COUNT:", len(cdata))
    if cdata:
        print("COLUMN FIELDS:", ", ".join(sorted(cdata[0].keys())))
        print("SAMPLE COLUMN:", json.dumps(cdata[0], indent=1)[:1200])
    (OUT / "probe_columns.json").write_text(json.dumps(cols, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
