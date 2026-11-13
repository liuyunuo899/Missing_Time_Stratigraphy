# -*- coding: utf-8 -*-
"""Case Study I: download the Macrostrat public stratigraphic dataset.

Macrostrat v2 REST API, licence CC-BY 4.0 (stated inside every API payload:
"success.license": "CC-BY 4.0").  Nothing is modified on download; raw CSVs are
checksummed and stored with a provenance record.

Run:
    python paper_project/src/download_macrostrat.py
"""
import hashlib
import json
import pathlib
import time
import urllib.request
from datetime import datetime, timezone

BASE = "https://macrostrat.org/api/v2"
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "ESR-review-repro/1.0 (mailto:earth.science.reviews.repro@example.org)"}

ENDPOINTS = {
    "macrostrat_units_all.csv": BASE + "/units?all&format=csv",
    "macrostrat_columns_all.csv": BASE + "/columns?all&format=csv",
    "macrostrat_sections_all.csv": BASE + "/sections?all&format=csv",
    "macrostrat_intervals_all.csv": BASE + "/defs/intervals?all&format=csv",
    "macrostrat_lithologies_all.csv": BASE + "/defs/lithologies?all&format=csv",
    "macrostrat_environments_all.csv": BASE + "/defs/environments?all&format=csv",
    "macrostrat_unit_boundaries_all.csv": BASE + "/units?all&format=csv&response=long",
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name, url, tries=4):
    dest = RAW / name
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=300) as fh:
                payload = fh.read()
                status = fh.status
            dest.write_bytes(payload)
            return {
                "file": str(dest),
                "url": url,
                "http_status": status,
                "bytes": len(payload),
                "sha256": sha256_of(dest),
                "download_seconds": round(time.time() - t0, 2),
                "downloaded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "attempts": k + 1,
            }, None
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3 * (k + 1))
    return None, "%s: %s" % (type(last).__name__, last)


def main():
    prov = {
        "dataset": "Macrostrat v2 REST API (units, columns, sections, geologic-time intervals, lithologies, environments)",
        "provider": "Macrostrat / University of Wisconsin-Madison",
        "landing_page": "https://macrostrat.org",
        "api_documentation": "https://macrostrat.org/docs/v2",
        "license": "CC-BY 4.0 (returned by the API in success.license)",
        "polite_use_note": "Macrostrat requests that bulk API users identify themselves and cite the project.",
        "access_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": [],
    }
    for name, url in ENDPOINTS.items():
        print("downloading", name, flush=True)
        rec, err = download(name, url)
        if err:
            print("  FAILED:", err)
            prov["files"].append({"file": name, "url": url, "status": "FAILED", "error": err})
            continue
        nlines = sum(1 for _ in open(RAW / name, "rb")) - 1
        rec["data_rows_excl_header"] = nlines
        prov["files"].append(rec)
        print("  ok %d bytes, %d rows, sha256=%s..." % (rec["bytes"], nlines, rec["sha256"][:16]))
    (RAW / "_macrostrat_provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print("wrote provenance ->", RAW / "_macrostrat_provenance.json")


if __name__ == "__main__":
    main()
