# -*- coding: utf-8 -*-
"""Third batch: recent literature (2020-2026) and the Macrostrat platform papers."""
import json
import pathlib
import urllib.parse
import urllib.request
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
HITS = ROOT / "references" / "raw_hits"
HITS.mkdir(parents=True, exist_ok=True)
MAILTO = "earth.science.reviews.repro@example.org"
UA = {"User-Agent": "ESR-review-repro/1.0 (mailto:%s)" % MAILTO}

TOPICS = [
    "stratigraphic completeness",
    "hiatus duration distribution",
    "sediment accumulation rate timespan bias",
    "unconformity time missing records",
    "sediment preservation catchment storage recycling",
    "stratigraphic forward model inverse calibration",
    "Macrostrat database",
    "age-depth model uncertainty hiatus",
    "accommodation generation sediment supply sequences numerical model",
    "power law tail exponent estimation bias finite sample",
]


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slug(t):
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in t.lower()]
    return "".join(keep)[:60]


def crossref_recent(q, since="2020-01-01"):
    url = ("https://api.crossref.org/works?rows=15&sort=relevance&mailto=%s"
           "&filter=from-pub-date:%s&query.bibliographic=%s"
           % (MAILTO, since, urllib.parse.quote(q)))
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as fh:
        data = json.load(fh)
    return url, data.get("message", {}).get("items", [])


def openalex_recent(q, since="2020-01-01"):
    url = ("https://api.openalex.org/works?per-page=20&mailto=%s&sort=relevance_score:desc"
           "&filter=from_publication_date:%s&search=%s" % (MAILTO, since, urllib.parse.quote(q)))
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as fh:
        data = json.load(fh)
    return url, data.get("results", [])


def print_cr(items):
    for it in items[:8]:
        yr = ""
        iss = it.get("issued", {}).get("date-parts", [[None]])
        if iss and iss[0]:
            yr = iss[0][0]
        print("   CR [%s] %s -- %s | %s | %s"
              % (yr, (it.get("title") or ["?"])[0][:66],
                 (it.get("container-title") or ["?"])[0][:30],
                 (it.get("author") or [{}])[0].get("family", "?"), it.get("DOI")))


def print_oa(items):
    for it in items[:8]:
        auth = (it.get("authorships") or [{}])[0].get("author", {}).get("display_name", "?")
        loc = (it.get("primary_location") or {}).get("source", {}) or {}
        print("   OA [%s] %s -- %s | %s | %s"
              % (it.get("publication_year"), (it.get("title") or "?")[:66],
                 (loc.get("display_name") or "?")[:30], auth, it.get("doi")))


def main():
    for i, q in enumerate(TOPICS, 1):
        print("=" * 78)
        print("TOPIC %02d: %s" % (i, q))
        try:
            url, items = crossref_recent(q)
            rec = {"source": "crossref_recent", "query": q, "url": url,
                   "retrieved_utc": utcnow(), "items": items}
            (HITS / ("N%02d_%s__crossref_recent.json" % (i, slug(q)))).write_text(
                json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
            print_cr(items)
        except Exception as exc:  # noqa: BLE001
            print("   crossref FAIL", exc)
        try:
            url, items = openalex_recent(q)
            rec = {"source": "openalex_recent", "query": q, "url": url,
                   "retrieved_utc": utcnow(), "results": items}
            (HITS / ("N%02d_%s__openalex_recent.json" % (i, slug(q)))).write_text(
                json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
            print_oa(items)
        except Exception as exc:  # noqa: BLE001
            print("   openalex FAIL", exc)


if __name__ == "__main__":
    main()
