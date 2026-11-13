# -*- coding: utf-8 -*-
"""Targeted title-based searches in Crossref for canonical works.

Fix of literature_search.py: an empty ``query.author=`` parameter previously
emptied every result set. Stage 3 support script.
"""
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

TITLE_QUERIES = [
    ("sediment accumulation rates completeness of stratigraphic sections", "Sadler"),
    ("estimation of completeness of stratigraphical sections using empirical data and theoretical models", "Sadler"),
    ("a new look at sedimentation rates and the completeness of the stratigraphic record", "Anders"),
    ("fractal model for the distribution of stratigraphic hiatuses", "Plotnick"),
    ("fractal structure of the stratigraphic record", "Rothman"),
    ("transience and persistence in the depositional record of continental margins", "Jerolmack"),
    ("shredding of environmental signals by sediment transport", "Jerolmack"),
    ("unreasonable effectiveness of stratigraphic and geomorphic experiments", "Paola"),
    ("real and apparent changes in sediment deposition rates through time", "Schumer"),
    ("measured rates of sediment accumulation a formal reconsideration", "Tipper"),
    ("global geologic maps are tectonic speedometers rates of rock cycling from area age frequencies", "Wilkinson"),
    ("geological completeness of paleontological sampling in North America", "Peters"),
    ("power-law distributions in empirical data", "Clauset"),
    ("pyBadlands a framework to simulate sediment transport landscape dynamics and basin stratigraphic evolution", "Salles"),
    ("updating uniformitarianism stratigraphy as just a set of sedimentary processes", "Miall"),
    ("influence of hiatuses on sediment accumulation rates", "Sadler"),
    ("expected duration of upward reaching hiatuses", "Sadler"),
    ("quantifying biostratigraphic and chronostratigraphic uncertainty", "Sadler"),
    ("random walk model for the spatial distribution of the ages of paralic to deep marine deposits", "Sadler"),
    ("Two-stage stochastic model of the stratigraphic record", "Sadler"),
]


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slug(t):
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in t.lower()]
    return "".join(keep)[:70]


def search(title, author):
    q = urllib.parse.quote(title)
    a = urllib.parse.quote(author)
    url = ("https://api.crossref.org/works?rows=5"
           "&select=DOI,title,subtitle,author,container-title,issued,volume,issue,page,type,publisher,URL"
           "&mailto=%s&query.bibliographic=%s&query.author=%s" % (MAILTO, q, a))
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as fh:
        data = json.load(fh)
    return {"source": "crossref", "query_title": title, "query_author": author, "url": url,
            "retrieved_utc": utcnow(), "items": data.get("message", {}).get("items", [])}


def main():
    for i, (title, author) in enumerate(TITLE_QUERIES, 1):
        try:
            rec = search(title, author)
        except Exception as exc:  # noqa: BLE001
            print("FAIL", title, type(exc).__name__, exc)
            continue
        p = HITS / ("T%02d_%s__crossref_title.json" % (i, slug(title)))
        p.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        print("T%02d %-72s -> %d hits" % (i, title[:72], len(rec["items"])))
        for it in rec["items"][:3]:
            yr = ""
            iss = it.get("issued", {}).get("date-parts", [[None]])
            if iss and iss[0]:
                yr = iss[0][0]
            print("      [%s] %s -- %s | %s | DOI %s"
                  % (yr, (it.get("title") or ["?"])[0][:64],
                     (it.get("container-title") or ["?"])[0][:34],
                     (it.get("author") or [{}])[0].get("family", "?"), it.get("DOI")))


if __name__ == "__main__":
    main()
