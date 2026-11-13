# -*- coding: utf-8 -*-
"""Second batch of targeted literature searches (theories, recent work, methods)."""
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

TARGETS = [
    ("fractal structure of the stratigraphic record", "Grotzinger"),
    ("time not our time physical limits to the predictability of sediment transport and erosion", "Paola"),
    ("flexible paleoclimate age-depth models using an autoregressive gamma process", "Blaauw"),
    ("simple Bayesian statistical radiocarbon age-depth calibration software", "Haslett"),
    ("likelihood ratio tests for model selection and non-nested hypotheses", "Vuong"),
    ("powerlaw a python package for analysis of heavy-tailed distributions", "Alstott"),
    ("analysis of heavy-tailed distributions in the geosciences", "Nakamura"),
    ("the ubiquity of stratigraphic incompleteness quantitative characterisation", "Tipper"),
    ("Marcov chain stratigraphic completeness", "Dacey"),
    ("toward a more precise definition of sediment accommodation with sequence stratigraphy", "Holbrook"),
    ("autostratigraphic modelling accommodation sediment supply with 1-D diffusion", "Rivenaes"),
    ("three-dimensional quantitative modelling of clastic sediment transport in basins", "Granjeon"),
    ("inverse approach stratigraphic modelling sensitivity study with multiple basins", "Bornholdt"),
    ("deep-sea hiatus record global distribution missing time deep water holes", "Westerhold"),
    ("sediment accumulation rates siayas oceans continental margin preservation bias", "Molnar"),
    ("stratigraphic model simulation of geological processes", "Harbaugh"),
    ("burial of organic carbon and pyrite sulfur in the modern ocean its geochemical", "Berner"),
    ("modelling accommodation versus sediment supply with sequence stratigraphy", "Burgess"),
]

RECENT_QUERIES = [
    "stratigraphic completeness paleobiology database rock record",
    "quantifying missing time unconformity resolution obscure",
    "time resolution geochronology calibration of the stratigraphic record",
    "preservation potential Miocene hiatus map sediments",
    "sink storage number Global cycle sediment routing",
    "Bayesian inversion stratigraphic forward model episodic sedimentation",
    "machine learning inference of missing time in borehole well logs",
    "hiatus detection multiproxy records astrochronology deep-sea cores",
    "age model error propagation for hiatus distribution estimates",
    "nonuniqueness inverse problem history matching geological model calibration",
]


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slug(t):
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in t.lower()]
    return "".join(keep)[:70]


def crossref(title, author, extra=""):
    url = ("https://api.crossref.org/works?rows=5"
           "&select=DOI,title,author,container-title,issued,volume,issue,page,publisher"
           "&mailto=%s&query.bibliographic=%s&query.author=%s%s"
           % (MAILTO, urllib.parse.quote(title), urllib.parse.quote(author), extra))
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as fh:
        data = json.load(fh)
    return url, data.get("message", {}).get("items", [])


def show(tag, rec):
    print("%s %-70s -> %d hits" % (tag, rec["query_title"][:70], len(rec["items"])))
    for it in rec["items"][:4]:
        yr = ""
        iss = it.get("issued", {}).get("date-parts", [[None]])
        if iss and iss[0]:
            yr = iss[0][0]
        print("      [%s] %s -- %s | %s | %s"
              % (yr, (it.get("title") or ["?"])[0][:62],
                 (it.get("container-title") or ["?"])[0][:32],
                 (it.get("author") or [{}])[0].get("family", "?"), it.get("DOI")))


def main():
    print("### TARGETTED TITLE+ AUTHOR SEARCHES")
    for i, (title, author) in enumerate(TARGETS, 1):
        try:
            url, items = crossref(title, author)
        except Exception as exc:  # noqa: BLE001
            print("B%02d FAIL %s" % (i, exc))
            continue
        rec = {"source": "crossref", "query_title": title, "query_author": author,
               "url": url, "retrieved_utc": utcnow(), "items": items}
        p = HITS / ("B%02d_%s__crossref_title.json" % (i, slug(title)))
        p.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        show("B%02d" % i, rec)

    print()
    print("### RECENT-WORK SEARCHES (published since 2019)")
    for i, q in enumerate(RECENT_QUERIES, 1):
        extra = "&filter=from-pub-date:2019-01-01,type:journal-article"
        try:
            url, items = crossref(q, "", extra)
        except Exception as exc:  # noqa: BLE001
            print("R%02d FAIL %s" % (i, exc))
            continue
        rec = {"source": "crossref", "query_title": q, "query_author": "", "url": url,
               "retrieved_utc": utcnow(), "items": items}
        p = HITS / ("R%02d_%s__crossref_recent.json" % (i, slug(q)))
        p.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        show("R%02d" % i, rec)


if __name__ == "__main__":
    main()
