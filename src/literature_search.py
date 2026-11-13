# -*- coding: utf-8 -*-
"""Stage 3: harvest candidate literature from Crossref + OpenAlex APIs.

Every hit is stored verbatim as JSON together with the query string and UTC
timestamp, so that the inclusion decisions reported in the manuscript can be
traced back to an actual API response.

Run:
    python paper_project/src/literature_search.py
Outputs:
    references/raw_hits/<label>__<source>.json
    references/raw_hits/_run_log.txt
"""
import json
import pathlib
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
HITS = ROOT / "references" / "raw_hits"
HITS.mkdir(parents=True, exist_ok=True)
MAILTO = "earth.science.reviews.repro@example.org"
UA = {"User-Agent": "ESR-review-repro/1.0 (mailto:%s)" % MAILTO}


def fetch_json(url, tries=3, pause=2.0):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as fh:
                return json.load(fh), fh.status
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(pause * (k + 1))
    print("  !! fetch failed:", url, type(last).__name__, last)
    return None, None


def slug(text):
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in text.lower()]
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:70]


# ----------------------------------------------------------------- searches --
TOPIC_QUERIES = [
    "stratigraphic completeness hiatus duration distribution",
    "sediment accumulation rates completeness stratigraphic sections Sadler",
    "fractal model distribution of stratigraphic hiatuses",
    "Sadler effect apparent sedimentation rate time span power law",
    "power law distribution hiatus durations sedimentary record",
    "unconformity formation controls bypassing sediment transport storage",
    "stratigraphic forward modelling inverse model tautochron toplap",
    "time stratigraphic fidelity environmental signal preservation transport noise",
    "shredding environmental signals by sediment transport",
    "uncertainty in age depth models versus chronostratigraphic error",
    "retrospective statistics age ranges uptake of the stratigraphic filter",
    "Phanerozoic rock recycling area age frequency of sedimentary rocks",
    "quantifying sampling completeness of the rock record with global databases",
    "missing time at unconformities and confidence intervals of chronostratigraphic completeness",
    "tautochrons, diastems, and the temporal completeness of basin sediment records",
]

TITLE_QUERIES = [
    "Sediment accumulation rates and the completeness of stratigraphic sections",
    "Estimation of completeness of stratigraphical sections using empirical data and theoretical models",
    "A new look at sedimentation rates and the completeness of the stratigraphic record",
    "A fractal model for the distribution of stratigraphic hiatuses",
    "Fractal structure of the stratigraphic record",
    "Transience and persistence in the depositional record of continental margins",
    "Shredding of environmental signals by sediment transport",
    "The unreasonable effectiveness of stratigraphic and geomorphic experiments",
    "Real and apparent changes in sediment deposition rates through time",
    "Measured rates of sediment accumulation a formal reconsideration",
    "Global geologic maps are tectonic speedometers rates of rock cycling from area-age frequencies",
    "The geological completeness of paleontological sampling in North America",
    "Power-law distributions in empirical data",
    "pyBadlands a framework to simulate sediment transport landscape dynamics and basin stratigraphic evolution",
    "Updating uniformitarianism stratigraphy as just a set of sedimentary processes",
    "The nature of the stratigraphical record",
    "Rhythms and the measurement of geologic time",
    "Influence of hiatuses on sediment accumulation rates",
]


def crossref_query(q, rows, extra=""):
    url = ("https://api.crossref.org/works?rows=%d&select=DOI,title,subtitle,author,"
           "container-title,issued,volume,issue,page,type,publisher,URL&mailto=%s&query.bibliographic=%s%s"
           % (rows, MAILTO, urllib.parse.quote(q), extra))
    data, status = fetch_json(url)
    return {"source": "crossref", "query": q, "url": url, "http_status": status,
            "retrieved_utc": utcnow(), "items": (data or {}).get("message", {}).get("items", [])}


def openalex_query(q, per_page):
    url = ("https://api.openalex.org/works?per-page=%d&mailto=%s&search=%s"
           % (per_page, MAILTO, urllib.parse.quote(q)))
    data, status = fetch_json(url)
    return {"source": "openalex", "query": q, "url": url, "http_status": status,
            "retrieved_utc": utcnow(), "results": (data or {}).get("results", []),
            "count": (data or {}).get("meta", {}).get("count")}


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    log = ["LITERATURE SEARCH RUN", "run_started_utc: " + utcnow(),
           "databases: Crossref REST API (api.crossref.org/works), OpenAlex REST API (api.openalex.org/works)",
           "note: no Web-of-Science/Scopus token was available in this environment; "
           "Crossref and OpenAlex index the same publisher-deposited metadata."]
    n_topic = len(TOPIC_QUERIES)
    for i, q in enumerate(TOPIC_QUERIES, 1):
        print("[%d/%d] crossref topic: %s" % (i, n_topic, q))
        rec = crossref_query(q, 20)
        p = HITS / ("%02dtopic_%s__crossref.json" % (i, slug(q)))
        p.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        print("     -> %d items -> %s" % (len(rec["items"]), p.name))
        log.append("crossref\t%s\t%s\tn_items=%d" % (utcnow(), q, len(rec["items"])))

    if True:
        for i, q in enumerate(TOPIC_QUERIES, 1):
            print("[%d/%d] openalex topic: %s" % (i, n_topic, q))
            rec = openalex_query(q, 25)
            p = HITS / ("%02dtopic_%s__openalex.json" % (i, slug(q)))
            p.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
            print("     -> %d items (count %s) -> %s" % (len(rec["results"]), rec["count"], p.name))
            log.append("openalex\t%s\t%s\tn_items=%d\tcorpus_count=%s"
                       % (utcnow(), q, len(rec["results"]), rec["count"]))

    for i, q in enumerate(TITLE_QUERIES, 1):
        print("[title %d/%d] %s" % (i, len(TITLE_QUERIES), q))
        rec = crossref_query(q, 5, "&query.author=&sort=relevance")
        p = HITS / ("T%02d_%s__crossref_title.json" % (i, slug(q)))
        p.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        print("     -> %d items -> %s" % (len(rec["items"]), p.name))
        log.append("crossref_title\t%s\t%s\tn_items=%d" % (utcnow(), q, len(rec["items"])))

    log.append("run_finished_utc: " + utcnow())
    (HITS / "_run_log.txt").write_text("\n".join(log), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
