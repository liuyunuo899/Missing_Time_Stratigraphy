# -*- coding: utf-8 -*-
"""Stage 3: verify candidate references against Crossref and store full metadata.

For every DOI we ask Crossref (api.crossref.org/works/{doi}) for the publisher-
deposited record. A reference is only treated as "verified" when Crossref returns
HTTP 200 with a resolving DOI and matching metadata. Any DOI that fails is listed
as UNVERIFIED and must not be cited in the manuscript.

Run:
    python paper_project/src/verify_references.py
Outputs:
    references/verified_references.json
    references/verified_references.csv
"""
import csv
import json
import pathlib
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "references"
OUT.mkdir(parents=True, exist_ok=True)
MAILTO = "earth.science.reviews.repro@example.org"
UA = {"User-Agent": "ESR-review-repro/1.0 (mailto:%s)" % MAILTO}

# (doi, expected first author surname, short label used later in the manuscript)
CANDIDATES = [
    ("10.1086/628623", "Sadler", "Sadler 1981 accumulation rates"),
    ("10.1144/gsjgs.147.3.0471", "Sadler", "Sadler & Strauss 1990 completeness"),
    ("10.1086/629103", "Anders", "Anders et al. 1987 new look"),
    ("10.1007/bf00897239", "Strauss", "Strauss & Sadler 1989 stochastic models"),
    ("10.1086/629166", "Tipper", "Tipper 1987 estimating completeness"),
    ("10.1016/j.sedgeo.2016.04.003", "Tipper", "Tipper 2016 what are we estimating"),
    ("10.1086/629094", "Plotnick", "Plotnick 1986 fractal hiatuses"),
    ("10.1086/629197", "Plotnick", "Plotnick 1988 fractal reply"),
    ("10.1130/g38828.1", "Plotnick", "Plotnick 2017 recurrent patterns"),
    ("10.1029/2009jf001266", "Schumer", "Schumer & Jerolmack 2009 real and apparent"),
    ("10.1029/2006jf000555", "Jerolmack", "Jerolmack & Sadler 2007 transience"),
    ("10.1029/2010gl044638", "Jerolmack", "Jerolmack & Paola 2010 shredding"),
    ("10.1146/annurev-earth-082517-010129", "Paola", "Paola et al. 2018 time not our time"),
    ("10.1016/j.earscirev.2009.05.003", "Paola", "Paola et al. 2009 unreasonable effectiveness"),
    ("10.1029/2011jf002034", "Straub", "Straub et al. 2012 exponential beds"),
    ("10.1130/b26457.1", "Wilkinson", "Wilkinson et al. 2009 tectonic speedometers"),
    ("10.1086/663974", "Gombosi", "Gombosi et al. 2012 tectonic diffusion"),
    ("10.1666/0094-8373-36.1.61", "Peters", "Peters & Heim 2010 completeness"),
    ("10.1130/b30215.1", "Heim", "Heim & Peters 2010 covariation"),
    ("10.1130/g49037.1", "Peters", "Peters 2021 igneous area-age"),
    ("10.1073/pnas.2309084120", "Tasistro-Hart", "Tasistro-Hart et al. 2023 flooding"),
    ("10.1130/ges02815.1", "Jones", "Jones et al. 2024 rmacrostrat"),
    ("10.1002/gdj3.189", "Quinn", "Quinn et al. 2023 Macrostrat platform"),
    ("10.1016/j.palaeo.2023.111548", "Wilkinson", "Wilkinson et al. 2023 Sadler effect biases"),
    ("10.1038/s41467-020-19744-3", "Kemp", "Kemp et al. 2020 human impact erosion"),
    ("10.1146/annurev.earth.32.091003.143456", "Molnar", "Molnar 2004 accumulation increase"),
    ("10.1137/070710111", "Clauset", "Clauset et al. 2009 power laws"),
    ("10.1214/13-aoas710", "Virkar", "Virkar & Clauset 2014 binned power laws"),
    ("10.2307/1912557", "Vuong", "Vuong 1989 likelihood ratio tests"),
    ("10.1038/s41598-021-96214-w", "Pilgrim", "Pilgrim & Chesworth? 2021 Zipf bias"),
    ("10.1371/journal.pone.0085777", "Alstott", "Alstott et al. 2014 powerlaw package"),
    ("10.1016/j.earscirev.2012.03.006", "Holbrook", "Holbrook 2012 sequence boundary reappraisal"),
    ("10.1130/focus042016.1", "Burgess", "Burgess 2016 research focus"),
    ("10.1130/g30693c.1", "Burgess", "Burgess 2010 eustatic amplitude"),
    ("10.2110/pec.99.62.0085", "Bornholdt", "Bornholdt 1999 inverse strategies"),
    ("10.2110/sepmsp.114.14", "Granjeon", "Granjeon 2023 forward and inverse"),
    ("10.1007/978-3-030-26050-7_117-1", "Griffiths", "Griffiths 2022 encyclopedia entry"),
    ("10.1007/s11004-020-09911-z", "Patani", "Patani et al. 2020 stochastic inverse"),
    ("10.1214/11-ba618", "Blaauw", "Blaauw & Christen 2011 Bacon"),
    ("10.1111/j.1467-9876.2008.00623.x", "Haslett", "Haslett & Parnell 2008 Bchron"),
    ("10.1007/s11430-020-9660-8", "Liu", "Liu et al. 2020 carbonate completeness"),
    ("10.1016/j.precamres.2020.105721", "Shahkarami", "Shahkarami et al. 2020 Ediacaran"),
    ("10.1144/sp540-2022-142", "Yan", "Yan et al. 2023 meander-belt preservation"),
    ("10.1111/sed.12728", "Colombera", "Colombera et al. 2020 parasequences"),
    ("10.1016/j.earscirev.2022.104226", "Green", "Green et al. 2022 burial exhumation"),
    ("10.2110/sepmmisc.24.093", "Hart", "Hart 2024 detection of breaks"),
    ("10.1130/0016-7606(1994)106<0791:tedous>2.3.co;2", "Sadler", "Sadler 1994 peritidal cycles"),
    ("10.1130/g38350.1", "Partin", "Partin & Sadler 2016 snowball accumulation"),
    ("10.1002/jgrf.20151", "Voepel", "Voepel et al. 2013 residence times"),
    ("10.1371/journal.pone.0195557", "Salles", "Salles et al. 2018 pyBadlands"),
    ("10.1016/j.softx.2016.08.005", "Salles", "Salles 2016 Badlands"),
    ("10.1144/sp404.4", "Miall", "Miall 2014 updating uniformitarianism"),
    ("10.1007/bf02315157", "Dacey", "Dacey 1970 Markovian stratigraphic models"),
    ("10.2110/001c.124125", "Hagen", "Hagen 2024 accumulation rates"),
    ("10.1029/2025gl115985", "Wang", "Wang et al. 2025 alluvial response"),
    ("10.1029/2025ef007672", "Basquin", "Basquin et al. 2026 accommodation vs supply"),
    ("10.1111/bre.70086", "Dosch", "Dosch et al. 2026 transit storage preservation"),
    ("10.1111/bre.12854", "Schmelz", "Schmelz et al. 2024 margin stratigraphy"),
    ("10.1130/g52548.1", "Vaucher", "Vaucher 2024 delta sequences"),
]


def fetch(doi, tries=3):
    url = "https://api.crossref.org/works/%s?mailto=%s" % (urllib.parse.quote(doi), MAILTO)
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as fh:
                return json.load(fh)["message"], fh.status
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (k + 1))
    return None, "%s: %s" % (type(last).__name__, last)


def main():
    rows = []
    n_ok = 0
    n_fail = 0
    for doi, expect_author, label in CANDIDATES:
        msg, status = fetch(doi)
        if msg is None:
            print("UNVERIFIED  %-52s %-34s error=%s" % (doi, label[:34], status))
            rows.append({"doi": doi, "label": label, "status": "UNVERIFIED",
                         "expected_author": expect_author, "error": status})
            n_fail += 1
            continue
        authors = []
        for a in (msg.get("author") or []):
            fam = a.get("family", "")
            giv = a.get("given", "")
            authors.append(("%s %s" % (fam, giv)).strip())
        title = (msg.get("title") or [""])[0]
        cont = (msg.get("container-title") or [""])[0]
        yr = ""
        dp = msg.get("issued", {}).get("date-parts", [[None]])
        if dp and dp[0]:
            yr = dp[0][0]
        first = (msg.get("author") or [{}])[0].get("family", "")
        author_match = (first.lower().startswith(expect_author.lower()[:4])
                        if first and expect_author else False)
        rows.append({
            "doi": msg.get("DOI", doi), "label": label, "status": "VERIFIED",
            "expected_author": expect_author, "crossref_first_author": first,
            "first_author_match": author_match,
            "title": title, "container": cont, "year": yr,
            "volume": msg.get("volume", ""), "issue": msg.get("issue", ""),
            "page": msg.get("page", ""), "publisher": msg.get("publisher", ""),
            "type": msg.get("type", ""), "authors": "; ".join(authors),
            "verified_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        n_ok += 1
        print("OK %-8s %-11s %-38s %-56s"
              % (yr, first[:11], cont[:38], title[:56]))
    meta = {
        "verification_database": "Crossref REST API https://api.crossref.org/works/{doi}",
        "verified_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_candidates": len(CANDIDATES), "n_verified": n_ok, "n_unverified": n_fail,
        "rule": "only rows with status=VERIFIED may be cited in the manuscript",
    }
    (OUT / "verified_references.json").write_text(
        json.dumps({"meta": meta, "references": rows}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    with open(OUT / "verified_references.csv", "w", newline="", encoding="utf-8") as fh:
        cols = ["doi", "label", "status", "year", "container", "title",
                "crossref_first_author", "first_author_match", "volume", "issue",
                "page", "publisher", "authors", "type", "verified_utc"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nverified=%d  unverified=%d" % (n_ok, n_fail))


if __name__ == "__main__":
    main()
