# -*- coding: utf-8 -*-
"""Build alphabetical author-year reference list and citation keys from the
Crossref-verified metadata table.

Only rows whose status is VERIFIED are used, so nothing that could not be
resolved against Crossref reaches the manuscript.

Run:
    python paper_project/src/build_references.py
Outputs:
    references/reference_list.txt
    references/citation_keys.json
"""
import html
import json
import pathlib
import re

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT / "references"
NAMED_FILE = REF / "named_references.json"


def parse_authors(s):
    out = []
    for a in (s or "").split(";"):
        a = a.strip()
        if a:
            out.append(a)
    return out


def tidy_number(v):
    """Crossref sometimes reports volume/issue as floats (e.g. 9.0)."""
    s = str(v or "").strip()
    if not s or s == "nan":
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    return s


def clean_html(s):
    """Decode publisher entities before removing residual HTML markup."""
    s = html.unescape(str(s))
    # Crossref book records sometimes encode a subtitle as an XML-like tag.
    # Preserve the semantic break instead of concatenating the two titles.
    s = re.sub(r"<subtitle\s*>", ": ", s, flags=re.IGNORECASE)
    s = re.sub(r"</subtitle\s*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    return " ".join(s.split())


def initials(given):
    parts = re.split(r"[\s\-]+", (given or "").strip())
    return "".join(p[0].upper() + "." for p in parts if p)


def build_key(first_author, year, used):
    norm_first = " ".join(w.capitalize() for w in first_author.split())
    base = re.sub(r"[^A-Za-z]", "", norm_first.split()[-1]) if norm_first else "Anon"
    key = "%s, %s" % (base, year)
    if key not in used:
        used.add(key)
        return key
    for suf in "abcdefghijklmnop":
        cand = "%s%s, %s" % (base, suf, year)
        if cand not in used:
            used.add(cand)
            return cand
    return key  # pragma: no cover


def format_entry(row):
    auth = parse_authors(row["authors"])
    pretty = []
    for a in auth:
        parts = a.split()
        if not parts:
            continue
        surname = parts[0]
        given = " ".join(parts[1:])
        pretty.append("%s %s" % (surname, initials(given)))
    if len(pretty) == 0:
        authors_str = "Anonymous"
    elif len(pretty) == 1:
        authors_str = pretty[0]
    else:
        authors_str = ", ".join(pretty[:-1]) + ", and " + pretty[-1]
    title = clean_html(str(row["title"] or "").strip()).rstrip(".")
    pieces = [authors_str, "%s. %s." % (row["year"], title)]
    container = clean_html(str(row.get("container") or "").strip())
    rtype = str(row.get("type") or "")
    vol = tidy_number(row.get("volume"))
    issue = tidy_number(row.get("issue"))
    page = str(row.get("page") or "").strip()
    core = container
    if vol and vol != "nan":
        core += " %s" % vol
    if issue and issue != "nan":
        core += "(%s)" % issue
    if page and page != "nan":
        core += ", %s" % page
    if core:
        pieces.append(("In: %s." % core) if rtype.startswith("book")
                      else (core + "."))
    if row.get("publisher"):
        pass
    # Angle brackets are legal in some older DOI suffixes (for example the
    # SADLER 1994 record) but are parsed as markup by Markdown. Percent-encoding
    # preserves the DOI URL without allowing raw-HTML consumption.
    doi_url = str(row["doi"]).replace("<", "%3C").replace(">", "%3E")
    pieces.append("https://doi.org/%s" % doi_url)
    return " ".join(p for p in pieces if p and p.strip())


def main():
    df = pd.read_csv(REF / "verified_references.csv")
    df = df[df["status"] == "VERIFIED"].copy()
    curated = REF / "final_reference_dois.json"
    if curated.exists():
        keep = {str(d).lower() for d in json.loads(curated.read_text(encoding="utf-8"))}
        df = df[df["doi"].str.lower().isin(keep)].copy()
    df["year"] = df["year"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["sort_author"] = df["crossref_first_author"].fillna("zzz").astype(str).str.casefold()
    df["sort_year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0)
    df = df.sort_values(["sort_author", "sort_year", "title"])

    named = json.loads(NAMED_FILE.read_text(encoding="utf-8")) if NAMED_FILE.exists() else {}
    used = set()
    rows = []
    keys = {}
    for _, r in df.iterrows():
        first = (str(r["crossref_first_author"] or "")).strip() or "Anon"
        key = build_key(first, r["year"], used)
        entry = format_entry(r)
        rows.append((key, entry, r["doi"], r["label"]))
        keys[r["doi"]] = {"key": key, "label": r["label"], "doi": r["doi"],
                          "title": r["title"], "year": r["year"]}
    lines = [entry + "  \\[%s\\]" % key for key, entry, doi, label in rows]
    (REF / "reference_list.txt").write_text("\n".join(lines), encoding="utf-8")
    (REF / "citation_keys.json").write_text(json.dumps(keys, indent=1), encoding="utf-8")
    manuscript_refs = ROOT / "manuscript" / "14_references.md"
    manuscript_refs.write_text(
        "## References\n\n" + "\n\n".join(entry for _, entry, _, _ in rows) + "\n",
        encoding="utf-8",
    )
    print("references:", len(rows))
    for key, entry, doi, label in rows:
        print("  %-22s %s" % (key, label))


if __name__ == "__main__":
    main()
