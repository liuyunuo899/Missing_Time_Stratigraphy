# -*- coding: utf-8 -*-
"""Exploratory QC of the raw Macrostrat download (no figure output)."""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

units = pd.read_csv(RAW / "macrostrat_unit_boundaries_all.csv", low_memory=False)
cols = pd.read_csv(RAW / "macrostrat_columns_all.csv", low_memory=False)
secs = pd.read_csv(RAW / "macrostrat_sections_all.csv", low_memory=False)

print("LONG-FILE COLUMNS (%d):" % units.shape[1])
print(list(units.columns))
print()
print("units rows:", len(units), " columns rows:", len(cols), " sections rows:", len(secs))

u = units.copy()
print("\n--- numeric summary of key fields ---")
for c in ["t_age", "b_age", "max_thick", "min_thick", "col_area", "pbdb_collections"]:
    if c in u.columns:
        s = pd.to_numeric(u[c], errors="coerce")
        print("%-16s n=%7d  nan=%6d  min=%12.4f  med=%10.4f  max=%12.2f"
              % (c, int(s.notna().sum()), int(s.isna().sum()), float(np.nanmin(s)),
                 float(np.nanmedian(s)), float(np.nanmax(s))))

print("\n--- integrity checks ---")
s_t = pd.to_numeric(u["t_age"], errors="coerce")
s_b = pd.to_numeric(u["b_age"], errors="coerce")
print("rows with b_age > t_age :", int((s_b > s_t).sum()))
print("rows with b_age == t_age:", int((s_b == s_t).sum()))
print("rows with t_age == 0    :", int((s_t == 0).sum()))
print("rows with max_thick == 0:", int((pd.to_numeric(u['max_thick'], errors='coerce') == 0).sum()))
print("rows with max_thick > 0 :", int((pd.to_numeric(u['max_thick'], errors='coerce') > 0).sum()))

print("\n--- project coverage ---")
print(u.groupby("project_id").size().to_string())

print("\n--- top columns by unit count ---")
print(u.groupby("col_id").size().sort_values(ascending=False).head(10).to_string())

print("\n--- age quantisation probe (are boundaries pinned to interval boundaries?) ---")
iv = pd.read_csv(RAW / "macrostrat_intervals_all.csv", low_memory=False)
print("intervals columns:", list(iv.columns)[:20])
if "b_age" in iv.columns:
    bnds = np.sort(pd.to_numeric(iv["b_age"], errors="coerce").dropna().unique())
    print("n distinct interval boundary ages:", len(bnds))
    bv = pd.concat([s_b, s_t]).dropna().unique()
    onboundary = np.isclose(bv[:, None], bnds[None, :], rtol=0, atol=1e-9).any(axis=1)
    print("fraction of unit boundary ages coinciding with an interval boundary: %.3f"
          % (onboundary.mean()))
