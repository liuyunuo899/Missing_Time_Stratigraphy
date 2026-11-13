# Data sources

The seven files in `data/raw/` were retrieved from Macrostrat v2 bulk CSV endpoints on 2026-11-10. Exact endpoint URLs, timestamps and SHA-256 digests are retained in `data/raw/_macrostrat_provenance.json`. The API reports CC-BY 4.0; attribution to Macrostrat and relevant source publications should be retained.

The geologic-time intervals export has 1,717 parsed records. An earlier author snapshot had 1,716; the added interval is not referenced by `t_int_id` or `b_int_id` in the downloaded unit dataset. Do not assert that downloads on different dates are byte-identical.

The unit long-response CSV has 35,481 parsed CSV records but 36,059 physical data lines in the downloader's earlier log because quoted fields contain embedded newlines. `Rows` in Table 1 should use parsed record counts.

The contents of `data/processed`, `data/synthetic`, and `data/sedapp` are generated derivatives rather than new third-party downloads.
