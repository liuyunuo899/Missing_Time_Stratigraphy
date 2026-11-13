# Missing Time in the Sedimentary Record — Code and Data

This archive provides the actual reproduced Macrostrat input snapshot, retained outputs, Python analysis and model scripts, and one PNG for each of Figures 1–13. Only the retained analyses are included. No unpublished missing supplementary results are claimed.

## Layout
- `config/`: Sedapp run configuration.
- `data/raw/`: seven Macrostrat v2 bulk CSV downloads (10 November 2026) plus provenance/checksums.
- `data/processed/`: derived observational inputs and numerical results.
- `data/synthetic/`: synthetic model outputs and summary files.
- `data/sedapp/`: Sedapp simulation output and associated arrays.
- `figures/`: 13 PNG figures, one per figure (no duplicate PDF exports).
- `src/`: analysis and figure-generating Python source.
- `DATA_SOURCES.md`: source/licence and snapshot caveats.
- `requirements.txt`: Python package requirements.

## Python environment
Python 3.11+ is recommended. At the project root:

```bash
python -m pip install -r requirements.txt
```

## Running the scripts
Run commands from the folder containing this README, in approximately this order:

```bash
python src/test_stats_tools.py
python src/data_processing.py
python src/analysis_real.py
python src/sadler_decomposition.py
python src/robustness_negative_controls.py
python src/round2_computations.py
python src/round3_computations.py
python src/experiments_synthetic.py
python src/calibration_sadler.py
python src/round3_bias_on_b.py
python src/run_sedapp_case.py
python src/make_figures.py
```

`download_macrostrat.py` is optional: *do not run it for exact comparison with this archive*, because a new API download can differ from the included snapshot. The long-response `macrostrat_unit_boundaries_all.csv` has 35,481 parsed records; 36,059 was a physical-line count in the original download log (embedded line breaks). The `.json` numerical summaries supplied here are retained results of previous executions; packaging did not freshly rerun every costly Monte Carlo procedure. Figure source matches the available later layout-corrected script; confirm against your most recent local `make_figures.py` before final deposition.

No generated figure image or archived numerical output should be interpreted as independent field validation. The Sedapp example is a model demonstration.
