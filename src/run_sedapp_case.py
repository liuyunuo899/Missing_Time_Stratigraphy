# -*- coding: utf-8 -*-
"""Run the user-supplied Sedapp Python port on a new deterministic case.

The case is a process-model demonstration, not an independent validation of a
real basin. All outputs are derived from the arrays returned by run_sedapp().
No random numbers are used.

Run:
    python3 src/run_sedapp_case.py
"""
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sedapp_py import run_sedapp

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "sedapp_case.json"
OUT = ROOT / "data" / "sedapp"
SOURCE = ROOT / "src" / "sedapp_py.py"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(CONFIG, encoding="utf-8") as handle:
        config = json.load(handle)
    p = dict(config["parameters"])
    dt = float(p["dt"])
    t_end = float(p["t1"])
    time_edges = np.arange(dt, t_end + 1e-10, dt)
    sl = p["sea_level"]
    p["sl_series"] = (float(sl["mean_m"]) + float(sl["amplitude_m"])
                      * np.sin(2.0 * np.pi * time_edges / float(sl["period_Myr"])
                               + float(sl["phase_radians"])))
    p["topo_samples"] = np.asarray(p["topo_samples"], dtype=float)
    p["subsidence_samples"] = np.asarray(p["subsidence_samples"], dtype=float)

    started = datetime.now(timezone.utc)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = run_sedapp(p)
    finished = datetime.now(timezone.utc)

    z = np.asarray(results["z_matrix"], dtype=float)
    lt = np.asarray(results["layer_thickness_matrix"], dtype=float)
    sf = np.asarray(results["sand_fraction_matrix"], dtype=float)
    por = np.asarray(results["porosity_matrix"], dtype=float)
    wd = np.asarray(results["wd_matrix"], dtype=float)
    x = np.asarray(results["x_value"], dtype=float)
    times = np.asarray(results["time_series"], dtype=float)
    base = np.asarray(results["h0"], dtype=float)
    shoreline_x = np.asarray(results["shoreline_x"], dtype=float)
    shoreline_y = np.asarray(results["shoreline_y"], dtype=float)
    nx, k = z.shape
    tolerance = 1e-8
    final_thickness = lt.sum(axis=1)
    geometry_residual = z[:, -1] - (base + final_thickness)
    array_names = ["z_matrix", "layer_thickness_matrix", "sand_fraction_matrix",
                   "porosity_matrix", "wd_matrix", "h0_series"]
    arrays_finite = {name: bool(np.isfinite(np.asarray(results[name])).all())
                     for name in array_names}
    if not all(arrays_finite.values()):
        raise RuntimeError("non-finite Sedapp output: %s" % arrays_finite)

    np.savez_compressed(
        OUT / "sedapp_case_output.npz",
        z_matrix=z, layer_thickness_matrix=lt, sand_fraction_matrix=sf,
        porosity_matrix=por, wd_matrix=wd, h0_series=results["h0_series"],
        x_value=x, time_series=times, h0=base, shoreline_x=shoreline_x,
        shoreline_y=shoreline_y, slope_fluvial=results["slope_fluvial"],
        sl_series=np.asarray(p["sl_series"], dtype=float))

    step_table = pd.DataFrame({
        "step": np.arange(1, k + 1),
        "time_Myr": times,
        "sea_level_m": np.asarray(p["sl_series"], dtype=float),
        "shoreline_x_m": shoreline_x,
        "shoreline_y_m": shoreline_y,
        "mean_water_depth_m": wd.mean(axis=0),
        "mean_net_surface_increment_m": lt.mean(axis=0),
        "minimum_net_surface_increment_m": lt.min(axis=0),
        "maximum_net_surface_increment_m": lt.max(axis=0),
        "n_cells_net_positive": (lt > tolerance).sum(axis=0),
        "n_cells_net_negative": (lt < -tolerance).sum(axis=0),
    })
    step_table.to_csv(OUT / "sedapp_case_by_timestep.csv", index=False)

    cell_table = pd.DataFrame({
        "cell": np.arange(1, nx + 1),
        "x_m": x,
        "final_surface_elevation_m": z[:, -1],
        "final_base_elevation_m": base,
        "final_compacted_thickness_m": final_thickness,
        "final_sand_fraction": sf[:, -1],
        "final_porosity": por[:, -1],
        "final_water_depth_m": wd[:, -1],
        "net_positive_increment_steps": (lt > tolerance).sum(axis=1),
        "net_negative_increment_steps": (lt < -tolerance).sum(axis=1),
        "net_positive_increment_time_fraction": (lt > tolerance).mean(axis=1),
    })
    cell_table.to_csv(OUT / "sedapp_case_by_cell.csv", index=False)

    run_configuration = {
        "case_name": config["case_name"],
        "scientific_role": config["scientific_role"],
        "parameters": config["parameters"],
        "derived_sea_level_series_m": np.asarray(p["sl_series"]).tolist(),
        "source_file": "src/sedapp_py.py",
        "source_sha256": sha256(SOURCE),
        "config_sha256": sha256(CONFIG),
        "deterministic": True,
        "random_seed": None,
    }
    with open(OUT / "sedapp_case_run_configuration.json", "w", encoding="utf-8") as handle:
        json.dump(run_configuration, handle, indent=1)

    summary = {
        "case_name": config["case_name"],
        "scientific_role": config["scientific_role"],
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_utc": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_s": (finished - started).total_seconds(),
        "source_sha256": run_configuration["source_sha256"],
        "config_sha256": run_configuration["config_sha256"],
        "n_cells": int(nx),
        "n_timesteps": int(k),
        "domain_length_m": float(p["L"]),
        "time_span_Myr": float(t_end - float(p["t"])),
        "dt_Myr": dt,
        "all_output_arrays_finite": True,
        "array_finite_checks": arrays_finite,
        "final_surface_elevation_range_m": [float(z[:, -1].min()),
                                             float(z[:, -1].max())],
        "final_compacted_thickness_range_m": [float(final_thickness.min()),
                                               float(final_thickness.max())],
        "maximum_absolute_geometry_residual_m": float(np.max(np.abs(geometry_residual))),
        "sea_level_range_m": [float(np.min(p["sl_series"])),
                               float(np.max(p["sl_series"]))],
        "shoreline_x_range_m": [float(shoreline_x.min()), float(shoreline_x.max())],
        "shoreline_migration_span_m": float(np.ptp(shoreline_x)),
        "net_positive_increment_steps": int(np.sum(lt > tolerance)),
        "net_negative_increment_steps": int(np.sum(lt < -tolerance)),
        "net_positive_increment_fraction": float(np.mean(lt > tolerance)),
        "warning_count": len(caught),
        "warning_messages": sorted({str(item.message) for item in caught}),
        "interpretation_limits": [
            "This is a deterministic process-model demonstration, not field validation.",
            "A positive layer-matrix entry is a net surface increment, not proof that an original depositional layer survived.",
            "The net-positive increment fraction is not stratigraphic completeness.",
            "Subsidence, compaction and erosion prevent direct conversion of positive entries into represented time.",
        ],
    }
    with open(OUT / "sedapp_case_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, allow_nan=False)

    print(json.dumps(summary, indent=1))
    print("Wrote Sedapp outputs to", OUT)


if __name__ == "__main__":
    main()
