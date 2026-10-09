"""QC comparison of 7468 sampling H5: WITHOUT versus WITH NORDIC.
Read-only: does not change either H5. Produces two PNG figures and one Excel file.
Layer profile is descriptive raw sampled signal, not a GLM result or a significance test.
"""
from pathlib import Path
import sys
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#WITHOUT_H5 = Path(r"E:\layerfmri\LAYER_s7468_warpednordic\codes\Laminarprofile\Laminar\1. Sampling\Without\pIns_laminar_sampling_7468_without_NORDIC.h5")
#WITH_H5 = Path(r"E:\layerfmri\LAYER_s7468_warpednordic\codes\Laminarprofile\Laminar\1. Sampling\With\pIns_laminar_sampling_7468_with_NORDIC.h5")

WITH_H5 = Path(r"D:\NORDIC_Subject1_project\Laminarprofile\1. Sampling\With\pIns_laminar_sampling_7485_with_NORDIC.h5")
WITHOUT_H5 = Path(r"D:\NORDIC_Subject1_project\Laminarprofile\1. Sampling\Without\pIns_laminar_sampling_7485_without_NORDIC.h5")

OUT_DIR = Path(r"D:\NORDIC_Subject1_project\Laminarprofile\4. QC Comparison")
OVERWRITE = False
SIDES = ("left", "right")
LABELS = {"without_NORDIC": "Without NORDIC", "with_NORDIC": "With NORDIC"}
COLORS = {"without_NORDIC": "#D55E00", "with_NORDIC": "#0072B2"}


def die(message):
    raise RuntimeError(message)


def check_output(path):
    if path.exists() and not OVERWRITE:
        die(f"Output already exists; set OVERWRITE=True only to replace it:\n{path}")


def read_h5(path, condition):
    if not path.is_file():
        die(f"Missing H5 file for {condition}:\n{path}")
    result = {"path": str(path), "condition": condition, "sides": {}}
    with h5py.File(path, "r") as h5:
        if "run_lengths" not in h5:
            die(f"{path}: missing run_lengths")
        run_lengths = np.asarray(h5["run_lengths"][:], dtype=int)
        result["run_lengths"] = run_lengths
        for side in SIDES:
            if side not in h5 or "layeractivation" not in h5[side]:
                die(f"{path}: missing {side}/layeractivation")
            data = h5[side]["layeractivation"]
            if data.ndim != 3 or data.shape[1] != 20:
                die(f"{path}: {side}/layeractivation must be columns × 20 depths × time; found {data.shape}")
            if data.shape[2] != int(run_lengths.sum()):
                die(f"{path}: {side} timepoints {data.shape[2]} disagree with run_lengths {run_lengths.tolist()}")
            # Read one depth at a time to keep memory use low.
            layer_mean = np.empty(20, dtype=np.float64)
            layer_sd_columns = np.empty(20, dtype=np.float64)
            for depth in range(20):
                values = np.asarray(data[:, depth, :], dtype=np.float64)
                if not np.isfinite(values).all():
                    die(f"{path}: non-finite sampled values in {side}, depth {depth}")
                per_column = values.mean(axis=1)
                layer_mean[depth] = per_column.mean()
                layer_sd_columns[depth] = per_column.std(ddof=1)
            # Average depths 0..14 separately for every timepoint: deep75 global mean time course.
            deep75 = np.asarray(data[:, :15, :], dtype=np.float64).mean(axis=(0, 1))
            all_depth = np.asarray(data[:, :, :], dtype=np.float64).mean(axis=(0, 1))
            result["sides"][side] = {
                "n_columns": int(data.shape[0]),
                "n_depths": int(data.shape[1]),
                "n_timepoints": int(data.shape[2]),
                "layer_mean": layer_mean,
                "layer_sd_columns": layer_sd_columns,
                "deep75_timeseries": deep75,
                "all_depth_timeseries": all_depth,
            }
    return result


def percent_signal_change(x):
    baseline = float(np.mean(x))
    if baseline == 0 or not np.isfinite(baseline):
        die("Cannot calculate percent signal change: zero/non-finite time-course mean")
    return (x / baseline - 1.0) * 100.0


def contiguous_runs(lengths):
    return np.cumsum(lengths)[:-1]


def make_timeseries_figure(data):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, constrained_layout=True)
    for ax, side in zip(axes, SIDES):
        for condition in LABELS:
            x = data[condition]["sides"][side]["deep75_timeseries"]
            ax.plot(percent_signal_change(x), color=COLORS[condition], linewidth=0.7, label=LABELS[condition])
        for boundary in contiguous_runs(data["without_NORDIC"]["run_lengths"]):
            ax.axvline(boundary - 0.5, color="0.45", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="0.55", linewidth=0.8)
        ax.set_title(f"{side.upper()} — deep 75% mean signal across all sampled columns")
        ax.set_ylabel("Signal change from each series mean (%)")
        ax.legend(frameon=False, loc="upper right")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Timepoint (three runs concatenated)")
    fig.suptitle("Sampling QC: deep-75% time-course comparison", fontsize=14)
    path = OUT_DIR / "Figure_01_deep75_timeseries_without_vs_with_NORDIC.png"
    fig.savefig(path, dpi=250, facecolor="white")
    plt.close(fig)
    return path


def make_layer_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False, constrained_layout=True)
    depths = np.arange(20)
    for ax, side in zip(axes, SIDES):
        for condition in LABELS:
            d = data[condition]["sides"][side]
            mean = d["layer_mean"]
            sem = d["layer_sd_columns"] / np.sqrt(d["n_columns"])
            ax.plot(depths, mean, marker="o", markersize=3.5, linewidth=1.8, color=COLORS[condition], label=LABELS[condition])
            ax.fill_between(depths, mean-sem, mean+sem, color=COLORS[condition], alpha=0.18, linewidth=0)
        ax.axvspan(-0.5, 14.5, color="0.85", alpha=0.35, label="Deep 75% used for selection")
        ax.set_xlim(-0.5, 19.5)
        ax.set_xticks(depths)
        ax.set_xlabel("Depth index (0 = white boundary; 19 = pial boundary)")
        ax.set_ylabel("Mean raw sampled signal")
        ax.set_title(f"{side.upper()} — mean signal by cortical depth")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Sampling QC: raw signal across 20 depths", fontsize=14)
    path = OUT_DIR / "Figure_02_raw_signal_depth_0_to_19_without_vs_with_NORDIC.png"
    fig.savefig(path, dpi=250, facecolor="white")
    plt.close(fig)
    return path


def make_excel(data):
    rows = []
    summary = []
    ts_rows = []
    for condition in LABELS:
        for side in SIDES:
            d = data[condition]["sides"][side]
            profile = d["layer_mean"]
            for depth in range(20):
                rows.append({
                    "condition": condition,
                    "hemisphere": side,
                    "depth_index_zero_based": depth,
                    "depth_index_one_based": depth + 1,
                    "normalized_depth_white_0_to_pial_1": depth / 19,
                    "mean_raw_sampled_signal": profile[depth],
                    "sd_of_column_means": d["layer_sd_columns"][depth],
                    "sem_of_column_means": d["layer_sd_columns"][depth] / np.sqrt(d["n_columns"]),
                    "n_columns": d["n_columns"],
                })
            x = d["deep75_timeseries"]
            summary.append({
                "condition": condition, "hemisphere": side, "n_columns": d["n_columns"],
                "n_depths": d["n_depths"], "n_timepoints": d["n_timepoints"],
                "deep75_raw_mean": x.mean(), "deep75_raw_sd_time": x.std(ddof=1),
                "deep75_temporal_cv_percent": 100*x.std(ddof=1)/x.mean(),
                "depth0_mean_raw_signal": profile[0], "depth19_mean_raw_signal": profile[19],
                "depth19_minus_depth0_raw_signal": profile[19]-profile[0],
                "depth19_over_depth0_ratio": profile[19]/profile[0],
            })
            for t, value in enumerate(x):
                ts_rows.append({"condition": condition, "hemisphere": side, "timepoint_zero_based": t, "deep75_raw_mean_signal": value, "deep75_percent_from_own_mean": percent_signal_change(x)[t]})
    metadata = pd.DataFrame({"field": ["without_h5", "with_h5", "run_lengths", "depth_definition", "deep75_definition", "interpretation"], "value": [data["without_NORDIC"]["path"], data["with_NORDIC"]["path"], json.dumps(data["without_NORDIC"]["run_lengths"].tolist()), "0 = white/GM-WM boundary; 19 = pial/GM-CSF boundary", "depths 0..14 (15 of 20)", "Descriptive sampling QC only; raw signal and no statistical inference."]})
    path = OUT_DIR / "QC_sampling_without_vs_with_NORDIC.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(rows).to_excel(writer, sheet_name="depth_values", index=False)
        pd.DataFrame(ts_rows).to_excel(writer, sheet_name="deep75_timeseries", index=False)
        metadata.to_excel(writer, sheet_name="metadata", index=False)
    return path


def main():
    for path in (WITHOUT_H5, WITH_H5):
        if not path.is_file():
            die(f"Input H5 not found:\n{path}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [OUT_DIR / "Figure_01_deep75_timeseries_without_vs_with_NORDIC.png", OUT_DIR / "Figure_02_raw_signal_depth_0_to_19_without_vs_with_NORDIC.png", OUT_DIR / "QC_sampling_without_vs_with_NORDIC.xlsx"]
    for path in outputs:
        check_output(path)
    data = {"without_NORDIC": read_h5(WITHOUT_H5, "without_NORDIC"), "with_NORDIC": read_h5(WITH_H5, "with_NORDIC")}
    if not np.array_equal(data["without_NORDIC"]["run_lengths"], data["with_NORDIC"]["run_lengths"]):
        die("Without/With run_lengths differ; not comparable as one aligned time-course.")
    for side in SIDES:
        if data["without_NORDIC"]["sides"][side]["n_timepoints"] != data["with_NORDIC"]["sides"][side]["n_timepoints"]:
            die(f"{side}: timepoint counts differ between H5 files.")
    print("Reading complete. Writing figures and Excel...")
    for path in (make_timeseries_figure(data), make_layer_figure(data), make_excel(data)):
        print("Saved:", path)
    print("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("ERROR:", error)
        sys.exit(1)
