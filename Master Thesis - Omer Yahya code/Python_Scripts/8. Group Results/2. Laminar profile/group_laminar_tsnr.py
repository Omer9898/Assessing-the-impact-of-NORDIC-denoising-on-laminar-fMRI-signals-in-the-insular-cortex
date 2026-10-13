"""Descriptive group tSNR across sampled cortical depths, paired NORDIC comparison.
Run: python group_laminar_tsnr.py
Dependencies: numpy pandas h5py matplotlib
No GLM/SPM, spatial smoothing, temporal filtering, detrending or inference.
"""
from pathlib import Path
from datetime import datetime
import json
import re
import traceback

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SUBJECTS = {
    "7485": Path(
        r"D:\NORDIC_Subject1_project\Laminarprofile"
    ),
    "7468": Path(
        r"E:\layerfmri\LAYER_s7468_warpednordic\codes\Laminarprofile\Laminar"
    ),
    "7469": Path(
        r"E:\layerfmri\LAYER_s7469_warpenordic\codes\laminar profile"
    ),
    "7482": Path(
        r"E:\layerfmri\LAYER_s7482_Warpenordic\codes\Laminar profile"
    ),
    "7484": Path(
        r"E:\layerfmri\LAYER_s7484_warpenordic\codes\Laminarprofile"
    ),
}

OUTPUT_ROOT = Path(r"C:\Users\L E N O V O\OneDrive\Desktop\Group  Analysis\laminar profile")

# Optional exact file overrides; ordinarily leave empty.
# Example: FILE_OVERRIDES[("7485", "Nordic")] = Path(r"D:\...\file.h5")
FILE_OVERRIDES = {}
# ==================================================================
TIME_BLOCK = 32
MAX_INVALID_FRACTION = 0.05
DPI = 300
CONDITIONS = ("Without Nordic", "Nordic")
SIDES = ("left", "right")
COLORS = {"Without Nordic": "#D97732", "Nordic": "#187AA5"}
SUBJECT_COLORS = ["#4477AA", "#EE6677", "#228833", "#AA3377", "#CCBB44"]
METHOD = "Sampled time series mean / temporal sample SD (ddof=1), no additional detrending"


def decode(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def input_path(subject, condition):
    key = (subject, condition)
    if key in FILE_OVERRIDES:
        return Path(FILE_OVERRIDES[key])
    folder, token = ("With", "with_NORDIC") if condition == "Nordic" else ("Without", "without_NORDIC")
    return SUBJECTS[subject] / "1. Sampling" / folder / f"pIns_laminar_sampling_{subject}_{token}.h5"


def depth_values(handle, count, warnings):
    if "depth_fractions" in handle:
        values = np.asarray(handle["depth_fractions"][:], dtype=float)
    else:
        if count != 20:
            raise ValueError("Missing depth_fractions and depth count is not 20")
        values = np.linspace(0, 1, count)
        warnings.append("Missing depth_fractions: inferred linspace(0,1,20) from supplied sampling scripts")
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError("Invalid depth vector")
    if np.any(np.diff(values) <= 0) or values[0] < 0 or values[-1] > 1:
        raise ValueError("Depths must increase within [0,1]")
    return values


def stream_moments(dataset, start, stop):
    shape = dataset.shape[:2]
    mean = np.zeros(shape, dtype=np.float64)
    m2 = np.zeros(shape, dtype=np.float64)
    valid = np.ones(shape, dtype=bool)
    n = 0
    for begin in range(start, stop, TIME_BLOCK):
        end = min(begin + TIME_BLOCK, stop)
        data = np.asarray(dataset[:, :, begin:end], dtype=np.float64)
        finite = np.isfinite(data)
        if np.any(np.all(finite & (data == 0), axis=(0, 1))):
            raise ValueError(f"Entire hemisphere is zero at a time point in [{begin},{end})")
        if np.any(np.all(~finite, axis=(0, 1))):
            raise ValueError(f"Entire hemisphere is non-finite at a time point in [{begin},{end})")
        valid &= finite.all(axis=2)
        data[~finite] = 0.0
        k = data.shape[2]
        batch_mean = data.mean(axis=2)
        centered = data - batch_mean[:, :, None]
        batch_m2 = np.einsum("ijk,ijk->ij", centered, centered)
        delta = batch_mean - mean
        m2 += batch_m2 + delta * delta * (n * k / (n + k))
        mean += delta * (k / (n + k))
        n += k
    if n < 2:
        raise ValueError("At least two time points required")
    sd = np.sqrt(np.maximum(m2, 0) / (n - 1))
    valid &= np.isfinite(mean) & np.isfinite(sd) & (mean > 0) & (sd > 0)
    return mean, sd, valid


def scan_indices(handle):
    if "volume_files" not in handle:
        raise ValueError("Missing volume_files: cannot verify temporal filename ordering")
    names = [decode(v) for v in handle["volume_files"][:]]
    indices = []
    for name in names:
        match = re.search(r"_(\d+)_Warped-to-Anat\.nii(?:\.gz)?$", name, re.I)
        if match is None:
            raise ValueError(f"Cannot extract numeric volume index: {name}")
        indices.append(int(match.group(1)))
    return names, np.array(indices, dtype=int)


def inspect_pair(subject, handles):
    warnings = []
    info = {"subject": subject, "warnings": warnings, "hemispheres": {}, "conditions": {}}
    reference_depth = None
    reference_runs = None
    time_indices = {}
    for condition, h in handles.items():
        expected = "with_NORDIC" if condition == "Nordic" else "without_NORDIC"
        if decode(h.attrs.get("subject", "")) != subject:
            raise ValueError(f"{subject}/{condition}: subject metadata mismatch")
        if decode(h.attrs.get("condition", "")) != expected:
            raise ValueError(f"{subject}/{condition}: condition metadata mismatch")
        if "complete" in h.attrs and not bool(h.attrs["complete"]):
            raise ValueError(f"{subject}/{condition}: complete=False")
        if "complete" not in h.attrs:
            warnings.append(f"{condition}: complete marker absent; direct data checks used")
        direction = decode(h.attrs.get("sampling_direction", "")).lower()
        if "white" not in direction or "pial" not in direction:
            raise ValueError(f"{subject}/{condition}: missing recognizable depth direction")
        if "run_lengths" not in h:
            raise ValueError("Missing run_lengths")
        runs_raw = np.asarray(h["run_lengths"][:])
        if runs_raw.ndim != 1 or not np.all(runs_raw == runs_raw.astype(int)):
            raise ValueError("Invalid run lengths")
        runs = runs_raw.astype(int)
        if len(runs) == 0 or np.any(runs < 2):
            raise ValueError("Invalid run lengths")
        names, indices = scan_indices(h)
        if len(names) != runs.sum():
            raise ValueError("volume_files count differs from run lengths")
        edges = np.r_[0, np.cumsum(runs)]
        for a, b in zip(edges[:-1], edges[1:]):
            if not np.all(np.diff(indices[a:b]) == 1):
                raise ValueError("Non-sequential volume indices within a run")
        time_indices[condition] = indices
        if "written_volumes" in h and not np.all(h["written_volumes"][:]):
            raise ValueError("Unwritten volumes detected")
        if "n_written_volumes" in h.attrs and int(h.attrs["n_written_volumes"]) != runs.sum():
            raise ValueError("n_written_volumes mismatch")
        for side in SIDES:
            ds = h[f"{side}/layeractivation"]
            if ds.ndim != 3 or ds.shape[0] == 0 or ds.shape[2] != runs.sum():
                raise ValueError(f"Invalid {side} dataset dimensions")
            d = depth_values(h, ds.shape[1], warnings)
            if reference_depth is None:
                reference_depth, reference_runs = d, runs
            if not np.allclose(d, reference_depth, atol=1e-10, rtol=0) or not np.array_equal(runs, reference_runs):
                raise ValueError("Depth/run mismatch within subject")
        info["conditions"][condition] = {
            "file": str(input_path(subject, condition)),
            "first_scan": names[0], "last_scan": names[-1],
            "t1_reference": decode(h.attrs.get("t1_reference", "unknown")),
            "sampling_direction": direction,
        }
    a, b = (handles[c] for c in CONDITIONS)
    if not np.array_equal(time_indices[CONDITIONS[0]], time_indices[CONDITIONS[1]]):
        raise ValueError("Pre/post numeric volume ordering differs")
    warnings.append("Numeric ordering checked; NORDIC run-folder identity still requires correct input provenance")
    for side in SIDES:
        if a[f"{side}/layeractivation"].shape != b[f"{side}/layeractivation"].shape:
            raise ValueError(f"{side}: paired shape mismatch")
        va = np.asarray(a[f"{side}/surface_vertex_indices"][:])
        vb = np.asarray(b[f"{side}/surface_vertex_indices"][:])
        ncol = a[f"{side}/layeractivation"].shape[0]
        if va.shape != (ncol,) or vb.shape != (ncol,) or not np.array_equal(va, vb) or len(np.unique(va)) != ncol:
            raise ValueError(f"{side}: vertex IDs/order differ or contain duplicates")
        coordinate_key = f"{side}/sampling_coordinates_vox"
        coords_status = "not available in both files"
        if coordinate_key in a and coordinate_key in b:
            ca, cb = np.asarray(a[coordinate_key][:]), np.asarray(b[coordinate_key][:])
            if ca.shape != (ncol, len(reference_depth), 3) or cb.shape != ca.shape or not np.isfinite(ca).all() or not np.isfinite(cb).all() or not np.allclose(ca, cb, atol=1e-5, rtol=0):
                raise ValueError(f"{side}: coordinate mismatch")
            coords_status = "paired coordinates matched"
        else:
            warnings.append(f"{side}: coordinate comparison unavailable; paired vertex IDs matched")
        info["hemispheres"][side] = {"columns": ncol, "coordinate_check": coords_status}
    info["run_lengths"] = reference_runs.tolist()
    info["depths"] = reference_depth.tolist()
    return reference_depth, reference_runs, info


def summarize_subjects(run_df):
    cols = ["subject", "hemisphere", "condition", "depth_index", "depth_fraction"]
    return run_df.groupby(cols, sort=False, as_index=False).agg(
        tsnr=("tsnr", "mean"), mean_signal=("mean_signal", "mean"),
        temporal_sd=("temporal_sd", "mean"), runs=("run", "nunique"))


def group_summary(subject_df):
    cols = ["hemisphere", "condition", "depth_index", "depth_fraction"]
    result = subject_df.groupby(cols, as_index=False).agg(
        mean=("tsnr", "mean"), sd=("tsnr", "std"), n=("subject", "nunique"))
    result["sem"] = result["sd"] / np.sqrt(result["n"])
    return result


def paired_differences(subject_df):
    keys = ["subject", "hemisphere", "depth_index", "depth_fraction"]
    wide = subject_df.pivot(index=keys, columns="condition", values="tsnr").reset_index()
    if wide[list(CONDITIONS)].isna().any().any():
        raise ValueError("Incomplete paired subject profiles")
    wide["difference"] = wide["Nordic"] - wide["Without Nordic"]
    return wide


def save_figure(fig, folder, stem):
    fig.savefig(folder / (stem + ".png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / (stem + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_depth(ax, ylabel="tSNR"):
    ax.set(xlim=(0, 1), xlabel="Cortical depth (white matter to pial)", ylabel=ylabel)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)


def draw_figures(subject_df, group_df, diff_df, folder):
    ids = list(SUBJECTS)
    palette = dict(zip(ids, SUBJECT_COLORS))
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    for ax, side in zip(axes, SIDES):
        for condition in CONDITIONS:
            for subject in ids:
                d = subject_df.query("hemisphere == @side and condition == @condition and subject == @subject").sort_values("depth_fraction")
                ax.plot(d.depth_fraction, d.tsnr, color=COLORS[condition], lw=.8, alpha=.24)
            d = group_df.query("hemisphere == @side and condition == @condition").sort_values("depth_fraction")
            x, y, sd = d.depth_fraction.to_numpy(), d["mean"].to_numpy(), d.sd.to_numpy()
            ax.fill_between(x, y-sd, y+sd, color=COLORS[condition], alpha=.14)
            ax.plot(x, y, color=COLORS[condition], lw=2.8, label=condition)
        style_depth(ax)
        ax.set_title(side.title() + " posterior insula")
        ax.legend(frameon=False)
    fig.suptitle("Group tSNR-depth | n=5 | mean +/- between-subject SD\nThin curves: individual subjects; descriptive analysis", fontsize=13)
    save_figure(fig, folder, "Figure_01_Group_tSNR_depth")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    for ax, side in zip(axes, SIDES):
        for subject in ids:
            d = diff_df.query("hemisphere == @side and subject == @subject").sort_values("depth_fraction")
            ax.plot(d.depth_fraction, d.difference, color=palette[subject], lw=1.5, alpha=.8, label=subject)
        d = diff_df.query("hemisphere == @side").groupby("depth_fraction").difference.agg(["mean", "std"])
        x, y, sd = d.index.to_numpy(), d["mean"].to_numpy(), d["std"].to_numpy()
        ax.fill_between(x, y-sd, y+sd, color="black", alpha=.09)
        ax.plot(x, y, color="black", lw=2.8, label="Group mean")
        ax.axhline(0, color="gray", ls="--", lw=1)
        style_depth(ax, "tSNR difference (Nordic - Without Nordic)")
        ax.set_title(side.title() + " posterior insula")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Paired depth differences | shading: between-subject SD", fontsize=13)
    save_figure(fig, folder, "Figure_02_Paired_depth_differences")

    fig, axes = plt.subplots(len(ids), 2, figsize=(11, 3*len(ids)), sharex=True, constrained_layout=True)
    for row, subject in enumerate(ids):
        for col, side in enumerate(SIDES):
            ax = axes[row, col]
            for condition in CONDITIONS:
                d = subject_df.query("subject == @subject and hemisphere == @side and condition == @condition").sort_values("depth_fraction")
                ax.plot(d.depth_fraction, d.tsnr, lw=2, color=COLORS[condition], label=condition)
            style_depth(ax)
            ax.set_title(f"Subject {subject} | {side.title()}")
            if row == 0:
                ax.legend(frameon=False)
    fig.suptitle("Individual tSNR profiles | run-averaged, all ROI columns", fontsize=14)
    save_figure(fig, folder, "Figure_03_All_individual_profiles")

    averages = subject_df.groupby(["subject", "hemisphere", "condition"], as_index=False).tsnr.mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, constrained_layout=True)
    for ax, side in zip(axes, SIDES):
        for subject in ids:
            d = averages.query("subject == @subject and hemisphere == @side").set_index("condition").reindex(CONDITIONS)
            ax.plot([0, 1], d.tsnr, "o-", color=palette[subject], lw=1.6, ms=7, label=subject)
        d = averages.query("hemisphere == @side").groupby("condition").tsnr.agg(["mean", "std"]).reindex(CONDITIONS)
        ax.errorbar([-.06, .94], d["mean"], yerr=d["std"], fmt="D", color="black", capsize=5, ms=7, label="Mean +/- SD")
        ax.set_xticks([0, 1], CONDITIONS)
        ax.set(xlim=(-.3, 1.3), ylabel="Depth-averaged tSNR", title=side.title()+" posterior insula")
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Paired subject comparison | equal average across 20 sampled depths", fontsize=13)
    save_figure(fig, folder, "Figure_04_Paired_subject_summary")
    averages.to_csv(folder.parent / "subject_depth_averaged_tSNR.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    limit = max(float(np.abs(diff_df.difference).max()), 1e-9)
    for ax, side in zip(axes, SIDES):
        d = diff_df.query("hemisphere == @side").pivot(index="subject", columns="depth_fraction", values="difference").reindex(ids)
        image = ax.imshow(d.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        positions = [0, 5, 10, 15, len(d.columns)-1]
        ax.set_xticks(positions, [f"{d.columns[i]:.2f}" for i in positions])
        ax.set_yticks(range(len(ids)), ids)
        ax.set(xlabel="Sampled cortical depth", ylabel="Subject", title=side.title()+" posterior insula")
    fig.colorbar(image, ax=list(axes), label="tSNR difference (Nordic - Without Nordic)", shrink=.9)
    fig.suptitle("Individual depth-difference heatmap | shared symmetric color scale", fontsize=13)
    save_figure(fig, folder, "Figure_05_Depth_difference_heatmap")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    for row, metric in enumerate(["mean_signal", "temporal_sd"]):
        for col, side in enumerate(SIDES):
            ax = axes[row, col]
            for condition in CONDITIONS:
                d = subject_df.query("hemisphere == @side and condition == @condition").groupby("depth_fraction")[metric].mean()
                ax.plot(d.index, d.values, color=COLORS[condition], lw=2.3, label=condition)
            label = "Mean signal (input intensity units)" if metric == "mean_signal" else "Temporal SD (input intensity units)"
            style_depth(ax, label)
            ax.set_title(side.title()+" | "+metric.replace("_", " "))
            ax.legend(frameon=False)
    fig.suptitle("Signal / temporal-variability QC | descriptive means across subjects\nInput scaling may differ between subjects; not activation measures", fontsize=12)
    save_figure(fig, folder, "Figure_06_Signal_and_temporal_SD_QC")


def main():
    if len(SUBJECTS) != 5:
        raise ValueError("This configured analysis requires exactly five subjects")
    for subject in SUBJECTS:
        for condition in CONDITIONS:
            path = input_path(subject, condition)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {subject}/{condition}: {path}")
    folder = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    figures = folder / "figures"
    figures.mkdir()
    rows, qc_rows, infos = [], [], []
    global_depth = None
    try:
        for subject in SUBJECTS:
            print(f"\nSUBJECT {subject}", flush=True)
            with h5py.File(input_path(subject, CONDITIONS[0]), "r") as a, h5py.File(input_path(subject, CONDITIONS[1]), "r") as b:
                handles = dict(zip(CONDITIONS, [a, b]))
                depths, runs, info = inspect_pair(subject, handles)
                infos.append(info)
                if global_depth is None:
                    global_depth = depths
                if depths.shape != global_depth.shape or not np.allclose(depths, global_depth, atol=1e-10, rtol=0):
                    raise ValueError("Different cortical depth grids between subjects")
                edges = np.r_[0, np.cumsum(runs)]
                for side in SIDES:
                    print(f"  {side}: {a[f'{side}/layeractivation'].shape[0]} columns", flush=True)
                    for run, (start, stop) in enumerate(zip(edges[:-1], edges[1:]), 1):
                        stats = {c: stream_moments(handles[c][f"{side}/layeractivation"], int(start), int(stop)) for c in CONDITIONS}
                        paired_valid = stats[CONDITIONS[0]][2] & stats[CONDITIONS[1]][2]
                        ncols = paired_valid.shape[0]
                        counts = paired_valid.sum(axis=0)
                        fraction = 1 - counts/ncols
                        if np.any(counts == 0) or np.any(fraction > MAX_INVALID_FRACTION):
                            raise ValueError(f"{subject}/{side}/run{run}: excessive invalid paired columns; max fraction={fraction.max():.4f}")
                        for j, depth in enumerate(depths):
                            qc_rows.append({"subject": subject, "hemisphere": side, "run": run,
                                "depth_index": j, "depth_fraction": depth, "total_columns": ncols,
                                "paired_valid_columns": int(counts[j]), "excluded_fraction": float(fraction[j])})
                            keep = paired_valid[:, j]
                            for c in CONDITIONS:
                                mean, sd, _ = stats[c]
                                rows.append({"subject": subject, "hemisphere": side, "condition": c,
                                    "run": run, "depth_index": j, "depth_fraction": float(depth),
                                    "tsnr": float(np.mean(mean[keep, j]/sd[keep, j])),
                                    "mean_signal": float(np.mean(mean[keep, j])),
                                    "temporal_sd": float(np.mean(sd[keep, j])),
                                    "paired_valid_columns": int(counts[j])})
                        print(f"    run {run}: checked and calculated", flush=True)
                        del stats
        run_df = pd.DataFrame(rows)
        subject_df = summarize_subjects(run_df)
        group_df = group_summary(subject_df)
        diff_df = paired_differences(subject_df)
        if not np.all(group_df.n == 5):
            raise ValueError("Group summary does not include five subjects everywhere")
        diff_group = diff_df.groupby(["hemisphere", "depth_index", "depth_fraction"], as_index=False).agg(
            mean_difference=("difference", "mean"), sd_difference=("difference", "std"), n=("subject", "nunique"))
        for name, df in [("tSNR_by_run", run_df), ("tSNR_by_subject", subject_df),
                         ("tSNR_group_summary", group_df), ("paired_depth_differences", diff_df),
                         ("paired_difference_group_summary", diff_group), ("paired_validity_QC", pd.DataFrame(qc_rows))]:
            df.to_csv(folder/(name+".csv"), index=False)
        provenance = {"method": METHOD, "analysis": "All sampled pIns ROI columns, left/right separately",
            "spatial_smoothing_added": False, "prior_input_smoothing": "Not verified; do not label as 4 mm smoothed",
            "temporal_filtering_added": False, "detrending_added": False, "SPM_used": False,
            "group_weighting": "Equal subjects; equal runs within subject; mean valid columnwise tSNR within each run/depth",
            "bands": "Between-subject sample SD, not confidence intervals or inferential significance",
            "invalid_policy": "Same paired column/depth mask per run; nonfinite samples, nonpositive mean or zero SD excluded in both conditions",
            "max_invalid_fraction": MAX_INVALID_FRACTION, "subjects": infos}
        (folder/"provenance_and_QC.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print("\nDrawing six figures...", flush=True)
        draw_figures(subject_df, group_df, diff_df, figures)
        methods = (
            "DESCRIPTIVE GROUP tSNR-DEPTH ANALYSIS\n\n"
            "Five subjects. Sampled cortical depths are not independently resolved histological layers.\n"
            "For each run, column and depth: temporal mean / sample temporal SD (ddof=1).\n"
            "No additional detrending, temporal filtering or smoothing is performed.\n"
            "Task, physiological and drift contributions remain in temporal SD.\n"
            "Columnwise tSNR is averaged, NOT tSNR of the ROI-averaged time course.\n"
            "Invalid column-depth time courses are excluded symmetrically in both conditions per run.\n"
            "Runs are averaged equally within subject; subjects are averaged equally at group level.\n"
            "Depth-averaged summaries are arithmetic averages across sampled depths, not volume-weighted cortex estimates.\n"
            "Figures show mean +/- between-subject SD. No p-values, significance tests or confidence intervals.\n"
            "Native-space vertex IDs are paired within subject, not across subjects.\n"
            "Prior spatial smoothing of source images has not been established.\n"
            "Missing depth metadata may be reconstructed from the supplied 20-depth sampling scripts and is logged.\n"
            "Matched metadata and file order do not independently certify anatomical registration or acquisition provenance.\n"
            "Higher tSNR alone is not evidence of preserved task responses or depth specificity.\n"
        )
        (folder/"METHODS.txt").write_text(methods, encoding="utf-8")
        (folder/"COMPLETE.json").write_text(json.dumps({"complete": True, "n_subjects": 5, "n_figures": 6}), encoding="utf-8")
        print(f"\nDONE. Results: {folder}", flush=True)
    except Exception:
        (folder/"FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
