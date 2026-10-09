import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon

# ============================================================
# Whole-brain masked tSNR BEFORE vs AFTER NORDIC
# PLUS one QC overlay image: left + right posterior insula only.
# The insula masks are NOT used in the tSNR calculation.
# Change ONLY paths in this CONFIG section.
# ============================================================

RUNS = [1, 2, 3]
OUTPUT_DIR = r"D:\NORDIC_Subject1_project\TSNR\Whole Brain"

# Used ONLY for the whole-brain tSNR calculation.
WHOLE_BRAIN_MASK = r"D:\Downloads\brain_mask.nii.gz"

QC_ANATOMICAL_REFERENCE = r"D:\NORDIC_Subject1_project\7485_extracted\UNI_MoCo_MPRAGEised_biascorrected.nii"

LEFT_POSTERIOR_INSULA_MASK = r"D:\Downloads\pins left mask\roi_pInsleft_17.nii.gz"
RIGHT_POSTERIOR_INSULA_MASK = r"D:\Downloads\pins right mask\roi_pInsright_118.nii.gz"

BEFORE_PATHS = {
    1: r"D:\NORDIC_Subject1_project\Motion and Regstration without Nordic\magnitude_run1_Warped-to-Anat_4D.nii.gz",  
    2: r"D:\NORDIC_Subject1_project\Motion and Regstration without Nordic\magnitude_run2_Warped-to-Anat_4D.nii.gz",  
    3: r"D:\NORDIC_Subject1_project\Motion and Regstration without Nordic\magnitude_run3_Warped-to-Anat_4D.nii.gz",  
}

# Your merged, warped NORDIC 4-D files: do not change these unless the
# filenames/folders where you saved the merged files are different.
AFTER_PATHS = {
    1: r"E:\layerfmri\LAYER_s7468_warpednordic\sec-02\layer\func\RUN1\run1_4D.nii.gz",
    2: r"E:\layerfmri\LAYER_s7468_warpednordic\sec-02\layer\func\RUN2\run2_4D.nii.gz",
    3: r"E:\layerfmri\LAYER_s7468_warpednordic\sec-02\layer\func\RUN3\run3_4D.nii.gz",
}



def require_file(path, description):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found:\n{path}")


def load_3d(path, description):
    require_file(path, description)
    image = nib.load(path)
    if image.ndim != 3:
        raise ValueError(f"{description} must be 3-D, but is {image.shape}:\n{path}")
    return image


def load_4d(path, description):
    require_file(path, description)
    image = nib.load(path)
    if image.ndim != 4:
        raise ValueError(f"{description} must be 4-D, but is {image.shape}:\n{path}")
    return image


def binary_mask(image, description):
    mask = np.asanyarray(image.dataobj) > 0
    if not np.any(mask):
        raise ValueError(f"{description} has no non-zero voxels.")
    return mask


def check_whole_brain_geometry(before_img, after_img, brain_mask_img, run):
    if before_img.shape != after_img.shape:
        raise ValueError(f"Run {run}: BEFORE shape {before_img.shape} != AFTER shape {after_img.shape}")
    if before_img.shape[:3] != brain_mask_img.shape:
        raise ValueError(f"Run {run}: data grid {before_img.shape[:3]} != brain-mask grid {brain_mask_img.shape}")
    if not np.allclose(before_img.affine, after_img.affine, atol=1e-4):
        raise ValueError(f"Run {run}: BEFORE and AFTER affines differ.")
    if not np.allclose(before_img.affine, brain_mask_img.affine, atol=1e-4):
        raise ValueError(f"Run {run}: data and whole-brain-mask affines differ.")


def masked_tsnr(image, mask, text):
    n_timepoints = image.shape[3]
    n_mask_voxels = int(mask.sum())
    total = np.zeros(n_mask_voxels, dtype=np.float64)
    total_sq = np.zeros(n_mask_voxels, dtype=np.float64)

    for t in range(n_timepoints):
        values = np.asanyarray(image.dataobj[..., t], dtype=np.float32)[mask]
        total += values
        total_sq += values * values
        if (t + 1) % 25 == 0 or t + 1 == n_timepoints:
            print(f"{text}: {t + 1}/{n_timepoints}", end="\r", flush=True)

    temporal_mean = total / n_timepoints
    temporal_variance = np.maximum(total_sq / n_timepoints - temporal_mean * temporal_mean, 0.0)
    temporal_std = np.sqrt(temporal_variance)
    valid = np.isfinite(temporal_mean) & np.isfinite(temporal_std) & (temporal_mean > 0) & (temporal_std > 0)
    tsnr = np.zeros(n_mask_voxels, dtype=np.float32)
    tsnr[valid] = temporal_mean[valid] / temporal_std[valid]
    print()
    return tsnr, valid


def summary(tsnr, valid):
    values = tsnr[valid]
    if values.size == 0:
        raise ValueError("No valid voxels remain inside the whole-brain mask.")
    return float(values.mean()), float(np.median(values)), int(values.size)


def save_whole_brain_plot(df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    ax = axes[0]
    for _, row in df.iterrows():
        color = "#2ca02c" if row["percent_change"] >= 0 else "#d62728"
        ax.plot([0, 1], [row["before_mean_tsnr"], row["after_mean_tsnr"]], marker="o", linewidth=2.2, color=color)
        ax.text(1.03, row["after_mean_tsnr"], f"Run {int(row['run'])}", va="center")
    ax.plot([0, 1], [df["before_mean_tsnr"].mean(), df["after_mean_tsnr"].mean()], marker="s", color="black", linewidth=3.4, label="Mean across runs")
    ax.set_xticks([0, 1], ["Before NORDIC", "After NORDIC"])
    ax.set_ylabel("Mean voxelwise tSNR in whole-brain mask")
    ax.set_title("Whole-brain masked tSNR")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in df["percent_change"]]
    bars = ax.bar(df["run"].astype(str), df["percent_change"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Run")
    ax.set_ylabel("Change in mean tSNR (%)")
    ax.set_title("Percent change after NORDIC")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, df["percent_change"]):
        ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.1f}%", ha="center", va="bottom" if value >= 0 else "top")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_one_insula_qc_overlay(anat_img, left_img, right_img, output_path):
    if anat_img.shape != left_img.shape or anat_img.shape != right_img.shape:
        raise ValueError("QC anatomy and left/right insula masks must have the same 3-D shape.")
    if not np.allclose(anat_img.affine, left_img.affine, atol=1e-4) or not np.allclose(anat_img.affine, right_img.affine, atol=1e-4):
        raise ValueError("QC anatomy and left/right insula masks must have matching affines.")

    anatomy = anat_img.get_fdata(dtype=np.float32)
    left_mask = binary_mask(left_img, "Left posterior-insula mask")
    right_mask = binary_mask(right_img, "Right posterior-insula mask")
    bilateral = left_mask | right_mask
    z = int(np.argmax(np.sum(bilateral, axis=(0, 1))))

    background = np.rot90(anatomy[:, :, z])
    left_slice = np.rot90(left_mask[:, :, z])
    right_slice = np.rot90(right_mask[:, :, z])
    finite = background[np.isfinite(background)]
    vmin, vmax = np.percentile(finite, [1, 99])

    fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
    ax.imshow(background, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    ax.imshow(np.ma.masked_where(~left_slice, left_slice), cmap="Blues", alpha=0.75, origin="lower", interpolation="nearest")
    ax.imshow(np.ma.masked_where(~right_slice, right_slice), cmap="autumn", alpha=0.75, origin="lower", interpolation="nearest")
    ax.set_title("Left + right posterior-insula ROI overlay", fontsize=13)
    ax.text(0.02, 0.03, "Blue = left pIns\nYellow/red = right pIns", color="white", transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round", facecolor="black", alpha=0.65, edgecolor="none"))
    ax.text(0.02, 0.93, f"z = {z}", color="white", transform=ax.transAxes, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def main():
    all_paths = [OUTPUT_DIR, WHOLE_BRAIN_MASK, LEFT_POSTERIOR_INSULA_MASK, RIGHT_POSTERIOR_INSULA_MASK, QC_ANATOMICAL_REFERENCE] + list(BEFORE_PATHS.values()) + list(AFTER_PATHS.values())
    if any("CHANGE_THIS" in path for path in all_paths):
        sys.exit("Set every path in the CONFIG section before running.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # One insula overlay only: it is separate from the whole-brain tSNR analysis.
    qc_anat = load_3d(QC_ANATOMICAL_REFERENCE, "QC anatomical reference")
    left_insula = load_3d(LEFT_POSTERIOR_INSULA_MASK, "Left posterior-insula mask")
    right_insula = load_3d(RIGHT_POSTERIOR_INSULA_MASK, "Right posterior-insula mask")
    insula_overlay_path = os.path.join(OUTPUT_DIR, "left_right_posterior_insula_QC_overlay.jpg")
    save_one_insula_qc_overlay(qc_anat, left_insula, right_insula, insula_overlay_path)
    print(f"Saved one insula QC image:\n{insula_overlay_path}")

    brain_mask_img = load_3d(WHOLE_BRAIN_MASK, "Whole-brain mask")
    brain_mask = binary_mask(brain_mask_img, "Whole-brain mask")
    rows = []

    for run in RUNS:
        print(f"\n========== WHOLE BRAIN — RUN {run} ==========")
        before_img = load_4d(BEFORE_PATHS[run], f"Run {run} BEFORE")
        after_img = load_4d(AFTER_PATHS[run], f"Run {run} AFTER")
        check_whole_brain_geometry(before_img, after_img, brain_mask_img, run)
        before_tsnr, before_valid = masked_tsnr(before_img, brain_mask, f"Run {run} BEFORE")
        after_tsnr, after_valid = masked_tsnr(after_img, brain_mask, f"Run {run} AFTER")
        common_valid = before_valid & after_valid
        before_mean, before_median, n_common = summary(before_tsnr, common_valid)
        after_mean, after_median, _ = summary(after_tsnr, common_valid)
        delta = after_mean - before_mean
        rows.append({"run": run, "n_common_brain_voxels": n_common, "before_mean_tsnr": before_mean, "after_mean_tsnr": after_mean, "delta_mean_tsnr": delta, "percent_change": 100 * delta / before_mean, "before_median_tsnr": before_median, "after_median_tsnr": after_median})

    df = pd.DataFrame(rows)
    table_path = os.path.join(OUTPUT_DIR, "whole_brain_masked_tsnr_by_run.csv")
    df.to_csv(table_path, index=False)

    before = df["before_mean_tsnr"].to_numpy()
    after = df["after_mean_tsnr"].to_numpy()
    try:
        t_stat, t_p = ttest_rel(after, before)
    except Exception:
        t_stat, t_p = np.nan, np.nan
    try:
        w_stat, w_p = wilcoxon(after, before)
    except Exception:
        w_stat, w_p = np.nan, np.nan
    stats_path = os.path.join(OUTPUT_DIR, "whole_brain_masked_tsnr_statistics.csv")
    pd.DataFrame([{"n_runs": len(df), "mean_before_tsnr": before.mean(), "mean_after_tsnr": after.mean(), "mean_percent_change": df["percent_change"].mean(), "paired_t_stat": t_stat, "paired_t_pvalue": t_p, "wilcoxon_stat": w_stat, "wilcoxon_pvalue": w_p}]).to_csv(stats_path, index=False)

    whole_brain_plot_path = os.path.join(OUTPUT_DIR, "whole_brain_masked_tsnr.png")
    save_whole_brain_plot(df, whole_brain_plot_path)

    print("\nFINISHED")
    print(table_path)
    print(stats_path)
    print(whole_brain_plot_path)
    print(insula_overlay_path)


if __name__ == "__main__":
    main()
