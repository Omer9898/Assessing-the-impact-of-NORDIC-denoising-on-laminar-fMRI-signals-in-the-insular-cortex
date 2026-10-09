import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon

# ============================================================
# POSTERIOR INSULA tSNR: BEFORE vs AFTER NORDIC
# Separate left and right posterior-insula masks, plus bilateral summary.
# Change ONLY paths in this CONFIG section.
# All masks and 4-D files must be in the same anatomical space/grid.
# ============================================================

RUNS = [1, 2, 3]
OUTPUT_DIR = r"D:\NORDIC_Subject1_project\TSNR\Insular"
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
    1: r"D:\NORDIC_Subject1_project\Motion and Regstration with Nordic\final_warps_run1\rebuild_run1_4D.nii.gz",
    2: r"D:\NORDIC_Subject1_project\Motion and Regstration with Nordic\final_warps_run2\run2_4D.nii.gz",
    3: r"D:\NORDIC_Subject1_project\Motion and Regstration with Nordic\final_warps_run3\run3_4D.nii.gz",
}


def require_file(path, description):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found:\n{path}")


def load_4d(path, description):
    require_file(path, description)
    image = nib.load(path)
    if image.ndim != 4:
        raise ValueError(f"{description} must be 4-D, but is {image.shape}:\n{path}")
    return image


def load_mask(path, description):
    require_file(path, description)
    image = nib.load(path)
    if image.ndim != 3:
        raise ValueError(f"{description} must be 3-D, but is {image.shape}:\n{path}")
    mask = np.asanyarray(image.dataobj) > 0
    if not np.any(mask):
        raise ValueError(f"{description} contains no non-zero voxels.")
    return image, mask


def check_geometry(before_img, after_img, left_img, right_img, run):
    if before_img.shape != after_img.shape:
        raise ValueError(f"Run {run}: BEFORE shape {before_img.shape} != AFTER shape {after_img.shape}")
    if before_img.shape[:3] != left_img.shape or before_img.shape[:3] != right_img.shape:
        raise ValueError(f"Run {run}: data grid {before_img.shape[:3]} does not match one or both mask grids.")
    if not np.allclose(before_img.affine, after_img.affine, atol=1e-4):
        raise ValueError(f"Run {run}: BEFORE and AFTER affines differ.")
    if not np.allclose(before_img.affine, left_img.affine, atol=1e-4):
        raise ValueError("Data and LEFT posterior-insula mask affines differ.")
    if not np.allclose(before_img.affine, right_img.affine, atol=1e-4):
        raise ValueError("Data and RIGHT posterior-insula mask affines differ.")


def masked_tsnr(image, mask, text):
    n_timepoints = image.shape[3]
    n_voxels = int(mask.sum())
    total = np.zeros(n_voxels, dtype=np.float64)
    total_sq = np.zeros(n_voxels, dtype=np.float64)

    for t in range(n_timepoints):
        values = np.asanyarray(image.dataobj[..., t], dtype=np.float32)[mask]
        total += values
        total_sq += values * values
        if (t + 1) % 25 == 0 or t + 1 == n_timepoints:
            print(f"{text}: {t + 1}/{n_timepoints}", end="\r", flush=True)

    mean = total / n_timepoints
    variance = np.maximum(total_sq / n_timepoints - mean * mean, 0.0)
    std = np.sqrt(variance)
    valid = np.isfinite(mean) & np.isfinite(std) & (mean > 0) & (std > 0)
    tsnr = np.zeros(n_voxels, dtype=np.float32)
    tsnr[valid] = mean[valid] / std[valid]
    print()
    return tsnr, valid


def summarize(tsnr, common_valid):
    values = tsnr[common_valid]
    if values.size == 0:
        raise ValueError("No valid common voxels remain inside this ROI mask.")
    return float(values.mean()), float(np.median(values)), int(values.size)


def analyze_roi(before_img, after_img, mask, run, roi_name):
    print(f"Run {run}, {roi_name}: calculating BEFORE tSNR...")
    before_tsnr, before_valid = masked_tsnr(before_img, mask, f"Run {run} {roi_name} BEFORE")
    print(f"Run {run}, {roi_name}: calculating AFTER tSNR...")
    after_tsnr, after_valid = masked_tsnr(after_img, mask, f"Run {run} {roi_name} AFTER")
    common_valid = before_valid & after_valid
    before_mean, before_median, n_common = summarize(before_tsnr, common_valid)
    after_mean, after_median, _ = summarize(after_tsnr, common_valid)
    delta = after_mean - before_mean
    return {"run": run, "roi": roi_name, "n_common_mask_voxels": n_common, "before_mean_tsnr": before_mean, "after_mean_tsnr": after_mean, "delta_mean_tsnr": delta, "percent_change": 100 * delta / before_mean, "before_median_tsnr": before_median, "after_median_tsnr": after_median}


def plot_results(df, output_png):
    roi_order = [
        "Left posterior insula",
        "Right posterior insula",
        "Bilateral posterior insula",
    ]

    roi_titles = {
        "Left posterior insula": "Left posterior insula",
        "Right posterior insula": "Right posterior insula",
        "Bilateral posterior insula": "Bilateral posterior insula",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

    for ax, roi in zip(axes, roi_order):
        roi_df = df[df["roi"] == roi].sort_values("run")

        runs = roi_df["run"].astype(str).tolist()
        x = np.arange(len(runs))
        width = 0.36

        before_values = roi_df["before_mean_tsnr"].values
        after_values = roi_df["after_mean_tsnr"].values

        bars_before = ax.bar(
            x - width / 2,
            before_values,
            width,
            label="Before NORDIC",
            color="#6baed6",
        )

        bars_after = ax.bar(
            x + width / 2,
            after_values,
            width,
            label="After NORDIC",
            color="#31a354",
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f"Run {run}" for run in runs])
        ax.set_ylabel("Mean voxelwise tSNR")
        ax.set_title(roi_titles[roi])
        ax.grid(axis="y", alpha=0.25)

        if roi == "Left posterior insula":
            ax.legend(frameon=False)

        for before_bar, after_bar, pct in zip(
            bars_before,
            bars_after,
            roi_df["percent_change"].values,
        ):
            height = max(before_bar.get_height(), after_bar.get_height())

            ax.text(
                after_bar.get_x() + after_bar.get_width() / 2,
                height,
                f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(
        "Posterior-insula tSNR before vs after NORDIC",
        fontsize=16,
        y=1.03,
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    placeholders = list(BEFORE_PATHS.values()) + list(AFTER_PATHS.values()) + [OUTPUT_DIR, LEFT_POSTERIOR_INSULA_MASK, RIGHT_POSTERIOR_INSULA_MASK]
    if any("CHANGE_THIS" in p for p in placeholders):
        sys.exit("Set all paths in the CONFIG section before running.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    left_img, left_mask = load_mask(LEFT_POSTERIOR_INSULA_MASK, "Left posterior-insula mask")
    right_img, right_mask = load_mask(RIGHT_POSTERIOR_INSULA_MASK, "Right posterior-insula mask")
    bilateral_mask = left_mask | right_mask
    rows = []

    for run in RUNS:
        print(f"\n========== POSTERIOR INSULA — RUN {run} ==========")
        before_img = load_4d(BEFORE_PATHS[run], f"Run {run} BEFORE")
        after_img = load_4d(AFTER_PATHS[run], f"Run {run} AFTER")
        check_geometry(before_img, after_img, left_img, right_img, run)
        rows.append(analyze_roi(before_img, after_img, left_mask, run, "Left posterior insula"))
        rows.append(analyze_roi(before_img, after_img, right_mask, run, "Right posterior insula"))
        rows.append(analyze_roi(before_img, after_img, bilateral_mask, run, "Bilateral posterior insula"))

    df = pd.DataFrame(rows)
    table_path = os.path.join(OUTPUT_DIR, "posterior_insula_tsnr_by_run.csv")
    df.to_csv(table_path, index=False)

    stats_rows = []
    for roi, data in df.groupby("roi"):
        before = data["before_mean_tsnr"].to_numpy()
        after = data["after_mean_tsnr"].to_numpy()
        try:
            t_stat, t_p = ttest_rel(after, before)
        except Exception:
            t_stat, t_p = np.nan, np.nan
        try:
            w_stat, w_p = wilcoxon(after, before)
        except Exception:
            w_stat, w_p = np.nan, np.nan
        stats_rows.append({"roi": roi, "n_runs": len(data), "mean_before_tsnr": before.mean(), "mean_after_tsnr": after.mean(), "mean_percent_change": data["percent_change"].mean(), "paired_t_stat": t_stat, "paired_t_pvalue": t_p, "wilcoxon_stat": w_stat, "wilcoxon_pvalue": w_p})
    stats_path = os.path.join(OUTPUT_DIR, "posterior_insula_tsnr_statistics.csv")
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)

    figure_path = os.path.join(OUTPUT_DIR, "posterior_insula_tsnr.png")
    plot_results(df, figure_path)
    print("\nFINISHED")
    print(table_path)
    print(stats_path)
    print(figure_path)


if __name__ == "__main__":
    main()
