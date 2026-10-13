"""Complete tSNR analysis and thesis figure.
Preserved subject paths. Mean +/- SD labels above ALL paired lines.
Install: python -m pip install numpy pandas matplotlib openpyxl
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= YOUR EXISTING PATHS =================
SUBJECT_PATHS = {
    "Subject01": {
        "whole_brain": Path(r"E:\layerfmri\LAYER_s7469_warpenordic\codes\TSNR\Whole Brain"),
        "posterior_insula": Path(r"E:\layerfmri\LAYER_s7469_warpenordic\codes\TSNR\Posterior Insular"),
    },
    "Subject02": {
        "whole_brain": Path(r"D:\NORDIC_Subject1_project\TSNR\Whole Brain"),
        "posterior_insula": Path(r"D:\NORDIC_Subject1_project\TSNR\Insular"),
    },
    "Subject03": {
        "whole_brain": Path(r"E:\layerfmri\LAYER_s7468_warpednordic\codes\TSNR\Whole Brain"),
        "posterior_insula": Path(r"E:\layerfmri\LAYER_s7468_warpednordic\codes\TSNR\Posterior Insular"),
    },
    "Subject04": {
        "whole_brain": Path(r"E:\layerfmri\LAYER_s7482_Warpenordic\codes\TSNR\Whole Brain"),
        "posterior_insula": Path(r"E:\layerfmri\LAYER_s7482_Warpenordic\codes\TSNR\Posterior Insular"),
    },
    "Subject05": {
        "whole_brain": Path(r"E:\layerfmri\LAYER_s7484_warpenordic\codes\TSNR\Whole Brain"),
        "posterior_insula": Path(r"E:\layerfmri\LAYER_s7484_warpenordic\codes\TSNR\Posterior Insular"),
    },
}
# Two spaces between Group and Analysis, as in your attached code.
OUTPUT_DIR = Path(r"C:\Users\L E N O V O\OneDrive\Desktop\Group  Analysis\Tsnr\old 3")
# ======================================================

REGIONS = {
    "Whole brain": ("whole_brain", "whole_brain_masked_tsnr_by_run.csv"),
    "Posterior insula": ("posterior_insula", "posterior_insula_tsnr_by_run.csv"),
}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
JITTER_WIDTH = 0.17
FIGURE_STEM = "FIGURE_GROUP_TSNR_JITTER_MEAN_SD_FINAL"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelsize": 13, "axes.titlesize": 15,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def normalize(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def choose_column(df, aliases, condition, path):
    lookup = {normalize(c): c for c in df.columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    words = {"before", "pre", "without"} if condition == "without" else {"after", "post", "with"}
    candidates = []
    for c in df.columns:
        tokens = set(normalize(c).split("_"))
        if tokens & words and not tokens & {"median", "delta", "percent", "change", "sd", "sem", "std"}:
            candidates.append(c)
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Cannot uniquely identify mean tSNR {condition} NORDIC in {path}. Columns: {list(df.columns)}")


def read_csv(folder, filename, subject, region):
    path = folder / filename
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    b = choose_column(df, ["before_mean_tsnr", "without_nordic_mean_tsnr", "without_nordic",
        "before_nordic", "before", "mean_tsnr_before", "tsnr_before_mean", "tsnr_before",
        "before_mean", "pre_nordic", "pre"], "without", path)
    a = choose_column(df, ["after_mean_tsnr", "with_nordic_mean_tsnr", "with_nordic",
        "after_nordic", "after", "mean_tsnr_after", "tsnr_after_mean", "tsnr_after",
        "after_mean", "post_nordic", "post"], "with", path)
    run_columns = [c for c in df.columns if normalize(c) in {"run", "run_number", "run_id"}]
    if len(run_columns) != 1:
        raise ValueError(f"Missing or ambiguous Run column in {path}: {list(df.columns)}")
    for c in df.columns:
        if normalize(c) in {"region", "roi", "hemisphere", "side"}:
            labels = df[c].map(normalize)
            if labels.nunique() > 1:
                mask = labels.str.contains("bilateral", regex=False) | labels.isin(["both", "combined"])
                if not mask.any():
                    raise ValueError(f"Multiple ROI/hemisphere rows without explicit bilateral rows: {path}")
                df = df.loc[mask].copy()
    values = df[[b, a]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if len(values) == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"Invalid or missing paired tSNR values: {path}. No subject will be silently dropped.")
    runs = df[run_columns[0]].astype(str).str.strip().str.lower().str.replace(r"^run[ _-]*", "", regex=True)
    if runs.duplicated().any():
        raise ValueError(f"Repeated run rows in {path}; check ROI structure before averaging.")
    return pd.DataFrame({"Subject": subject, "Region": region, "Run": runs.to_numpy(),
        "Without_NORDIC": values[:, 0], "With_NORDIC": values[:, 1]})


def summarize(raw, subjects):
    means = raw.groupby(["Region", "Subject"], sort=False)[["Without_NORDIC", "With_NORDIC"]].mean()
    counts = raw.groupby(["Region", "Subject"], sort=False).size()
    tables, rows = {}, []
    for region in REGIONS:
        table = means.loc[region].reindex(subjects).copy()
        if table.isna().any().any():
            raise ValueError(f"Missing subject in {region}")
        table["Number_of_Runs"] = counts.loc[region].reindex(subjects)
        table["Difference_With_minus_Without"] = table.With_NORDIC - table.Without_NORDIC
        tables[region] = table
        row = {"Region": region, "N_subjects": len(subjects)}
        for condition in ["Without_NORDIC", "With_NORDIC"]:
            values = table[condition]
            row[condition+"_mean"] = values.mean()
            row[condition+"_SD"] = values.std(ddof=1)
            row[condition+"_SEM"] = values.std(ddof=1)/np.sqrt(len(values))
        rows.append(row)
    return tables, pd.DataFrame(rows)


def draw_panel(ax, region, table, stat, letter, ymax):
    values = table[["Without_NORDIC", "With_NORDIC"]].to_numpy()
    means = np.array([stat.Without_NORDIC_mean, stat.With_NORDIC_mean])
    sds = np.array([stat.Without_NORDIC_SD, stat.With_NORDIC_SD])
    ax.bar([0, 1], means, width=.66, color=["#cee3ee", "#d8edc2"],
        edgecolor="#666666", linewidth=1.2, zorder=1)
    ax.errorbar(np.array([0, 1])-.265, means, yerr=sds, fmt="none",
        ecolor="black", elinewidth=1.6, capsize=4, capthick=1.6, zorder=3)
    offsets = np.linspace(-JITTER_WIDTH, JITTER_WIDTH, len(table))
    for i, pair in enumerate(values):
        xs = np.array([0, 1])+offsets[i]
        ax.plot(xs, pair, color=COLORS[i], lw=1.6, alpha=.72, zorder=4)
        ax.scatter(xs, pair, s=82, color=COLORS[i], edgecolors="white", linewidths=.9, zorder=5)

    # FINAL CHANGE: dedicated top band ABOVE ALL lines, points and SD whiskers.
    # Only horizontal placement follows the two conditions; values are unchanged.
    for j in range(2):
        ax.text(j, .91*ymax, f"{means[j]:.2f} ± {sds[j]:.2f}", ha="center",
            va="center", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=.25", facecolor="white", edgecolor="none"), zorder=7)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Without NORDIC", "With NORDIC"])
    ax.set_ylabel("Mean tSNR")
    ax.set_title(region, fontweight="bold", pad=15)
    ax.set_xlim(-.47, 1.47)
    ax.set_ylim(0, ymax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.16, linewidth=.8)
    ax.set_axisbelow(True)
    ax.text(-.10, 1.04, letter, transform=ax.transAxes, fontsize=20, fontweight="bold", va="top")


def export(fig, destination, stem):
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(destination/f"{stem}.{extension}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    subjects = list(SUBJECT_PATHS)
    if len(subjects) != 5:
        raise ValueError("Exactly five subjects are expected.")
    missing = [str(paths[key]/filename) for paths in SUBJECT_PATHS.values()
        for key, filename in REGIONS.values() if not (paths[key]/filename).is_file()]
    if missing:
        raise FileNotFoundError("Check these input paths:\n"+"\n".join(missing))
    raw = pd.concat([read_csv(paths[key], filename, subject, region)
        for subject, paths in SUBJECT_PATHS.items()
        for region, (key, filename) in REGIONS.items()], ignore_index=True)
    run_sets = raw.groupby(["Region", "Subject"])["Run"].agg(lambda x: tuple(sorted(x)))
    if run_sets.nunique() != 1:
        raise ValueError("Run identifiers/counts differ across subjects/regions:\n"+run_sets.to_string())
    tables, summary = summarize(raw, subjects)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUTPUT_DIR/"ALL_SUBJECTS_ALL_RUNS_UPDATED.csv", index=False)
    summary.to_csv(OUTPUT_DIR/"GROUP_TSNR_MEAN_SD_UPDATED.csv", index=False)
    for region, table in tables.items():
        table.reset_index().to_csv(OUTPUT_DIR/f"{region.replace(' ', '_')}_SUBJECT_MEANS_UPDATED.csv", index=False)
    with pd.ExcelWriter(OUTPUT_DIR/"GROUP_TSNR_MEAN_SD_UPDATED.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Group mean SD SEM", index=False)
        raw.to_excel(writer, sheet_name="All runs", index=False)
        for region, table in tables.items():
            table.reset_index().to_excel(writer, sheet_name=region, index=False)

    max_extent = 0.0
    for region, table in tables.items():
        stat = summary.loc[summary.Region==region].iloc[0]
        max_extent = max(max_extent, table[["Without_NORDIC", "With_NORDIC"]].to_numpy().max(),
            stat.Without_NORDIC_mean+stat.Without_NORDIC_SD,
            stat.With_NORDIC_mean+stat.With_NORDIC_SD)
    ymax = max_extent*1.22
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, (region, table), letter in zip(axes, tables.items(), "AB"):
        draw_panel(ax, region, table, summary.loc[summary.Region==region].iloc[0], letter, ymax)
    fig.suptitle("Group-level temporal signal-to-noise ratio (tSNR)", fontsize=17, fontweight="bold", y=.99)
    fig.subplots_adjust(left=.075, right=.98, bottom=.13, top=.80, wspace=.27)
    export(fig, OUTPUT_DIR, FIGURE_STEM)
    for region, table in tables.items():
        fig, ax = plt.subplots(figsize=(6.7, 5.7))
        draw_panel(ax, region, table, summary.loc[summary.Region==region].iloc[0], "", ymax)
        fig.subplots_adjust(left=.16, right=.97, bottom=.14, top=.88)
        export(fig, OUTPUT_DIR, region.replace(' ', '_')+"_JITTER_MEAN_SD_FINAL")
    caption = (
        "Whole-brain (A) and posterior-insula (B) tSNR without and with NORDIC. "
        "Run-level mean tSNR was averaged within each of five participants, then across participants. "
        "Bars show group means; numbers in the upper band and black error bars show mean ± sample SD across participants (not SEM). "
        "Five coloured points per condition represent individual participants. Matching colours and connecting lines denote paired measurements. "
        "Fixed horizontal offsets separate overlapping points without modifying tSNR values. "
        "No subject legend, significance brackets or p-values are displayed."
    )
    (OUTPUT_DIR/"FIGURE_CAPTION_MEAN_SD_FINAL.txt").write_text(caption, encoding="utf-8")
    print(f"Completed. Output folder: {OUTPUT_DIR}")
    print(f"Main figure: {FIGURE_STEM}.png / .pdf / .svg")


if __name__ == "__main__":
    main()
