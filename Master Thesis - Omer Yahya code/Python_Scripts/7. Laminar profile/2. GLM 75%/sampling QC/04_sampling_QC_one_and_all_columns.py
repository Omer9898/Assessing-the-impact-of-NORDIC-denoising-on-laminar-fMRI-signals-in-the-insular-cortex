"""Sampling QC only: no GLM, no inferred task labels.
Install: python -m pip install numpy h5py matplotlib
Run after both sampling files finish. Change INPUTS if needed.
"""
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"D:\NORDIC_Subject1_project\Laminarprofile")
INPUTS = {
    "Without NORDIC": Path(
        r"D:\NORDIC_Subject1_project\Laminarprofile"
        r"\1. Sampling\Without"
        r"\pIns_laminar_sampling_7485_without_NORDIC.h5"
    ),
    "With NORDIC": Path(
        r"D:\NORDIC_Subject1_project\Laminarprofile"
        r"\1. Sampling\With"
        r"\pIns_laminar_sampling_7485_with_NORDIC.h5"
    ),
}
OUT = Path(r"D:\NORDIC_Subject1_project\Laminarprofile\2. GLM 75%")
REPRESENTATIVE_LOCAL_INDEX = {"left": None, "right": None}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for path in INPUTS.values():
        if not path.is_file():
            raise FileNotFoundError(f"Change INPUTS: missing {path}")
    with h5py.File(INPUTS["Without NORDIC"], "r") as pre, h5py.File(INPUTS["With NORDIC"], "r") as post:
        files = {"Without NORDIC": pre, "With NORDIC": post}
        lengths = np.asarray(pre["run_lengths"][:], dtype=int)
        if not np.array_equal(lengths, post["run_lengths"][:]):
            raise ValueError("Run lengths differ before/after NORDIC")
        boundaries = np.cumsum(lengths)[:-1] - 0.5
        for side in ("left", "right"):
            ids = pre[side]["surface_vertex_indices"][:]
            if not np.array_equal(ids, post[side]["surface_vertex_indices"][:]):
                raise ValueError(f"{side}: vertex identities/order differ")
            shape = pre[side]["layeractivation"].shape
            if shape != post[side]["layeractivation"].shape or shape[1] != 20 or shape[2] != lengths.sum():
                raise ValueError(f"{side}: invalid or unmatched sampling dimensions")
            index = REPRESENTATIVE_LOCAL_INDEX[side]
            if index is None:
                index = len(ids) // 2
            if not 0 <= index < len(ids):
                raise ValueError(f"{side}: representative index out of range")
            representatives, averages, deepall = {}, {}, {}
            for name, handle in files.items():
                ds = handle[side]["layeractivation"]
                rep = ds[index, :, :].astype(float)
                total = np.zeros(shape[1:], dtype=float)
                count = np.zeros(shape[1:], dtype=np.int64)
                deep = np.empty((shape[0], shape[2]), dtype=np.float32)
                for start in range(0, shape[0], 64):
                    stop = min(start + 64, shape[0])
                    block = ds[start:stop, :, :].astype(float)
                    finite = np.isfinite(block)
                    total += np.where(finite, block, 0).sum(axis=0)
                    count += finite.sum(axis=0)
                    deep[start:stop] = np.nanmean(block[:, :15, :], axis=1)
                avg = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
                representatives[name], averages[name], deepall[name] = rep, avg, deep
            def heatmaps(arrays, title, ylabel, filename, depth=False):
                finite_values = np.concatenate([a[np.isfinite(a)] for a in arrays.values()])
                if not finite_values.size:
                    raise ValueError(f"{side}: no finite data")
                low, high = np.percentile(finite_values, [1, 99])
                fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
                for ax, (name, a) in zip(axes, arrays.items()):
                    im = ax.imshow(a, origin="lower", aspect="auto", interpolation="nearest", vmin=low, vmax=high)
                    ax.set(title=name, xlabel="Time point", ylabel=ylabel)
                    for boundary in boundaries:
                        ax.axvline(boundary, color="white", linestyle="--", linewidth=.7)
                    if depth:
                        ax.axhline(14.5, color="white", linestyle="--", linewidth=.8)
                fig.suptitle(title)
                fig.colorbar(im, ax=list(axes), label="Signal intensity (shared 1st-99th percentile scale)")
                fig.savefig(OUT / filename, dpi=180)
                plt.close(fig)
            heatmaps(representatives, f"{side}: one column; local index {index}, surface vertex {ids[index]}", "Depth index (0=white, 19=pial)", f"{side}_one_column.png", True)
            heatmaps(averages, f"{side}: mean across ALL {len(ids)} columns at each depth", "Depth index (0=white, 19=pial)", f"{side}_all_columns_mean_by_depth.png", True)
            heatmaps(deepall, f"{side}: ALL columns, each averaged over depths 0-14", "Column local index", f"{side}_all_columns_deep75.png")
            fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True, sharex=True)
            for name in files:
                axes[0].plot(np.nanmean(representatives[name][:15], axis=0), label=name, linewidth=.6)
                axes[1].plot(np.nanmean(deepall[name], axis=0), label=name, linewidth=.8)
            for ax, title in zip(axes, ("ONE column: deep75 mean", "ALL columns: mean of column deep75 series")):
                ax.set(title=f"{side}: {title}", ylabel="Signal intensity")
                ax.legend()
                for boundary in boundaries:
                    ax.axvline(boundary, color="black", linestyle="--", linewidth=.7)
            axes[1].set_xlabel("Time point; dashed lines separate runs")
            fig.savefig(OUT / f"{side}_deep75_one_and_all.png", dpi=180)
            plt.close(fig)
    print(f"Saved 8 QC figures to {OUT}")
    print("These figures do NOT estimate pain, cognition, or anticipation effects.")

if __name__ == "__main__":
    main()
