from pathlib import Path
import sys
import h5py
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from nibabel.freesurfer.io import read_geometry

# ============================================================================
# CLEAN METHODS FIGURE: contour-style white/pial surfaces, similar to example
#
# Unlike the previous figure, this script does NOT draw the full red ROI mask.
# It draws only:
#   Green  = white / GM-WM surface contour (near selected axial slice)
#   Yellow = pial / GM-CSF surface contour (near selected axial slice)
#   Red    = sparse pIns surface-ROI vertices (not a large voxel block)
#   Blue   = a few representative white-to-pial paths
#   White  = 20 sampling depths on each representative path
#
# This is a methodological illustration, not an activation map.
# ============================================================================

T1_FILE = Path(
    r"D:\NORDIC_Subject1_project\7485_extracted"
    r"\UNI_MoCo_MPRAGEised_biascorrected.nii"
)

SURF_DIR = Path(
    r"D:\NORDIC_Subject1_project\7485_extracted\7485\surf"
)

SAMPLED_H5 = Path(
    r"D:\NORDIC_Subject1_project\Laminarprofile\1.GLM_with"
    r"\pIns_laminar_sampling_7485.h5"
)

OUTDIR = Path(
    r"D:\NORDIC_Subject1_project\Laminarprofile\TSNR Depth\visualization"
)

OUTPUT_PNG = OUTDIR / "Figure_pIns_surface_contours_and_sampling_paths.png"
OUTPUT_PDF = OUTDIR / "Figure_pIns_surface_contours_and_sampling_paths.pdf"

N_DEPTHS = 20
N_DISPLAY_COLUMNS = 7
SURFACE_TOLERANCE_VOX = 0.75
ROI_TOLERANCE_VOX = 1.25
CROP_MARGIN_VOX = 9

WHITE_COLOR = "#00B83F"
PIAL_COLOR = "#F0D000"
ROI_COLOR = "#E31A1C"
PATH_COLOR = "#1976B9"
DEPTH_COLOR = "#FFFFFF"


def check_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file:\n{path}")


def ras_to_voxel(coords, affine):
    hom = np.column_stack((coords, np.ones(coords.shape[0])))
    return (np.linalg.inv(affine) @ hom.T).T[:, :3]


def to_local_ids(ids, side, n_lh, n_current):
    ids = np.asarray(ids, dtype=np.int64)
    if ids.min() >= 0 and ids.max() < n_current:
        return ids
    if side == "right":
        converted = ids - n_lh
        if converted.min() >= 0 and converted.max() < n_current:
            return converted
    raise RuntimeError(f"Cannot convert {side} IDs {ids.min()}..{ids.max()} to local surface IDs.")


def choose_slice(white_roi, pial_roi):
    midpoint = (white_roi[:, 2] + pial_roi[:, 2]) / 2
    return int(np.round(np.median(midpoint)))


def choose_paths(white, pial, z):
    midpoint = (white[:, :2] + pial[:, :2]) / 2
    crosses = np.flatnonzero(
        (np.minimum(white[:, 2], pial[:, 2]) <= z)
        & (np.maximum(white[:, 2], pial[:, 2]) >= z)
    )
    if crosses.size < N_DISPLAY_COLUMNS:
        crosses = np.argsort(np.abs((white[:, 2] + pial[:, 2]) / 2 - z))[:100]

    candidates = midpoint[crosses]
    selected = [crosses[np.argmin(candidates[:, 0])]]
    while len(selected) < min(N_DISPLAY_COLUMNS, crosses.size):
        selected_xy = np.array([midpoint[i] for i in selected])
        distances = np.min(
            np.linalg.norm(candidates[:, None, :] - selected_xy[None, :, :], axis=2),
            axis=1,
        )
        candidate = crosses[np.argmax(distances)]
        if candidate in selected:
            break
        selected.append(candidate)
    return np.asarray(selected, dtype=np.int64)


def crop_limits(white_roi, pial_roi):
    all_xy = np.vstack((white_roi[:, :2], pial_roi[:, :2]))
    return (
        int(np.floor(all_xy[:, 0].min() - CROP_MARGIN_VOX)),
        int(np.ceil(all_xy[:, 0].max() + CROP_MARGIN_VOX)),
        int(np.floor(all_xy[:, 1].min() - CROP_MARGIN_VOX)),
        int(np.ceil(all_xy[:, 1].max() + CROP_MARGIN_VOX)),
    )


def draw_contour(ax, points_xy, color, label):
    """Draw a smooth-looking contour from surface points near the axial slice."""
    if points_xy.shape[0] < 3:
        return
    ax.scatter(points_xy[:, 0], points_xy[:, 1], s=1.8, color=color, linewidths=0, zorder=4, label=label)


def draw_panel(ax, side, t1, white_all, pial_all, white_roi, pial_roi, z, paths):
    background = t1[:, :, z].T
    lo, hi = np.percentile(background[np.isfinite(background)], [1, 99])
    ax.imshow(background, cmap="gray", origin="lower", vmin=lo, vmax=hi, interpolation="nearest")

    # Only surface points close to this exact slice: contour-like display.
    white_near = white_all[np.abs(white_all[:, 2] - z) <= SURFACE_TOLERANCE_VOX]
    pial_near = pial_all[np.abs(pial_all[:, 2] - z) <= SURFACE_TOLERANCE_VOX]
    draw_contour(ax, white_near[:, :2], WHITE_COLOR, "White surface")
    draw_contour(ax, pial_near[:, :2], PIAL_COLOR, "Pial surface")

    # Sparse red pIns ROI vertices only near the selected slice. This replaces
    # the large blocky volumetric red mask in the previous figure.
    roi_mid = (white_roi + pial_roi) / 2
    roi_near = roi_mid[np.abs(roi_mid[:, 2] - z) <= ROI_TOLERANCE_VOX]
    ax.scatter(
        roi_near[:, 0], roi_near[:, 1],
        s=7, color=ROI_COLOR, alpha=0.85, linewidths=0, zorder=5,
    )

    depths = np.linspace(0, 1, N_DEPTHS)
    for idx in paths:
        points = white_roi[idx][None, :] + depths[:, None] * (pial_roi[idx] - white_roi[idx])[None, :]
        ax.plot(points[:, 0], points[:, 1], color=PATH_COLOR, linewidth=0.9, zorder=7)
        ax.scatter(points[:, 0], points[:, 1], s=5, color=DEPTH_COLOR, edgecolors=PATH_COLOR, linewidths=0.2, zorder=8)
        ax.scatter(white_roi[idx, 0], white_roi[idx, 1], s=13, color=WHITE_COLOR, edgecolors="black", linewidths=0.25, zorder=9)
        ax.scatter(pial_roi[idx, 0], pial_roi[idx, 1], s=13, color=PIAL_COLOR, edgecolors="black", linewidths=0.25, zorder=9)

    # One clear arrow explaining direction.
    idx = paths[len(paths) // 2]
    ax.add_patch(FancyArrowPatch(
        white_roi[idx, :2], pial_roi[idx, :2], arrowstyle="->",
        mutation_scale=10, linewidth=0.9, color=PATH_COLOR, zorder=10,
    ))

    x0, x1, y0, y1 = crop_limits(white_roi, pial_roi)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("A  Left pIns" if side == "left" else "B  Right pIns", loc="left", fontsize=12, fontweight="bold")


def main():
    for path in (
        T1_FILE, SAMPLED_H5,
        SURF_DIR / "lh.white", SURF_DIR / "lh.pial",
        SURF_DIR / "rh.white", SURF_DIR / "rh.pial",
    ):
        check_file(path)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    t1_img = nib.load(str(T1_FILE))
    t1 = np.asanyarray(t1_img.dataobj, dtype=np.float32)
    if t1.ndim > 3:
        t1 = t1[..., 0]

    lh_white, _ = read_geometry(str(SURF_DIR / "lh.white"))
    lh_pial, _ = read_geometry(str(SURF_DIR / "lh.pial"))
    rh_white, _ = read_geometry(str(SURF_DIR / "rh.white"))
    rh_pial, _ = read_geometry(str(SURF_DIR / "rh.pial"))

    with h5py.File(SAMPLED_H5, "r") as h5:
        ids = {
            "left": np.asarray(h5["left"]["surface_vertex_indices"][:], dtype=np.int64),
            "right": np.asarray(h5["right"]["surface_vertex_indices"][:], dtype=np.int64),
        }

    surfaces = {"left": (lh_white, lh_pial), "right": (rh_white, rh_pial)}
    panel = {}

    for side in ("left", "right"):
        white_surf, pial_surf = surfaces[side]
        local = to_local_ids(ids[side], side, len(lh_white), len(white_surf))
        white_all = ras_to_voxel(white_surf, t1_img.affine)
        pial_all = ras_to_voxel(pial_surf, t1_img.affine)
        white_roi = white_all[local]
        pial_roi = pial_all[local]
        z = choose_slice(white_roi, pial_roi)
        z = int(np.clip(z, 0, t1.shape[2] - 1))
        paths = choose_paths(white_roi, pial_roi, z)
        panel[side] = (white_all, pial_all, white_roi, pial_roi, z, paths)
        print(f"{side.upper()}: z={z}; displayed paths={len(paths)}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), gridspec_kw={"wspace": 0.08})
    for ax, side in zip(axes, ("left", "right")):
        draw_panel(ax, side, t1, *panel[side])

    handles = [
        Line2D([0], [0], color=WHITE_COLOR, marker="o", markersize=4, linewidth=1, label="GM/WM (white) surface"),
        Line2D([0], [0], color=PIAL_COLOR, marker="o", markersize=4, linewidth=1, label="GM/CSF (pial) surface"),
        Line2D([0], [0], color=ROI_COLOR, marker="o", markersize=5, linewidth=0, label="pIns surface ROI vertices"),
        Line2D([0], [0], color=PATH_COLOR, linewidth=1, label="Representative white-to-pial path"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor=DEPTH_COLOR, markeredgecolor=PATH_COLOR, markersize=4, label="20 depth samples"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("Subject-specific pIns cortical-depth sampling geometry", fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.935, "White matter → pial surface; 20 normalized sampling depths", ha="center", fontsize=9.5, color="#444444")

    fig.savefig(OUTPUT_PNG, dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PDF, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", OUTPUT_PNG)
    print("Saved:", OUTPUT_PDF)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}")
        sys.exit(1)
