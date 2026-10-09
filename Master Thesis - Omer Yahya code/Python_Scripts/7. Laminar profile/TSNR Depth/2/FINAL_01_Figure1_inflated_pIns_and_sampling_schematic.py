from pathlib import Path
import sys
import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from nibabel.freesurfer.io import read_geometry, read_morph_data
from nilearn import plotting

SUBJECT = "7485"

SURF_DIR = Path(
    r"D:\NORDIC_Subject1_project\7485_extracted\7485\surf"
)

H5_FILE = Path(
    r"D:\NORDIC_Subject1_project\Laminarprofile\GLM_with"
    r"\pIns_laminar_sampling_7485.h5"
)

OUTDIR = Path(
    r"D:\NORDIC_Subject1_project\Laminarprofile\TSNR Depth\visualization\2\8"
)

N_DEPTHS = 20
DPI = 450

ROI_TEAL = "#1B8A89"
ROI_EDGE = "#0C3B3A"
TEXT_COLOR = "#202020"
WM_BLUE = "#2A6FBB"
PIAL_RED = "#B3343F"


def validate_required_files():
    required = [
        H5_FILE,
        SURF_DIR / "lh.inflated",
        SURF_DIR / "rh.inflated",
        SURF_DIR / "lh.sulc",
        SURF_DIR / "rh.sulc",
    ]

    missing = [str(path) for path in required if not path.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing required input file(s):\n" + "\n".join(missing)
        )


def get_rh_local_ids(rh_ids_from_h5, n_lh_vertices, n_rh_vertices):
    rh_ids_from_h5 = np.asarray(rh_ids_from_h5, dtype=np.int64)

    if rh_ids_from_h5.size == 0:
        raise RuntimeError("RH ROI contains zero sampled vertices.")

    local_valid = (
        rh_ids_from_h5.min() >= 0
        and rh_ids_from_h5.max() < n_rh_vertices
    )

    global_valid = (
        rh_ids_from_h5.min() >= n_lh_vertices
        and rh_ids_from_h5.max() < (n_lh_vertices + n_rh_vertices)
    )

    if local_valid:
        return rh_ids_from_h5, "local RH FreeSurfer vertex indices"

    if global_valid:
        return (
            rh_ids_from_h5 - n_lh_vertices,
            "global bilateral indices converted to local RH indices"
        )

    raise RuntimeError(
        "RH surface_vertex_indices do not match either local RH indices "
        "or global bilateral vertex indexing."
    )


def load_h5_roi_ids():
    lh_vertices, _ = read_geometry(str(SURF_DIR / "lh.inflated"))
    rh_vertices, _ = read_geometry(str(SURF_DIR / "rh.inflated"))

    with h5py.File(H5_FILE, "r") as h5:
        lh_ids = np.asarray(
            h5["left"]["surface_vertex_indices"][:],
            dtype=np.int64
        )

        rh_raw_ids = np.asarray(
            h5["right"]["surface_vertex_indices"][:],
            dtype=np.int64
        )

    if lh_ids.size == 0:
        raise RuntimeError("LH ROI contains zero sampled vertices.")

    if lh_ids.min() < 0 or lh_ids.max() >= len(lh_vertices):
        raise RuntimeError(
            "LH surface_vertex_indices do not match lh.inflated."
        )

    rh_ids, rh_index_note = get_rh_local_ids(
        rh_raw_ids,
        n_lh_vertices=len(lh_vertices),
        n_rh_vertices=len(rh_vertices)
    )

    return lh_ids, rh_ids, rh_index_note


def roi_boundary_vertices(faces, roi_mask):
    boundary = np.zeros(len(roi_mask), dtype=bool)

    for triangle in faces:
        tri_roi = roi_mask[triangle]

        if np.any(tri_roi) and not np.all(tri_roi):
            boundary[triangle[tri_roi]] = True

    return boundary


def roi_boundary_edges(faces, roi_mask):
    """
    Return mesh edges at the ROI boundary:
    one endpoint inside ROI and the other outside.
    """
    all_edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ])

    all_edges = np.sort(all_edges, axis=1)
    all_edges = np.unique(all_edges, axis=0)

    is_boundary = roi_mask[all_edges[:, 0]] != roi_mask[all_edges[:, 1]]
    return all_edges[is_boundary]


def render_inflated_roi(hemi, ids):
    vertices, faces = read_geometry(str(SURF_DIR / f"{hemi}.inflated"))
    sulc = read_morph_data(str(SURF_DIR / f"{hemi}.sulc"))

    ids = np.asarray(ids, dtype=np.int64)

    if ids.size == 0:
        raise RuntimeError(f"{hemi}: ROI has zero vertices.")

    if ids.min() < 0 or ids.max() >= len(vertices):
        raise RuntimeError(
            f"{hemi}: ROI indices do not match the inflated surface."
        )

    roi_mask = np.zeros(len(vertices), dtype=bool)
    roi_mask[ids] = True

    roi_map = np.full(len(vertices), np.nan, dtype=float)
    roi_map[roi_mask] = 1.0

    fig = plotting.plot_surf_stat_map(
    surf_mesh=(vertices, faces),
    stat_map=roi_map,
    bg_map=sulc,
    hemi="left" if hemi == "lh" else "right",
    view="lateral",
    cmap=ListedColormap(["#E76F51"]),
    colorbar=False,
    threshold=0.5,
    vmin=0.0,
    vmax=1.0,
    bg_on_data=False,
    title=None

)

    ax = fig.axes[0]

    boundary_edges = roi_boundary_edges(faces, roi_mask)

    for edge in boundary_edges:
        xyz = vertices[edge]

        ax.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            color="#102B2A",
            linewidth=0.65,
            alpha=0.95
        )

    output_path = OUTDIR / f"_temporary_{hemi}_inflated_roi.png"

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white"
    )

    plt.close(fig)

    return output_path


def make_sampling_schematic(output_path):
    fig, ax = plt.subplots(figsize=(3.25, 4.5), facecolor="white")

    depths = np.linspace(0, 1, N_DEPTHS)

    x = 0.45
    y_deep = 0.13
    y_superficial = 0.88

    y = y_deep + depths * (y_superficial - y_deep)

    ax.plot(
        [x, x],
        [y_deep, y_superficial],
        color=TEXT_COLOR,
        linewidth=2.2,
        zorder=1
    )

    ax.scatter(
        np.full(N_DEPTHS, x),
        y,
        c=depths,
        cmap="viridis",
        s=38,
        edgecolors=TEXT_COLOR,
        linewidths=0.30,
        zorder=2
    )

    ax.scatter(
        [x],
        [y_deep],
        color=WM_BLUE,
        s=95,
        edgecolors=TEXT_COLOR,
        linewidths=0.6,
        zorder=3
    )

    ax.scatter(
        [x],
        [y_superficial],
        color=PIAL_RED,
        s=95,
        edgecolors=TEXT_COLOR,
        linewidths=0.6,
        zorder=3
    )

    ax.annotate(
        "",
        xy=(0.76, y_superficial),
        xytext=(0.76, y_deep),
        arrowprops=dict(
            arrowstyle="->",
            color=TEXT_COLOR,
            linewidth=1.6
        )
    )

    ax.text(
        0.83,
        0.505,
        "deep → superficial",
        rotation=90,
        va="center",
        ha="center",
        fontsize=10,
        color=TEXT_COLOR
    )

    ax.text(
        0.35,
        y_deep,
        "White/GM boundary\n"
        "d = 0\n"
        "Depth 1",
        ha="right",
        va="center",
        fontsize=9,
        color=WM_BLUE
    )

    ax.text(
        0.35,
        y_superficial,
        "Pial boundary\n"
        "d = 1\n"
        "Depth 20",
        ha="right",
        va="center",
        fontsize=9,
        color=PIAL_RED
    )

    ax.text(
        0.50,
        0.01,
        "20 equidistant positions\n"
        "along each corresponding\n"
        "white-to-pial vertex pair",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TEXT_COLOR
    )

    ax.set_xlim(0.02, 1.0)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white"
    )

    plt.close(fig)


def write_qc_csv(lh_ids, rh_ids, rh_index_note):
    output_csv = OUTDIR / "Figure_1_sampling_QC_summary.csv"

    rows = [
        ["subject", SUBJECT],
        ["number_of_depths", N_DEPTHS],
        ["left_pIns_sampled_vertices", len(lh_ids)],
        ["right_pIns_sampled_vertices", len(rh_ids)],
        ["right_index_interpretation", rh_index_note],
        ["sampling_model", "x(d) = W + d*(P-W)"],
        ["depth_definition", "d=0 white/GM boundary; d=1 pial boundary"],
        ["sampling_type", "equidistant linear white-to-pial sampling"],
        ["not_used", "LayNii / equivolume / histological layers"],
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "value"])
        writer.writerows(rows)

    return output_csv


def assemble_figure(lh_img, rh_img, schematic_img, n_lh, n_rh):
    fig = plt.figure(figsize=(12.2, 6.6), facecolor="white")

    grid = GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[1.35, 1.35, 0.72],
        wspace=0.025
    )

    panels = [
        (
            fig.add_subplot(grid[0, 0]),
            lh_img,
            "A",
            "Left posterior insula",
            f"{n_lh} sampled cortical columns"
        ),
        (
            fig.add_subplot(grid[0, 1]),
            rh_img,
            "B",
            "Right posterior insula",
            f"{n_rh} sampled cortical columns"
        ),
        (
            fig.add_subplot(grid[0, 2]),
            schematic_img,
            "C",
            "Cortical-depth sampling",
            ""
        ),
    ]

    for ax, image_path, letter, title, subtitle in panels:
        ax.imshow(mpimg.imread(image_path))
        ax.axis("off")

        ax.text(
            0.02,
            0.98,
            letter,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=16,
            fontweight="bold",
            color=TEXT_COLOR
        )

        ax.set_title(title, fontsize=11.5, pad=6, color=TEXT_COLOR)

        if subtitle:
            ax.text(
                0.5,
                0.015,
                subtitle,
                transform=ax.transAxes,
                va="bottom",
                ha="center",
                fontsize=9,
                color=TEXT_COLOR
            )

    fig.suptitle(
        "Subject 7485: posterior-insula ROI and cortical-depth sampling",
        fontsize=15,
        y=0.978,
        color=TEXT_COLOR
    )

    fig.text(
        0.5,
        0.018,
        "Teal: pIns vertices used for matched pre-/post-NORDIC sampling. "
        "Gray: inflated cortical anatomy (sulcal shading). "
        "Depth is a relative white-to-pial cortical fraction; it is not "
        "equivolume sampling or a histological cortical-layer label.",
        ha="center",
        fontsize=9.2,
        color=TEXT_COLOR
    )

    png_path = OUTDIR / "Figure_1_pIns_ROI_and_sampling_schematic.png"
    pdf_path = OUTDIR / "Figure_1_pIns_ROI_and_sampling_schematic.pdf"

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white"
    )

    plt.close(fig)

    return png_path, pdf_path


def main():
    validate_required_files()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    lh_ids, rh_ids, rh_index_note = load_h5_roi_ids()

    lh_img = render_inflated_roi("lh", lh_ids)
    rh_img = render_inflated_roi("rh", rh_ids)

    schematic_img = OUTDIR / "_temporary_sampling_schematic.png"
    make_sampling_schematic(schematic_img)

    png_path, pdf_path = assemble_figure(
        lh_img=lh_img,
        rh_img=rh_img,
        schematic_img=schematic_img,
        n_lh=len(lh_ids),
        n_rh=len(rh_ids)
    )

    qc_path = write_qc_csv(
        lh_ids=lh_ids,
        rh_ids=rh_ids,
        rh_index_note=rh_index_note
    )

    for temp_path in [lh_img, rh_img, schematic_img]:
        if temp_path.exists():
            temp_path.unlink()

    for output_path in [png_path, pdf_path, qc_path]:
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(
                f"Output was not created correctly: {output_path}"
            )

    print("\nSUCCESS")
    print(f"Figure PNG: {png_path}")
    print(f"Figure PDF: {pdf_path}")
    print(f"QC CSV:     {qc_path}")
    print(f"LH columns: {len(lh_ids)}")
    print(f"RH columns: {len(rh_ids)}")
    print(f"RH indexing: {rh_index_note}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)