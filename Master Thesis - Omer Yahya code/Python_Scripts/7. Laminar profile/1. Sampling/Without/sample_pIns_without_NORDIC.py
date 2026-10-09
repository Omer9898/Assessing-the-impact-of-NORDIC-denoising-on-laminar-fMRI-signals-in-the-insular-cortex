from pathlib import Path
import re
import sys
import time

import h5py
import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import read_geometry
from scipy.ndimage import map_coordinates
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

SUBJECT = "7485"

FS_ROOT = Path(
    r"D:\NORDIC_Subject1_project\7485_extracted"
)

T1_FILE = Path(
    r"D:\NORDIC_Subject1_project\7485_extracted\UNI_MoCo_MPRAGEised_biascorrected.nii"
)

ROI_DIR = Path(
    r"D:\NORDIC_Subject1_project\Laminarprofile\masks"
)
RUN_DIRS = [
    Path(
        r"C:\Users\L E N O V O\OneDrive\Desktop\subject 1\run 1"
    ), 
    Path(
        r"C:\Users\L E N O V O\OneDrive\Desktop\subject 1\run 2"
    ),
    Path(
        r"C:\Users\L E N O V O\OneDrive\Desktop\subject 1\run 3"
    ),
]

OUT_DIR = Path(
    r"D:\NORDIC_Subject1_project\Laminarprofile\1. Sampling\Without"
)

OUTPUT_FILE = OUT_DIR / "pIns_laminar_sampling_7485_without_NORDIC.h5"

ROI_STEMS = {
    "left": "roi_pInsleft_17",
    "right": "roi_pInsright_118",
}

N_LAYERS = 20
N_VOLUMES_PER_RUN = 407
N_RUNS = 3
EXPECTED_TOTAL_VOLUMES = N_VOLUMES_PER_RUN * N_RUNS

INTERPOLATION_ORDER = 3


# ============================================================
# HELPERS
# ============================================================

def die(message):
    raise RuntimeError(message)


def log(message):
    print(
        f"[{time.strftime('%H:%M:%S')}] {message}",
        flush=True,
    )

def ordered_run_files(run_dir):
    pattern = re.compile(
        r"^mag_POCS_r(\d+)_(\d+)_Warped-to-Anat\.nii\.gz$"
    )

    files = []
    run_numbers = set()

    for path in run_dir.glob("mag_POCS_r*_Warped-to-Anat.nii.gz"):
        match = pattern.fullmatch(path.name)
        if match:
            run_numbers.add(int(match.group(1)))
            files.append((int(match.group(2)), path))

    if len(run_numbers) != 1:
        die(
            f"{run_dir}: expected files from one run only; "
            f"found run numbers {sorted(run_numbers)}."
        )

    files.sort(key=lambda item: item[0])
    indices = [item[0] for item in files]
    expected = list(range(1000, 1000 + N_VOLUMES_PER_RUN))

    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        extra = sorted(set(indices) - set(expected))
        die(
            f"{run_dir}\n"
            f"Expected {N_VOLUMES_PER_RUN} volumes, found {len(files)}.\n"
            f"Missing indices: {missing}\n"
            f"Unexpected indices: {extra}"
        )

    return [item[1] for item in files]


def load_roi_mask(stem, n_vertices):
    nii_file = ROI_DIR / f"{stem}.nii.gz"
    lh_file = ROI_DIR / f"{stem}_lh.shape.gii"
    rh_file = ROI_DIR / f"{stem}_rh.shape.gii"

    for path in (nii_file, lh_file, rh_file):
        if not path.is_file():
            die(f"Missing ROI file:\n{path}")

    lh_values = np.asarray(
        nib.load(str(lh_file)).darrays[0].data,
        dtype=np.float64,
    ).ravel()

    rh_values = np.asarray(
        nib.load(str(rh_file)).darrays[0].data,
        dtype=np.float64,
    ).ravel()

    mask_values = np.concatenate(
        (lh_values, rh_values)
    )

    mask = np.abs(mask_values) >= 1e-2

    if mask.size != n_vertices:
        die(
            f"{stem}: ROI has {mask.size} vertices, "
            f"but surfaces have {n_vertices} vertices."
        )

    if not np.any(mask):
        die(f"{stem}: ROI mask is empty.")

    return mask


def ras_to_spm_vox(vertices_ras, affine):
    homogeneous = np.column_stack(
        (
            vertices_ras,
            np.ones(vertices_ras.shape[0]),
        )
    )

    vertices_vox = np.linalg.solve(
        affine,
        homogeneous.T,
    ).T

    return vertices_vox[:, :3]


def make_coordinates(
    white_vox,
    pial_vox,
    mask,
):
    white = white_vox[mask]
    pial = pial_vox[mask]

    depths = np.linspace(
        0.0,
        1.0,
        N_LAYERS,
        dtype=np.float64,
    )

    coordinates_spm = (
        white[:, None, :]
        + depths[None, :, None]
        * (pial - white)[:, None, :]
    )

    coordinates_python = coordinates_spm.reshape(-1, 3).T

    thickness_vox = np.linalg.norm(
        pial - white,
        axis=1,
    ).astype(np.float32)

    return coordinates_python, thickness_vox


def ensure_inputs(fs_subject, surf_dir):
    required_files = [
        T1_FILE,
        surf_dir / "lh.white",
        surf_dir / "rh.white",
        surf_dir / "lh.pial",
        surf_dir / "rh.pial",
    ]

    for path in required_files:
        if not path.is_file():
            die(f"Missing required file:\n{path}")

    if not fs_subject.is_dir():
        die(f"Missing FreeSurfer subject directory:\n{fs_subject}")

    for run_dir in RUN_DIRS:
        if not run_dir.is_dir():
            die(f"Missing run directory:\n{run_dir}")


def validate_coordinates(
    coordinates,
    image_shape,
):
    for side in ("left", "right"):
        coords = coordinates[side]

        inside = (
            (coords[0] >= 0)
            & (coords[0] <= image_shape[0] - 1)
            & (coords[1] >= 0)
            & (coords[1] <= image_shape[1] - 1)
            & (coords[2] >= 0)
            & (coords[2] <= image_shape[2] - 1)
        )

        percentage_inside = 100.0 * inside.mean()

        log(
            f"{side}: "
            f"{percentage_inside:.2f}% "
            "sampling coordinates inside image"
        )

        if inside.mean() < 0.95:
            die(
                f"{side}: too many sampling points "
                "are outside the functional image."
            )


# ============================================================
# MAIN
# ============================================================

def main():
    log("Step 1 — Checking inputs")

    fs_subject = FS_ROOT / SUBJECT
    surf_dir = fs_subject / "surf"

    ensure_inputs(fs_subject, surf_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("Step 2 — Reading FreeSurfer surfaces")

    lh_white, _ = read_geometry(
        str(surf_dir / "lh.white")
    )
    rh_white, _ = read_geometry(
        str(surf_dir / "rh.white")
    )
    lh_pial, _ = read_geometry(
        str(surf_dir / "lh.pial")
    )
    rh_pial, _ = read_geometry(
        str(surf_dir / "rh.pial")
    )

    white_ras = np.vstack(
        (lh_white, rh_white)
    )
    pial_ras = np.vstack(
        (lh_pial, rh_pial)
    )

    if white_ras.shape != pial_ras.shape:
        die(
            "White and pial surfaces do not "
            "have matching dimensions."
        )

    n_vertices = white_ras.shape[0]

    log(
        f"Loaded {n_vertices:,} vertices "
        "across both hemispheres"
    )

    log("Step 3 — Transforming surfaces to T1 voxel space")

    t1_image = nib.load(str(T1_FILE))

    white_vox = ras_to_spm_vox(
        white_ras,
        t1_image.affine,
    )
    pial_vox = ras_to_spm_vox(
        pial_ras,
        t1_image.affine,
    )

    log("Step 4 — Preparing ROI and layer coordinates")

    roi_masks = {
        side: load_roi_mask(
            ROI_STEMS[side],
            n_vertices,
        )
        for side in ("left", "right")
    }

    coordinates = {}
    thickness = {}

    for side in ("left", "right"):
        coordinates[side], thickness[side] = (
            make_coordinates(
                white_vox,
                pial_vox,
                roi_masks[side],
            )
        )

        n_columns = int(
            roi_masks[side].sum()
        )

        log(
            f"{side}: {n_columns:,} columns "
            f"× {N_LAYERS} depths"
        )

    log("Step 5 — Ordering functional volumes")

    run_files = [
        ordered_run_files(run_dir)
        for run_dir in RUN_DIRS
    ]

    all_files = [
        path
        for run in run_files
        for path in run
    ]

    if len(all_files) != EXPECTED_TOTAL_VOLUMES:
        die(
            f"Expected {EXPECTED_TOTAL_VOLUMES} volumes, "
            f"found {len(all_files)}."
        )

    log(
        f"Validated {N_RUNS} runs × "
        f"{N_VOLUMES_PER_RUN} volumes = "
        f"{len(all_files)} volumes"
    )

    first_image = nib.load(str(all_files[0]))
    image_shape = first_image.shape[:3]

    validate_coordinates(
        coordinates,
        image_shape,
    )

    log("Step 6 — Creating HDF5 output")

    with h5py.File(OUTPUT_FILE, "w") as h5:
        h5.attrs["subject"] = SUBJECT
        h5.attrs["condition"] = "without_NORDIC"
        h5.attrs["n_layers"] = N_LAYERS
        h5.attrs["n_volumes"] = len(all_files)
        h5.attrs["interpolation_order"] = (
            INTERPOLATION_ORDER
        )
        h5.attrs["t1_reference"] = str(T1_FILE)
        h5.attrs["sampling_direction"] = (
            "depth 0 = white; depth 19 = pial"
        )
        h5.attrs["deep75_depth_indices_zero_based"] = (
            "0 through 14"
        )

        h5.create_dataset(
            "volume_files",
            data=np.array(
                [str(path) for path in all_files],
                dtype=h5py.string_dtype(),
            ),
        )

        h5.create_dataset(
            "run_lengths",
            data=np.array(
                [N_VOLUMES_PER_RUN] * N_RUNS,
                dtype=np.int32,
            ),
        )

        for side in ("left", "right"):
            n_columns = int(
                roi_masks[side].sum()
            )

            group = h5.create_group(side)

            group.create_dataset(
                "surface_vertex_indices",
                data=np.flatnonzero(
                    roi_masks[side]
                ).astype(np.int64),
            )

            group.create_dataset(
                "column_thickness_vox",
                data=thickness[side],
            )

            group.create_dataset(
                "layeractivation",
                shape=(
                    n_columns,
                    N_LAYERS,
                    len(all_files),
                ),
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )

            group.create_dataset(
                "columnspecificts",
                shape=(
                    n_columns,
                    len(all_files),
                ),
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )

            group.create_dataset(
                "all_columns_average",
                shape=(
                    N_LAYERS,
                    len(all_files),
                ),
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )

        log(
            "Step 7 — Sampling functional volumes"
        )

        for vol_idx, file_path in enumerate(
            tqdm(
                all_files,
                desc="Sampling without NORDIC",
                unit="volume",
            )
        ):
            image = nib.load(str(file_path))

            if (
                image.shape[:3] != t1_image.shape[:3]
                or not np.allclose(
                    image.affine,
                    t1_image.affine,
                    rtol=0,
                    atol=1e-5,
                )
            ):
                die(
                    f"Functional image grid differs from T1:\n{file_path}"
                )

            data = image.get_fdata(dtype=np.float32)

            for side in ("left", "right"):
                n_columns = int(
                    roi_masks[side].sum()
                )

                values = map_coordinates(
                    data,
                    coordinates[side],
                    order=INTERPOLATION_ORDER,
                    mode="constant",
                    cval=np.nan,
                    prefilter=True,
                )

                values = values.reshape(
                    n_columns,
                    N_LAYERS,
                ).astype(np.float32)

                deep75 = np.nanmean(
                    values[:, :15],
                    axis=1,
                    dtype=np.float32,
                )

                all_columns_average = np.nanmean(
                    values,
                    axis=0,
                    dtype=np.float32,
                )

                h5[side][
                    "layeractivation"
                ][:, :, vol_idx] = values

                h5[side][
                    "columnspecificts"
                ][:, vol_idx] = deep75

                h5[side][
                    "all_columns_average"
                ][:, vol_idx] = (
                    all_columns_average
                )

            if (vol_idx + 1) % 25 == 0:
                h5.flush()
                log(
                    f"Checkpoint: "
                    f"{vol_idx + 1}/{len(all_files)}"
                )

    log("Completed successfully")
    log(f"Output file:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"ERROR: {error}")
        sys.exit(1)