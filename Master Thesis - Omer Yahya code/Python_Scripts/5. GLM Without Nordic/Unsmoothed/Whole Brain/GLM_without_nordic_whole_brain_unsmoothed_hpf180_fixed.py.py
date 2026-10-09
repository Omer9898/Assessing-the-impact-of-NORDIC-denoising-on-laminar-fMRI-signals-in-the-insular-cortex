from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.contrasts import (
    compute_contrast,
    expression_to_contrast_vector,
)
from nilearn.glm.first_level import (
    make_first_level_design_matrix,
    run_glm,
)


# =============================================================================
# PARAMETERS
# =============================================================================

TR = 3.0
N_VOLUMES = 407
HIGH_PASS_PERIOD_SECONDS = 180.0
HIGH_PASS_HZ = 1.0 / HIGH_PASS_PERIOD_SECONDS

SLAB_Z = 8

# Explicitly unsmoothed.
SMOOTHING_FWHM_MM = 0.0


# =============================================================================
# EDIT THESE PATHS
# =============================================================================

MASK_SOURCE = Path(
    r"D:\Downloads\brain_mask.nii.gz"
)

OUTPUT_BASE = Path(
    r"D:\NORDIC_Subject1_project\GLM_without_nordic\unsmoothed\whole brain"
)

RUN_OUTPUT = OUTPUT_BASE / (
    "GLM_without_nordic_whole_brain_unsmoothed_hpf180_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
)


RUNS = {
    1: {
        "func": Path(
            r"D:\NORDIC_Subject1_project\Motion and Regstration without Nordic"
            r"\magnitude_run1_Warped-to-Anat_4D.nii.gz"
        ),
        "events": Path(
            r"D:\NORDIC_Subject1_project\GLM_without_nordic"
            r"\events_run1_used.csv"
        ),
        "motion": Path(
            r"D:\Downloads\run1\func\mag_POCS_r1_MoCorr.txt"
        ),
        "compcor": Path(
            r"D:\Downloads\run1\func\compcor_regressors_r1.txt"
        ),
    },
    2: {
        "func": Path(
            r"D:\NORDIC_Subject1_project\Motion and Regstration without Nordic"
            r"\magnitude_run2_Warped-to-Anat_4D.nii.gz"
        ),
        "events": Path(
            r"D:\NORDIC_Subject1_project\GLM_without_nordic"
            r"\events_run2_used.csv"
        ),
        "motion": Path(
            r"D:\Downloads\run2\func\mag_POCS_r2_MoCorr.txt"
        ),
        "compcor": Path(
            r"D:\Downloads\run2\func\compcor_regressors_r2.txt"
        ),
    },
    3: {
        "func": Path(
            r"D:\NORDIC_Subject1_project\Motion and Regstration without Nordic"
            r"\magnitude_run3_Warped-to-Anat_4D.nii.gz"
        ),
        "events": Path(
            r"D:\NORDIC_Subject1_project\GLM_without_nordic"
            r"\events_run3_used.csv"
        ),
        "motion": Path(
            r"D:\Downloads\run3\func\mag_POCS_r3_MoCorr.txt"
        ),
        "compcor": Path(
            r"D:\Downloads\run3\func\compcor_regressors_r3.txt"
        ),
    },
}


# =============================================================================
# CONTRASTS
# =============================================================================

CONTRASTS = {
    "low_cognition_minus_high_cognition": (
        "pain_low_cogn_high_pain + pain_low_cogn_low_pain "
        "- pain_high_cogn_high_pain - pain_high_cogn_low_pain"
    ),
    "high_pain_minus_low_pain": (
        "pain_high_cogn_high_pain - pain_high_cogn_low_pain "
        "+ pain_low_cogn_high_pain - pain_low_cogn_low_pain"
    ),
}


REQUIRED_CONDITIONS = {
    "pain_high_cogn_high_pain",
    "pain_high_cogn_low_pain",
    "pain_low_cogn_high_pain",
    "pain_low_cogn_low_pain",
}


# =============================================================================
# HELPERS
# =============================================================================

def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found:\n{path}")


def save_nifti(
    data: np.ndarray,
    reference: nib.Nifti1Image,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    header = reference.header.copy()
    header.set_data_shape(data.shape)
    header.set_data_dtype(np.float32)

    image = nib.Nifti1Image(
        np.asarray(data, dtype=np.float32),
        reference.affine,
        header,
    )
    image.to_filename(str(path))


# =============================================================================
# EVENTS AND CONFOUNDS
# =============================================================================

def load_events(path: Path) -> pd.DataFrame:
    require_file(path, "Events file")

    events = pd.read_csv(path, sep=None, engine="python")
    events.columns = [
        str(column).strip().lower()
        for column in events.columns
    ]

    required_columns = {"onset", "duration", "trial_type"}
    missing_columns = required_columns.difference(events.columns)

    if missing_columns:
        raise RuntimeError(
            f"Missing event columns {sorted(missing_columns)}:\n{path}"
        )

    events = events[
        ["onset", "duration", "trial_type"]
    ].copy()

    events["onset"] = pd.to_numeric(
        events["onset"],
        errors="raise",
    )
    events["duration"] = pd.to_numeric(
        events["duration"],
        errors="raise",
    )
    events["trial_type"] = (
        events["trial_type"]
        .astype(str)
        .str.strip()
    )

    missing_conditions = REQUIRED_CONDITIONS.difference(
        set(events["trial_type"])
    )

    if missing_conditions:
        raise RuntimeError(
            f"Missing required conditions "
            f"{sorted(missing_conditions)}:\n{path}"
        )

    return events


def load_numeric(
    path: Path,
    label: str,
    n_rows: int,
) -> pd.DataFrame:
    require_file(path, label)

    table = pd.read_csv(
        path,
        sep=None,
        engine="python",
        header=None,
    )

    table = table.apply(
        pd.to_numeric,
        errors="coerce",
    )

    table = table.dropna(
        axis=0,
        how="all",
    ).dropna(
        axis=1,
        how="all",
    )

    if table.isna().any().any():
        raise RuntimeError(
            f"{label} contains non-numeric values:\n{path}"
        )

    if len(table) != n_rows:
        raise RuntimeError(
            f"{label} has {len(table)} rows; "
            f"expected {n_rows}:\n{path}"
        )

    return table.reset_index(drop=True)


def make_design(run_paths: dict) -> pd.DataFrame:
    events = load_events(run_paths["events"])

    motion = load_numeric(
        run_paths["motion"],
        "Motion file",
        N_VOLUMES,
    )

    compcor = load_numeric(
        run_paths["compcor"],
        "CompCor file",
        N_VOLUMES,
    )

    if motion.shape[1] != 6:
        raise RuntimeError(
            f"Motion file must have 6 columns; "
            f"found {motion.shape[1]}:\n"
            f"{run_paths['motion']}"
        )

    motion.columns = [
        f"motion_{index:02d}"
        for index in range(1, motion.shape[1] + 1)
    ]

    compcor.columns = [
        f"compcor_{index:02d}"
        for index in range(1, compcor.shape[1] + 1)
    ]

    confounds = pd.concat(
        [motion, compcor],
        axis=1,
    )

    standard_deviation = confounds.std(
        axis=0,
        ddof=0,
    )

    constant_columns = list(
        standard_deviation[
            standard_deviation <= 1e-12
        ].index
    )

    if constant_columns:
        print(
            "Removing constant confounds:",
            constant_columns,
        )

        confounds = confounds.drop(
            columns=constant_columns
        )

        standard_deviation = confounds.std(
            axis=0,
            ddof=0,
        )

    confounds = (
        confounds - confounds.mean(axis=0)
    ) / standard_deviation

    frame_times = (
        np.arange(N_VOLUMES, dtype=float) * TR
        + TR / 2.0
    )

    return make_first_level_design_matrix(
        frame_times=frame_times,
        events=events,
        hrf_model="spm",
        drift_model="cosine",
        high_pass=HIGH_PASS_HZ,
        add_regs=confounds.to_numpy(dtype=float),
        add_reg_names=list(confounds.columns),
    )


# =============================================================================
# WHOLE-BRAIN MASK
# =============================================================================

def create_binary_whole_brain_mask(
    reference: nib.Nifti1Image,
    output_path: Path,
) -> np.ndarray:
    require_file(
        MASK_SOURCE,
        "Whole-brain mask",
    )

    mask_image = nib.load(str(MASK_SOURCE))

    if mask_image.shape[:3] != reference.shape[:3]:
        raise RuntimeError(
            f"Whole-brain mask shape "
            f"{mask_image.shape[:3]} does not match "
            f"functional shape {reference.shape[:3]}."
        )

    if not np.allclose(
        mask_image.affine,
        reference.affine,
        atol=1e-4,
    ):
        raise RuntimeError(
            "Whole-brain mask affine does not match "
            "the functional affine."
        )

    mask_data = np.asarray(
        mask_image.dataobj,
        dtype=np.float32,
    )

    binary_mask = (
        np.isfinite(mask_data)
        & (mask_data > 0)
    )

    if not binary_mask.any():
        raise RuntimeError(
            "The whole-brain mask is empty."
        )

    save_nifti(
        binary_mask.astype(np.float32),
        reference,
        output_path,
    )

    print(
        f"Binary whole-brain mask saved:\n{output_path}"
    )
    print(
        f"Brain-mask voxels: {int(binary_mask.sum())}"
    )
    print(
        f"Mask coverage: "
        f"{100.0 * binary_mask.mean():.4f}%"
    )

    return binary_mask


# =============================================================================
# RUN-WISE UNSMOOTHED GLM
# =============================================================================

def estimate_run(
    image: nib.Nifti1Image,
    design: pd.DataFrame,
    binary_mask: np.ndarray,
    run: int,
    output_folder: Path,
) -> None:
    shape = image.shape[:3]
    x_matrix = design.to_numpy(dtype=float)

    contrast_vectors = {
        name: expression_to_contrast_vector(
            expression,
            design.columns,
        )
        for name, expression in CONTRASTS.items()
    }

    maps = {
        name: {
            kind: np.zeros(
                shape,
                dtype=np.float32,
            )
            for kind in (
                "effect_size",
                "variance",
                "t_stat",
            )
        }
        for name in CONTRASTS
    }

    print()
    print(f"Starting unsmoothed GLM for Run {run}")
    print("Whole-brain analysis; no spatial smoothing.")
    print("Noise model: AR(1)")

    for z0 in range(0, shape[2], SLAB_Z):
        z1 = min(z0 + SLAB_Z, shape[2])

        core_mask = binary_mask[:, :, z0:z1]

        if not core_mask.any():
            continue

        n_voxels = int(core_mask.sum())

        y = np.empty(
            (N_VOLUMES, n_voxels),
            dtype=np.float32,
        )

        for timepoint in range(N_VOLUMES):
            volume = np.asarray(
                image.dataobj[:, :, z0:z1, timepoint],
                dtype=np.float32,
            )

            y[timepoint] = volume[core_mask]

        means = y.mean(axis=0)

        good = (
            np.isfinite(y).all(axis=0)
            & (y.std(axis=0) > 1e-8)
            & (np.abs(means) > 1e-8)
        )

        if good.any():
            y_percent = (
                100.0
                * (y[:, good] / means[good] - 1.0)
            )

            labels, results = run_glm(
                y_percent,
                x_matrix,
                noise_model="ar1",
                n_jobs=1,
                verbose=0,
            )

            all_indices = np.where(core_mask)

            good_indices = tuple(
                axis[good]
                for axis in all_indices
            )

            for name, vector in contrast_vectors.items():
                contrast = compute_contrast(
                    labels,
                    results,
                    vector,
                    stat_type="t",
                )

                values = {
                    "effect_size": np.asarray(
                        contrast.effect_size(),
                        dtype=np.float32,
                    ).ravel(),
                    "variance": np.asarray(
                        contrast.effect_variance(),
                        dtype=np.float32,
                    ).ravel(),
                    "t_stat": np.asarray(
                        contrast.stat(),
                        dtype=np.float32,
                    ).ravel(),
                }

                for kind, values_1d in values.items():
                    slab_map = np.zeros(
                        core_mask.shape,
                        dtype=np.float32,
                    )

                    slab_map[good_indices] = values_1d

                    maps[name][kind][
                        :,
                        :,
                        z0:z1,
                    ] = slab_map

        del y
        gc.collect()

        print(
            f"Run {run}: processed "
            f"z={z0}:{z1} / {shape[2]}",
            flush=True,
        )

    for contrast_name in CONTRASTS:
        for kind in maps[contrast_name]:
            output_path = (
                output_folder
                / f"{contrast_name}_{kind}.nii.gz"
            )

            save_nifti(
                maps[contrast_name][kind],
                image,
                output_path,
            )


# =============================================================================
# FIXED EFFECTS
# =============================================================================

def calculate_fixed_effects(
    reference: nib.Nifti1Image,
    binary_mask: np.ndarray,
    contrast_name: str,
    fixed_folder: Path,
) -> None:
    effect_images = [
        nib.load(
            str(
                RUN_OUTPUT
                / f"run{run}"
                / f"{contrast_name}_effect_size.nii.gz"
            )
        )
        for run in (1, 2, 3)
    ]

    variance_images = [
        nib.load(
            str(
                RUN_OUTPUT
                / f"run{run}"
                / f"{contrast_name}_variance.nii.gz"
            )
        )
        for run in (1, 2, 3)
    ]

    shape = reference.shape[:3]

    fixed_effect = np.zeros(
        shape,
        dtype=np.float32,
    )
    fixed_variance = np.zeros(
        shape,
        dtype=np.float32,
    )
    fixed_t = np.zeros(
        shape,
        dtype=np.float32,
    )
    valid_mask = np.zeros(
        shape,
        dtype=np.uint8,
    )

    print()
    print(
        f"Calculating fixed effects: "
        f"{contrast_name}"
    )

    for z0 in range(0, shape[2], SLAB_Z):
        z1 = min(z0 + SLAB_Z, shape[2])

        effects = np.stack(
            [
                np.asarray(
                    image.dataobj[:, :, z0:z1],
                    dtype=np.float32,
                )
                for image in effect_images
            ],
            axis=0,
        )

        variances = np.stack(
            [
                np.asarray(
                    image.dataobj[:, :, z0:z1],
                    dtype=np.float32,
                )
                for image in variance_images
            ],
            axis=0,
        )

        valid = (
            binary_mask[:, :, z0:z1]
            & np.isfinite(effects).all(axis=0)
            & np.isfinite(variances).all(axis=0)
            & (variances > 0).all(axis=0)
        )

        weights = np.zeros_like(
            variances,
            dtype=np.float32,
        )

        np.divide(
            1.0,
            variances,
            out=weights,
            where=valid[None, ...],
        )

        weight_sum = weights.sum(axis=0)

        effect_slab = np.zeros(
            valid.shape,
            dtype=np.float32,
        )
        variance_slab = np.zeros(
            valid.shape,
            dtype=np.float32,
        )
        t_slab = np.zeros(
            valid.shape,
            dtype=np.float32,
        )

        weighted_effect = (
            weights * effects
        ).sum(axis=0)

        effect_slab[valid] = (
            weighted_effect[valid]
            / weight_sum[valid]
        )

        variance_slab[valid] = (
            1.0 / weight_sum[valid]
        )

        t_slab[valid] = (
            effect_slab[valid]
            / np.sqrt(variance_slab[valid])
        )

        fixed_effect[
            :,
            :,
            z0:z1,
        ] = effect_slab

        fixed_variance[
            :,
            :,
            z0:z1,
        ] = variance_slab

        fixed_t[
            :,
            :,
            z0:z1,
        ] = t_slab

        valid_mask[
            :,
            :,
            z0:z1,
        ] = valid.astype(np.uint8)

        print(
            f"Fixed effects {contrast_name}: "
            f"z={z0}:{z1} / {shape[2]}",
            flush=True,
        )

    save_nifti(
        fixed_effect,
        reference,
        fixed_folder
        / f"{contrast_name}_FIXED_EFFECTS_effect_size.nii.gz",
    )

    save_nifti(
        fixed_variance,
        reference,
        fixed_folder
        / f"{contrast_name}_FIXED_EFFECTS_variance.nii.gz",
    )

    save_nifti(
        fixed_t,
        reference,
        fixed_folder
        / f"{contrast_name}_FIXED_EFFECTS_t_stat.nii.gz",
    )

    save_nifti(
        valid_mask,
        reference,
        fixed_folder
        / f"{contrast_name}_FIXED_EFFECTS_valid_mask.nii.gz",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    RUN_OUTPUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    fixed_folder = RUN_OUTPUT / "fixed_effects"
    fixed_folder.mkdir()

    images = {}
    designs = {}
    reference = None

    print("=" * 78)
    print(
        "WHOLE-BRAIN GLM WITHOUT NORDIC "
        "- UNSMOOTHED - HPF 180 s"
    )
    print("=" * 78)
    print(f"Output: {RUN_OUTPUT}")

    for run in (1, 2, 3):
        paths = RUNS[run]

        require_file(
            paths["func"],
            f"Functional Run {run}",
        )

        image = nib.load(str(paths["func"]))

        if (
            len(image.shape) != 4
            or image.shape[3] != N_VOLUMES
        ):
            raise RuntimeError(
                f"Run {run} must be 4D with "
                f"{N_VOLUMES} volumes; "
                f"found {image.shape}"
            )

        print(
            f"Run {run}: shape={image.shape}; "
            f"header TR={image.header.get_zooms()[3]}; "
            f"model TR={TR}",
            flush=True,
        )

        if reference is None:
            reference = image
        else:
            if image.shape[:3] != reference.shape[:3]:
                raise RuntimeError(
                    f"Run {run} has a different spatial shape."
                )

            if not np.allclose(
                image.affine,
                reference.affine,
                atol=1e-4,
            ):
                raise RuntimeError(
                    f"Run {run} has a different affine."
                )

        images[run] = image
        designs[run] = make_design(paths)

    binary_mask = create_binary_whole_brain_mask(
        reference,
        RUN_OUTPUT / "brain_mask_binary_checked.nii.gz",
    )

    for run in (1, 2, 3):
        run_folder = RUN_OUTPUT / f"run{run}"
        run_folder.mkdir()

        designs[run].to_csv(
            run_folder / "design_matrix.csv",
            index=False,
        )

        estimate_run(
            images[run],
            designs[run],
            binary_mask,
            run,
            run_folder,
        )

    for contrast_name in CONTRASTS:
        calculate_fixed_effects(
            reference,
            binary_mask,
            contrast_name,
            fixed_folder,
        )

    metadata = {
        "analysis": (
            "GLM without Nordic, whole brain, "
            "unsmoothed"
        ),
        "method": (
            "run-wise slab GLM with "
            "inverse-variance fixed effects"
        ),
        "TR_seconds": TR,
        "volumes_per_run": N_VOLUMES,
        "smoothing_fwhm_mm": 0.0,
        "high_pass_period_seconds": (
            HIGH_PASS_PERIOD_SECONDS
        ),
        "high_pass_hz": HIGH_PASS_HZ,
        "hrf_model": "spm",
        "drift_model": "cosine",
        "noise_model": "ar1",
        "mask_source": str(MASK_SOURCE),
        "output_root": str(RUN_OUTPUT),
        "contrasts": CONTRASTS,
    }

    (
        fixed_folder / "analysis_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("COMPLETED SUCCESSFULLY")
    print(f"Output root: {RUN_OUTPUT}")
    print(f"Fixed-effects folder: {fixed_folder}")


if __name__ == "__main__":
    main()