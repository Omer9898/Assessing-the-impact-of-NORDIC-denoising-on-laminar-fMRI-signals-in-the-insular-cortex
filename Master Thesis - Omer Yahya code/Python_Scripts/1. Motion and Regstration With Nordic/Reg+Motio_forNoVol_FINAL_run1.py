import os
import sys
import time
import gc
import argparse
import ants
import nibabel as nib

NORDIC_4D = r"D:\FINAL RUN 1\OY - Files after Nordic\output_nordic_1\NORDIC_mag_POCS_r1.nii.gz"
FUNC_DIR = r"D:\Downloads\run1\func"
MATS_DIR = r"D:\Downloads\run1\func\run1_mats\run1_mats"
ANAT_REF = r"D:\Downloads\run1\func\mag_POCS_r1_mean_MotDistCor_anatomySpace.nii.gz"
OUT_DIR = r"D:\NORDIC_Subject1_project\Motion and Regstration with Nordic\final_warps"

PREFIX = "mag_POCS_r1"
FIRST_LABEL = 1000
N_VOLUMES = 407
INTERPOLATOR = "linear"

DIST_AFFINE = os.path.join(FUNC_DIR, "mag_POCS_r1_DistCorr_00GenericAffine.mat")
DIST_WARP = os.path.join(FUNC_DIR, "mag_POCS_r1_DistCorr_01Warp.nii.gz")
CUSTOM_REG = os.path.join(FUNC_DIR, "custom_reg2anat.txt")
KNOWN_1000 = os.path.join(FUNC_DIR, "mag_POCS_r1_1000_Warped-to-Anat.nii.gz")


def motion_path(label):
    return os.path.join(MATS_DIR, f"{PREFIX}_{label}_0GenericAffine.mat")


def check_files():
    required = [NORDIC_4D, ANAT_REF, DIST_AFFINE, DIST_WARP, CUSTOM_REG, KNOWN_1000]
    required += [motion_path(FIRST_LABEL + i) for i in range(N_VOLUMES)]
    missing = [p for p in required if not os.path.isfile(p)]
    if missing:
        sys.exit("MISSING:\n" + "\n".join(missing[:10]))
    shape = nib.load(NORDIC_4D).shape
    if shape != (210, 210, 110, N_VOLUMES):
        sys.exit(f"Unexpected NORDIC shape: {shape}")


def warp_volume(series, index, fixed):
    label = FIRST_LABEL + index
    moving = ants.slice_image(series, axis=3, idx=index)
    try:
        # ANTs applies the LAST transform first.
        # Actual order: motion -> DistCorr affine -> DistCorr warp -> custom_reg2anat.
        return ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=[CUSTOM_REG, DIST_WARP, DIST_AFFINE, motion_path(label)],
            whichtoinvert=[False, False, False, False],
            interpolator=INTERPOLATOR,
        )
    finally:
        del moving


def test_one():
    check_files()
    series = ants.image_read(NORDIC_4D)
    fixed = ants.image_read(KNOWN_1000)
    result = warp_volume(series, 0, fixed)
    os.makedirs(OUT_DIR, exist_ok=True)
    output = os.path.join(OUT_DIR, "TEST_NORDIC_1000_Warped-to-Anat.nii.gz")
    ants.image_write(result, output)
    print("TEST FINISHED")
    print(output)


def run_all():
    check_files()
    os.makedirs(OUT_DIR, exist_ok=True)
    series = ants.image_read(NORDIC_4D)
    fixed = ants.image_read(ANAT_REF)
    start = time.time()

    for i in range(N_VOLUMES):
        label = FIRST_LABEL + i
        result = warp_volume(series, i, fixed)

        output = os.path.join(
            OUT_DIR,
            f"NORDIC_{label}_Warped-to-Anat.nii.gz"
        )

        ants.image_write(result, output)

        del result
        gc.collect()

        percent = (i + 1) / N_VOLUMES * 100
        elapsed = (time.time() - start) / 60

        print(
            f"Progress: {i + 1}/{N_VOLUMES} "
            f"({percent:.1f}%) | {elapsed:.1f} min",
            end="\r",
            flush=True
        )

    print()
    print("FINISHED")
    print(OUT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-one", action="store_true")
    args = parser.parse_args()
    if args.test_one:
        test_one()
    else:
        run_all()