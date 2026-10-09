import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.plotting import plot_roi

# ============================================================
# ROI overlay check in anatomy space
# This script creates ONE overlay PNG per run so you can
# visually confirm that the left/right pIns ROI masks are
# correctly located on the anatomy-space functional image.
# ============================================================

# Runs to process
RUNS = [1, 2, 3]

# Folder for each run.
# Each folder should contain the anatomy-space mean functional image:
# mag_POCS_rX_mean_MotDistCor_anatomySpace.nii.gz
RUN_FUNC_DIRS = {
    1: r"D:/Downloads/run1/func",
    2: r"D:/Downloads/run2/func",
    3: r"D:/Downloads/run3/func",
}

# Left and right ROI masks in volume space
ROI_MASK_LEFT_VOL = r"D:/Downloads/pins left mask/roi_pInsleft_17.nii.gz"
ROI_MASK_RIGHT_VOL = r"D:/Downloads/pins right mask/roi_pInsright_118.nii.gz"

# Root output directory
OUTPUT_ROOT = r"D:\NORDIC_Subject1_project\Functional Mask\nordic_qc_python"


def ensure_output_dir(run: int) -> str:
    """
    Create one output folder per run.
    Example: .../run1, .../run2, .../run3
    """
    out = os.path.join(OUTPUT_ROOT, f"run{run}")
    os.makedirs(out, exist_ok=True)
    return out


def anat_space_reference(run: int) -> str:
    """
    Return the anatomy-space mean functional image for one run.
    This image is used as the grayscale background.
    """
    return os.path.join(
        RUN_FUNC_DIRS[run],
        f"mag_POCS_r{run}_mean_MotDistCor_anatomySpace.nii.gz"
    )


def main():
    # Check that mask files exist before starting
    if not os.path.exists(ROI_MASK_LEFT_VOL):
        raise FileNotFoundError(f"Left ROI mask not found: {ROI_MASK_LEFT_VOL}")
    if not os.path.exists(ROI_MASK_RIGHT_VOL):
        raise FileNotFoundError(f"Right ROI mask not found: {ROI_MASK_RIGHT_VOL}")

    # Loop through run1, run2, run3
    for run in RUNS:
        ref_path = anat_space_reference(run)

        if not os.path.exists(ref_path):
            print(f"[skip] run {run}: reference image not found -> {ref_path}")
            continue

        out_dir = ensure_output_dir(run)

        # --------------------------------------------------------
        # Step 1: Load the anatomy-space reference image
        # This is the background image you want to see in gray.
        # --------------------------------------------------------
        ref_img = nib.load(ref_path)

        # --------------------------------------------------------
        # Step 2: Load the left and right ROI masks
        # These are the original ROI NIfTI files.
        # --------------------------------------------------------
        left_mask_img = nib.load(ROI_MASK_LEFT_VOL)
        right_mask_img = nib.load(ROI_MASK_RIGHT_VOL)

        # --------------------------------------------------------
        # Step 3: Resample each mask to the reference image grid
        # Why?
        # Because the ROI mask and the anatomy-space image may not
        # have exactly the same shape, voxel size, or affine.
        #
        # Why nearest interpolation?
        # Because this is a binary/categorical mask, not a smooth image.
        # "nearest" preserves mask labels and avoids partial values.
        # --------------------------------------------------------
       
        # --------------------------------------------------------
        # Step 4: Convert resampled masks into binary arrays
        # Any voxel > 0 becomes part of the ROI.
        # --------------------------------------------------------
        left_data = (left_mask_img.get_fdata() > 0).astype(np.uint8)
        right_data = (right_mask_img.get_fdata() > 0).astype(np.uint8)

        # --------------------------------------------------------
        # Step 5: Build one combined overlay image
        #
        # overlay_data == 0  -> background / no ROI
        # overlay_data == 1  -> left ROI
        # overlay_data == 2  -> right ROI
        #
        # This allows both masks to appear in one output image.
        # --------------------------------------------------------
        overlay_data = np.zeros(left_data.shape, dtype=np.uint8)
        overlay_data[(left_data > 0) | (right_data > 0)] = 1

        # Save the combined overlay as a NIfTI image in memory
        overlay_img = nib.Nifti1Image(
            overlay_data,
            ref_img.affine,
            ref_img.header
        )

        # --------------------------------------------------------
        # Step 6: Plot ROI overlay on top of the reference image
        #
        # bg_img      -> anatomy-space mean functional image (gray)
        # overlay_img -> left/right ROI (colored)
        #
        # display_mode="z" and cut_coords=1:
        # show one axial slice automatically chosen by nilearn.
        # --------------------------------------------------------
        fig = plt.figure(figsize=(6, 8))

        display = plot_roi(
            roi_img=overlay_img,
            bg_img=ref_img,
            title=f"Run {run}: pIns ROI overlay in anat space",
            display_mode="z",
            cut_coords=1,
            draw_cross=False,
            black_bg=True,
            cmap="Reds",
            colorbar=False,
            figure=fig
        )

        # --------------------------------------------------------
        # Step 7: Save the final PNG
        # This is the file you inspect visually.
        # --------------------------------------------------------
        out_png = os.path.join(out_dir, f"roi_overlay_check_run{run}.png")
        display.savefig(out_png)

        # Close figures to avoid memory buildup
        display.close()
        plt.close(fig)

        print(f"[done] run {run}")
        print(f"  reference image : {ref_path}")
        print(f"  overlay output  : {out_png}")


if __name__ == "__main__":
    main()