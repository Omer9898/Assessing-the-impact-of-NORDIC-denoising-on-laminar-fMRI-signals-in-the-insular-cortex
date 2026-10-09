import os
import re
import glob
import gzip
import sys
import numpy as np
import nibabel as nib

INPUT_DIR = r"D:\NORDIC_Subject1_project\Motion and Regstration with Nordic\final_warps_run3"
OUTPUT_FILE = os.path.join(INPUT_DIR, "run3_4D.nii.gz")
EXPECTED_VOLUMES = 407
TR_SECONDS = 1.0

files = glob.glob(os.path.join(INPUT_DIR, "NORDIC_*_Warped-to-Anat.nii.gz"))

def label(path):
    m = re.search(r"NORDIC_(\d+)_Warped-to-Anat\.nii\.gz$", os.path.basename(path))
    if not m:
        raise ValueError(f"Bad filename: {path}")
    return int(m.group(1))

files.sort(key=label)
if len(files) != EXPECTED_VOLUMES:
    sys.exit(f"ERROR: found {len(files)} files, expected {EXPECTED_VOLUMES}.")

first = nib.load(files[0])
shape3d = first.shape
header = first.header.copy()
header.set_data_dtype(np.float32)
header.set_data_shape(shape3d + (EXPECTED_VOLUMES,))
header.set_zooms(first.header.get_zooms()[:3] + (TR_SECONDS,))
header.set_slope_inter(1.0, 0.0)
header["vox_offset"] = 352.0
header["magic"] = b"n+1\x00"

with gzip.open(OUTPUT_FILE, "wb") as out:
    out.write(header.binaryblock)
    out.write(b"\x00\x00\x00\x00")
    for i, path in enumerate(files):
        image = nib.load(path)
        if image.shape != shape3d or not np.allclose(image.affine, first.affine, atol=1e-4):
            sys.exit(f"ERROR: different geometry: {os.path.basename(path)}")
        out.write(np.asfortranarray(image.get_fdata(dtype=np.float32)).tobytes(order="F"))
        print(f"Progress: {i+1}/{EXPECTED_VOLUMES} ({(i+1)*100/EXPECTED_VOLUMES:.1f}%)", end="\r", flush=True)

print("\nFINISHED")
print(OUTPUT_FILE)
