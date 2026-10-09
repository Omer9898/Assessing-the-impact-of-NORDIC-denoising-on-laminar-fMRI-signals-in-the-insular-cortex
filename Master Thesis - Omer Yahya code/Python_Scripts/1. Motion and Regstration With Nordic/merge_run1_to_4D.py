import os
import re
import glob
import gzip
import sys
import numpy as np
import nibabel as nib

INPUT_DIR = r"D:\NORDIC_Subject1_project\Motion and Regstration with Nordic\final_warps_run1"
OUTPUT_FILE = os.path.join(INPUT_DIR, "rebuild_run1_4D.nii.gz")
EXPECTED_VOLUMES = 407
TR_SECONDS = 3.0  # CHANGE THIS to the real TR from your scan protocol.

pattern = os.path.join(INPUT_DIR, "NORDIC_*_Warped-to-Anat.nii.gz")
files = glob.glob(pattern)

def label(path):
    match = re.search(r"NORDIC_(\d+)_Warped-to-Anat\.nii\.gz$", os.path.basename(path))
    if not match:
        raise ValueError(f"Bad filename: {path}")
    return int(match.group(1))

files.sort(key=label)

if len(files) != EXPECTED_VOLUMES:
    sys.exit(f"ERROR: found {len(files)} files, expected {EXPECTED_VOLUMES}.")

labels = [label(p) for p in files]
if labels != list(range(labels[0], labels[0] + EXPECTED_VOLUMES)):
    sys.exit("ERROR: volume labels are not continuous.")

first = nib.load(files[0])
shape3d = first.shape
if len(shape3d) != 3:
    sys.exit("ERROR: input files must be 3-D.")

dtype = np.dtype(np.float32)
header = first.header.copy()
header.set_data_dtype(dtype)
header.set_data_shape(shape3d + (EXPECTED_VOLUMES,))
zooms = first.header.get_zooms()[:3]
header.set_zooms(tuple(zooms) + (TR_SECONDS,))
header.set_slope_inter(1.0, 0.0)
header["vox_offset"] = 352.0
header["magic"] = b"n+1\x00"

print(f"Merging {EXPECTED_VOLUMES} volumes...")
with gzip.open(OUTPUT_FILE, "wb") as out:
    out.write(header.binaryblock)
    out.write(b"\x00\x00\x00\x00")

    for i, path in enumerate(files):
        image = nib.load(path)
        if image.shape != shape3d or not np.allclose(image.affine, first.affine, atol=1e-4):
            sys.exit(f"ERROR: different geometry at {os.path.basename(path)}")

        volume = image.get_fdata(dtype=np.float32)
        out.write(np.asfortranarray(volume, dtype=dtype).tobytes(order="F"))

        percent = (i + 1) * 100 / EXPECTED_VOLUMES
        print(f"Progress: {i + 1}/{EXPECTED_VOLUMES} ({percent:.1f}%)", end="\r", flush=True)

print("\nFINISHED")
print(OUTPUT_FILE)
