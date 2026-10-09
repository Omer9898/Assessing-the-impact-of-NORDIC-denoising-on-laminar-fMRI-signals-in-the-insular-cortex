from builtins import RuntimeError
import gzip
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

EXPECTED_VOLUMES = 407
FIRST_LABEL = 1000
LAST_LABEL = FIRST_LABEL + EXPECTED_VOLUMES - 1
TR_SECONDS = 3.0  # Change to the real acquisition TR if different.

RUNS = {
    1: Path(r"D:\Downloads\run1\func"),
    2: Path(r"D:\Downloads\run2\func"),
    3: Path(r"D:\Downloads\run3\func"),
}

OUT_DIR = Path(r"D:\NORDIC_Subject1_project\Motion and Regstration without Nordic")


def get_files(run_number, input_dir):
    prefix = f"mag_POCS_r{run_number}_"
    pattern = re.compile(
        rf"^{re.escape(prefix)}(\d+)_Warped-to-Anat\.nii\.gz$"
    )
    found = {}
    for path in input_dir.glob(f"{prefix}*_Warped-to-Anat.nii.gz"):
        match = pattern.match(path.name)
        if match:
            found[int(match.group(1))] = path

    expected_labels = set(range(FIRST_LABEL, LAST_LABEL + 1))
    actual_labels = set(found)
    missing = sorted(expected_labels - actual_labels)
    extra = sorted(actual_labels - expected_labels)

    if missing or extra or len(found) != EXPECTED_VOLUMES:
        raise RuntimeError(
            f"Run {run_number}: expected labels {FIRST_LABEL}-{LAST_LABEL}. "
            f"Missing={missing[:10]}, extra={extra[:10]}, found={len(found)}"
        )

    return [found[label] for label in range(FIRST_LABEL, LAST_LABEL + 1)]


def write_4d(files, output_file):
    first = nib.load(str(files[0]), mmap=True)
    if len(first.shape) != 3:
        raise RuntimeError(f"First input is not 3-D: {files[0]}")

    shape3d = first.shape
    affine = first.affine
    header = first.header.copy()
    header.set_data_dtype(np.float32)
    header.set_data_shape(shape3d + (EXPECTED_VOLUMES,))
    header.set_zooms(tuple(first.header.get_zooms()[:3]) + (TR_SECONDS,))
    header.set_slope_inter(1.0, 0.0)
    header["vox_offset"] = 352.0
    header["magic"] = b"n+1\x00"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_file, "wb", compresslevel=1) as out:
        out.write(header.binaryblock)
        out.write(b"\x00\x00\x00\x00")

        for index, path in enumerate(files, start=1):
            image = nib.load(str(path), mmap=True)
            if image.shape != shape3d:
                raise RuntimeError(
                    f"Different shape in {path.name}: {image.shape}; expected {shape3d}"
                )
            if not np.allclose(image.affine, affine, atol=1e-4):
                raise RuntimeError(f"Different affine in {path.name}")
            if not np.allclose(
                image.header.get_zooms()[:3], first.header.get_zooms()[:3], atol=1e-6
            ):
                raise RuntimeError(f"Different voxel size in {path.name}")

            volume = image.get_fdata(dtype=np.float32)
            out.write(np.asfortranarray(volume).tobytes(order="F"))
            print(f"{output_file.name}: {index}/{EXPECTED_VOLUMES}", end="\r", flush=True)

    print(f"\nFinished: {output_file}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for run_number, input_dir in RUNS.items():
        files = get_files(run_number, input_dir)
        output = OUT_DIR / f"magnitude_run{run_number}_Warped-to-Anat_4D.nii.gz"
        print(f"\nRun {run_number}: {len(files)} volumes")
        write_4d(files, output)

    print("\nAll three 4-D magnitude files were created successfully.")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
