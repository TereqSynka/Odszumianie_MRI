import os
import gzip
import nibabel as nib
import numpy as np
import cv2

SOURCE_DIR = "C:/Users/PC/Desktop/Odszumianie-main"
OUTPUT_FOLDER = "normalne"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

nii_files = [
    os.path.join(SOURCE_DIR, f) 
    for f in os.listdir(SOURCE_DIR) 
    if f.endswith(".nii.gz") or f.endswith(".nii")
][:10]

print(f"Znaleziono {len(nii_files)} plików.")

for file_path in nii_files:
    print("Przetwarzam:", os.path.basename(file_path))

    if file_path.endswith(".gz"):
        img = nib.load(file_path)
    else:
        img = nib.load(file_path)

    data = img.get_fdata()

    patient_name = os.path.basename(file_path).replace(".nii.gz", "").replace(".nii", "")

    slices = {
        "axial": data[:, :, data.shape[2] // 2],
        "coronal": data[:, data.shape[1] // 2, :],
        "sagittal": data[data.shape[0] // 2, :, :]
    }

    for view_name, img_slice in slices.items():
        view_folder = os.path.join(OUTPUT_FOLDER, view_name)
        os.makedirs(view_folder, exist_ok=True)

        img_slice = (
            (img_slice - img_slice.min()) /
            (img_slice.max() - img_slice.min())
            * 255
        ).astype(np.uint8)

        cv2.imwrite(
            os.path.join(
                view_folder,
                f"{patient_name}.png"
            ),
            img_slice
        )

print("Gotowe.")