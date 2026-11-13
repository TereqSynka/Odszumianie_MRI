import numpy as np
import pandas as pd
import os
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2

folder_path = "normalne"
valid_extensions = (".png", ".jpg")

image_files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.lower().endswith(valid_extensions)
]

if len(image_files) == 0:
    image_files = [None] * 30

noise_levels = [0.05, 0.15, 0.30]
base_noisy_folder = "zaszumione"
base_filtered_folder = "odszumione"

all_results = []

for idx, file_path in enumerate(image_files):
    if file_path is not None:
        original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        file_name_base = os.path.splitext(os.path.basename(file_path))[0]
    else:
        original_image = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        file_name_base = f"synth_{idx+1}"

    if original_image is None:
        continue

    original_float = original_image.astype(np.float64) / 255.0

    for std_dev in noise_levels:
        noisy_output_folder = os.path.join(base_noisy_folder, f"szum_{std_dev}")
        os.makedirs(noisy_output_folder, exist_ok=True)

        noise = np.random.normal(
            loc=0.0, scale=std_dev, size=original_float.shape
        )
        noisy_float = np.clip(original_float + noise, 0, 1)
        current_noisy = (noisy_float * 255).astype(np.uint8)

        noisy_filename = os.path.join(
            noisy_output_folder, f"{file_name_base}.png"
        )
        cv2.imwrite(noisy_filename, current_noisy)

        filters = {}
        times = {}

        t0 = time.perf_counter()
        filters["Gauss"] = cv2.GaussianBlur(current_noisy, (5, 5), sigmaX=1.5)
        times["Gauss"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        filters["Medianowy"] = cv2.medianBlur(current_noisy, ksize=5)
        times["Medianowy"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        filters["NLM"] = cv2.fastNlMeansDenoising(
            current_noisy, h=40, templateWindowSize=7, searchWindowSize=21
        )
        times["NLM"] = time.perf_counter() - t0

        for method_name, filtered_img in filters.items():
            filtered_output_folder = os.path.join(
                base_filtered_folder, f"szum_{std_dev}", method_name
            )
            os.makedirs(filtered_output_folder, exist_ok=True)

            filtered_filename = os.path.join(
                filtered_output_folder, f"{file_name_base}.png"
            )
            cv2.imwrite(filtered_filename, filtered_img)

            val_psnr = psnr(original_image, filtered_img)
            val_ssim = ssim(original_image, filtered_img, data_range=255)

            all_results.append(
                {
                    "Obraz_ID": idx + 1,
                    "Poziom_Szumu": std_dev,
                    "Metoda": method_name,
                    "PSNR": val_psnr,
                    "SSIM": val_ssim,
                    "Czas_s": times[method_name],
                }
            )

df_results = pd.DataFrame(all_results)

summary_table = (
    df_results.groupby(["Poziom_Szumu", "Metoda"])[
        ["PSNR", "SSIM", "Czas_s"]
    ]
    .mean()
    .reset_index()
)

summary_table["PSNR"] = summary_table["PSNR"].round(2)
summary_table["SSIM"] = summary_table["SSIM"].round(4)
summary_table["Czas_s"] = summary_table["Czas_s"].round(4)

summary_table.to_csv("wyniki_podsumowanie.csv", index=False)
summary_table.to_markdown("wyniki_podsumowanie.md", index=False)

def wyswietl_czytelna_tabele(df):
    df_formatted = df.copy()
    
    df_formatted["PSNR"] = df_formatted["PSNR"].round(2)
    df_formatted["SSIM"] = df_formatted["SSIM"].round(4)
    df_formatted["Czas_s"] = df_formatted["Czas_s"].round(4)
    
    df_formatted = df_formatted.rename(columns={
        "Poziom_Szumu": "Poziom Szumu",
        "Metoda": "Metoda Odszumiania",
        "PSNR": "PSNR (dB)",
        "SSIM": "SSIM",
        "Czas_s": "Czas (s)"
    })
    
    print("\n" + "=" * 65)
    print(" PODSUMOWANIE WYNIKÓW ANALIZY JAKOŚCI ODSZUMIANIA MRI")
    print("=" * 65)
    
    try:
        from tabulate import tabulate
        print(tabulate(df_formatted, headers="keys", tablefmt="fancy_grid", showindex=False))
    except ImportError:
        print(df_formatted.to_string(index=False))
        
    print("=" * 65)

wyswietl_czytelna_tabele(summary_table)

print(f"\nLiczba obrazów: {len(image_files)}")
print("Zapisano tabele do plików: wyniki_podsumowanie.csv oraz wyniki_podsumowanie.md")