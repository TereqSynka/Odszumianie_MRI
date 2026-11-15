
import os
import re
import time
from itertools import combinations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.stats import wilcoxon
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SEED = 42
FOLDER = "normalne"
VALID_EXTENSIONS = (".png", ".jpg")
NOISE_LEVELS = [0.05, 0.15, 0.30]

GAUSS_KSIZE = (5, 5)
GAUSS_SIGMA = 1.5
MEDIAN_KSIZE = 5
NLM_H = 40
NLM_TEMPLATE = 7
NLM_SEARCH = 21

TIME_REPEATS = 5
EXAMPLE_IDX = 0
ERRMAP_NOISE = 0.15

BASELINE = "No filtering"
M_GAUSS = "Gaussian filter"
M_MEDIAN = "Median filter"
M_NLM = "NLM"
METHODS = [M_GAUSS, M_MEDIAN, M_NLM]
ALL_METHODS = [BASELINE] + METHODS
MARKERS = {M_GAUSS: "o", M_MEDIAN: "s", M_NLM: "^"}

BASE_NOISY_FOLDER = "noisy"
BASE_FILTERED_FOLDER = "filtered"
OUT_DIR = "results"

rng = np.random.default_rng(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


def slug(text):
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")


def comma(x, _=None):
    return f"{x:g}".replace(".", ",")


def build_filters():
    return {
        BASELINE: lambda img: img.copy(),
        M_GAUSS: lambda img: cv2.GaussianBlur(img, GAUSS_KSIZE, sigmaX=GAUSS_SIGMA),
        M_MEDIAN: lambda img: cv2.medianBlur(img, ksize=MEDIAN_KSIZE),
        M_NLM: lambda img: cv2.fastNlMeansDenoising(
            img, h=NLM_H, templateWindowSize=NLM_TEMPLATE,
            searchWindowSize=NLM_SEARCH),
    }


def measure(fn, img, repeats=TIME_REPEATS):
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(img)
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times)) * 1000.0


def metrics(reference, img):
    return (
        peak_signal_noise_ratio(reference, img, data_range=255),
        structural_similarity(reference, img, data_range=255),
    )


if not os.path.isdir(FOLDER):
    raise FileNotFoundError(f"Missing image folder: '{FOLDER}'")

image_files = sorted(
    os.path.join(FOLDER, f)
    for f in os.listdir(FOLDER)
    if f.lower().endswith(VALID_EXTENSIONS)
)
if not image_files:
    raise FileNotFoundError(
        f"No files matching {VALID_EXTENSIONS} found in folder '{FOLDER}'")

_warm = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
for _fn in build_filters().values():
    _fn(_warm)

all_results = []
example = {}
n_used = 0

for idx, file_path in enumerate(image_files):
    original = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        print(f"Skipped (failed to load): {file_path}")
        continue
    n_used += 1
    name = os.path.splitext(os.path.basename(file_path))[0]
    original_float = original.astype(np.float64) / 255.0

    if idx == EXAMPLE_IDX:
        example["orig"] = original

    for sigma in NOISE_LEVELS:
        noise = rng.normal(0.0, sigma, size=original_float.shape)
        noisy = np.round(np.clip(original_float + noise, 0, 1) * 255).astype(np.uint8)

        noisy_dir = os.path.join(BASE_NOISY_FOLDER, f"noise_{sigma}")
        os.makedirs(noisy_dir, exist_ok=True)
        cv2.imwrite(os.path.join(noisy_dir, f"{name}.png"), noisy)

        results_here = {}
        for method, fn in build_filters().items():
            filtered, t_ms = measure(fn, noisy)
            results_here[method] = filtered

            out_dir = os.path.join(BASE_FILTERED_FOLDER, f"noise_{sigma}", slug(method))
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"{name}.png"), filtered)

            val_psnr, val_ssim = metrics(original, filtered)
            all_results.append({
                "Image_ID": idx + 1,
                "Noise_Level": sigma,
                "Method": method,
                "PSNR": val_psnr,
                "SSIM": val_ssim,
                "Time_ms": np.nan if method == BASELINE else t_ms,
            })

        if idx == EXAMPLE_IDX:
            example[sigma] = results_here

df = pd.DataFrame(all_results)
df["Method"] = pd.Categorical(df["Method"], categories=ALL_METHODS, ordered=True)

summary = (
    df.groupby(["Noise_Level", "Method"], observed=True)[["PSNR", "SSIM", "Time_ms"]]
    .agg(["mean", "std"])
)
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
summary = summary.reset_index()

summary_out = summary.copy()
for col in summary_out.columns:
    if col.startswith("PSNR"):
        summary_out[col] = summary_out[col].round(2)
    elif col.startswith("SSIM"):
        summary_out[col] = summary_out[col].round(4)
    elif col.startswith("Time"):
        summary_out[col] = summary_out[col].round(3)

summary_out.to_csv(os.path.join(OUT_DIR, "results_summary.csv"),
                    index=False, sep=";", decimal=",", encoding="utf-8-sig")
df.to_csv(os.path.join(OUT_DIR, "results_raw.csv"),
          index=False, sep=";", decimal=",", encoding="utf-8-sig")

print("\n" + "=" * 90)
print(" RESULTS SUMMARY (mean and std. dev.)")
print("=" * 90)
print(summary_out.to_string(index=False))
print("=" * 90)
print(f"Number of used images: {n_used}")

if n_used < 6:
    print("\nNote: with fewer than 6 images, the Wilcoxon test cannot yield p < 0.05.")

stat_rows = []
for metric in ["PSNR", "SSIM", "Time_ms"]:
    methods = ALL_METHODS if metric != "Time_ms" else METHODS
    pairs = list(combinations(methods, 2))
    for sigma in NOISE_LEVELS:
        piv = (
            df[df["Noise_Level"] == sigma]
            .pivot_table(index="Image_ID", columns="Method", values=metric, observed=True)
        )
        for a, b in pairs:
            x, y = piv[a].to_numpy(), piv[b].to_numpy()
            if np.allclose(x, y):
                p = 1.0
            else:
                try:
                    p = float(wilcoxon(x, y).pvalue)
                except ValueError:
                    p = np.nan
            stat_rows.append({
                "Metric": metric, "Noise_Level": sigma, "Method_A": a, "Method_B": b,
                "Mean_A": x.mean(), "Mean_B": y.mean(),
                "p": p, "p_Bonferroni": min(1.0, p * len(pairs)) if not np.isnan(p) else np.nan,
            })

stats = pd.DataFrame(stat_rows)
stats["Significant_005"] = stats["p_Bonferroni"] < 0.05
stats.to_csv(os.path.join(OUT_DIR, "wilcoxon_tests.csv"),
              index=False, sep=";", decimal=",", encoding="utf-8-sig")
print("\nSaved statistical tests: results/wilcoxon_tests.csv")
print(stats[stats["Metric"] != "Time_ms"]
      .query("Significant_005 == False")
      .loc[:, ["Metric", "Noise_Level", "Method_A", "Method_B", "p_Bonferroni"]]
      .to_string(index=False))
print("(above: pairs where NO significant difference was detected)")


def rysuj(ax, col, ylabel, title, log=False, with_baseline=True):
    if with_baseline:
        b = (df[df["Method"] == BASELINE].groupby("Noise_Level")[col].mean())
        ax.plot(b.index, b.values, linestyle="--", color="gray", label=BASELINE)
    for method in METHODS:
        sub = (df[df["Method"] == method]
               .groupby("Noise_Level")[col].agg(["mean", "std"]))
        ax.errorbar(sub.index, sub["mean"], yerr=sub["std"],
                    marker=MARKERS[method], capsize=3, label=method)
    ax.set_xticks(NOISE_LEVELS)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}".replace(".", ",")))
    ax.set_xlabel("Noise level (σ)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(comma))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
rysuj(axes[0], "PSNR", "PSNR [dB]", "a) PSNR")
rysuj(axes[1], "SSIM", "SSIM", "b) SSIM")
rysuj(axes[2], "Time_ms", "Time [ms]", "c) Execution time", log=True, with_baseline=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_psnr_ssim_time.png"), dpi=300)
plt.close(fig)

if "orig" in example:
    orig = example["orig"]
    cols = ["Original", BASELINE] + METHODS
    titles = ["Original", "Noisy"] + METHODS
    fig, axes = plt.subplots(len(NOISE_LEVELS), len(cols),
                             figsize=(3 * len(cols), 3.1 * len(NOISE_LEVELS)))
    for r, sigma in enumerate(NOISE_LEVELS):
        for c, col in enumerate(cols):
            ax = axes[r, c]
            img = orig if col == "Original" else example[sigma][col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"σ = {sigma:.2f}".replace(".", ","), fontsize=11)
            if col != "Original":
                p, s = metrics(orig, img)
                ax.set_xlabel(f"PSNR {p:.2f} dB | SSIM {s:.3f}".replace(".", ","), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "figure_image_comparison.png"), dpi=300)
    plt.close(fig)

    sigma = ERRMAP_NOISE if ERRMAP_NOISE in example else NOISE_LEVELS[1]
    cols = [BASELINE] + METHODS
    fig, axes = plt.subplots(1, len(cols), figsize=(3 * len(cols), 3.4))
    for ax, col in zip(axes, cols):
        diff = np.abs(orig.astype(np.int16) - example[sigma][col].astype(np.int16))
        im = ax.imshow(diff, cmap="inferno", vmin=0, vmax=100)
        ax.set_title(col if col != BASELINE else "Noisy", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="|difference| (0–255)")
    fig.suptitle(f"Difference maps relative to reference image, σ = {sigma:.2f}".replace(".", ","))
    plt.savefig(os.path.join(OUT_DIR, "figure_difference_maps.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

print(f"\nKoniec, zapisano w folderze:'{OUT_DIR}/' folder.")
