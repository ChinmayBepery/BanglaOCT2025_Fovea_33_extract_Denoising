import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.measure import shannon_entropy

# ===========================
# USER PATHS
# ===========================
NOISY_ROOT = r"33_Extracted_OCT_data"
CLEAN_ROOT=r"33_Extracted_OCT_data_denoised"
#NOISY_ROOT = r"G:\My Drive\OCT_Project\Colab_Notebooks\data"
#CLEAN_ROOT = r"G:\My Drive\OCT_Project\Colab_Notebooks\local_data_clean"

CSV_OUT = "ReferenceFree_Denoising_Metrics_sample_test.csv"
FIG_OUT = "ReferenceFree_Metrics_Boxplots_sample_test.png"

CLASSES = ["DryAMD", "WetAMD", "NonAMD"]
IMG_EXT = (".bmp", ".png", ".jpg")

# ===========================
# HELPERS
# ===========================
def load_volume(folder):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(IMG_EXT):
            img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imgs.append(img.astype(np.float32) / 255.0)
    return np.stack(imgs) if len(imgs) > 0 else None

def resize_volume(vol, target_hw):
    """Resize each slice to match clean resolution"""
    resized = []
    for s in vol:
        resized.append(cv2.resize(s, target_hw, interpolation=cv2.INTER_LINEAR))
    return np.stack(resized)

def local_variance(vol):
    return np.var(vol)

def edge_strength(vol):
    return np.mean([np.mean(sobel(s)) for s in vol])

def interslice_corr(vol):
    if vol.shape[0] < 2:
        return np.nan
    corrs = []
    for i in range(vol.shape[0] - 1):
        a, b = vol[i].flatten(), vol[i+1].flatten()
        if np.std(a) > 0 and np.std(b) > 0:
            corrs.append(np.corrcoef(a, b)[0, 1])
    return np.mean(corrs) if corrs else np.nan

def entropy_mean(vol):
    return np.mean([shannon_entropy(s) for s in vol])

# ===========================
# MAIN PIPELINE
# ===========================
records = []
skipped = 0

for cls in CLASSES:
    noisy_cls = os.path.join(NOISY_ROOT, cls)
    clean_cls = os.path.join(CLEAN_ROOT, cls)

    if not os.path.exists(noisy_cls) or not os.path.exists(clean_cls):
        continue

    patients = sorted(set(os.listdir(noisy_cls)) & set(os.listdir(clean_cls)))

    for patient in patients:
        noisy_path = os.path.join(noisy_cls, patient)
        clean_path = os.path.join(clean_cls, patient)

        if not os.path.isdir(noisy_path) or not os.path.isdir(clean_path):
            continue

        noisy_vol = load_volume(noisy_path)
        clean_vol = load_volume(clean_path)

        if noisy_vol is None or clean_vol is None:
            skipped += 1
            continue

        # -----------------------------
        # RESOLUTION ALIGNMENT (CRITICAL)
        # -----------------------------
        target_h, target_w = clean_vol.shape[1], clean_vol.shape[2]
        noisy_vol = resize_volume(noisy_vol, (target_w, target_h))

        records.append({
            "Class": cls,
            "Patient_ID": patient,
            "Slices_Noisy": noisy_vol.shape[0],
            "Slices_Clean": clean_vol.shape[0],
            "ΔLNV": local_variance(noisy_vol) - local_variance(clean_vol),
            "ESPR": edge_strength(clean_vol) / (edge_strength(noisy_vol) + 1e-8),
            "ΔISC": interslice_corr(clean_vol) - interslice_corr(noisy_vol),
            "ΔEntropy": entropy_mean(noisy_vol) - entropy_mean(clean_vol)
        })

# ===========================
# SAFETY CHECK
# ===========================
df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError(" No valid patient volumes processed. Check folder structure or image formats.")

df.to_csv(CSV_OUT, index=False)

print("==============================================")
print(" Reference-free denoising metrics completed")
print(" Patients processed:", len(df))
print(" Patients skipped:", skipped)
print(" Saved to:", CSV_OUT)
print("==============================================")

# ===========================
# CLASS-WISE AGGREGATION
# ===========================
print("\nClass-wise Mean Metrics:")
print(df.groupby("Class")[["ΔLNV", "ESPR", "ΔISC", "ΔEntropy"]].mean().round(4))

# ===========================
# MDPI-READY BOXPLOTS
# ===========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype"],
    "font.size": 16
})

metrics = ["ΔLNV", "ESPR", "ΔISC", "ΔEntropy"]
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for ax, m in zip(axes, metrics):
    data = [df[df["Class"] == c][m].dropna() for c in CLASSES]
    ax.boxplot(data, labels=CLASSES, patch_artist=True)
    ax.set_title(m)
    ax.grid(True, linestyle="--", alpha=0.5)

fig.suptitle("Reference-Free Denoising Metrics (Patient-wise Paired Analysis)", fontsize=16)
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=300)
plt.close()

print(" Boxplots saved to:", FIG_OUT)
