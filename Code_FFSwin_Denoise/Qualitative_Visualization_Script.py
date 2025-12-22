import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# USER PATHS
# ==========================
NOISY_ROOT = r"33_Extracted_OCT_data"
CLEAN_ROOT=r"33_Extracted_OCT_data_denoised"
OUT_DIR = "Qualitative_Visualizations_sample_test"
#NOISY_ROOT = r"G:\My Drive\OCT_Project\Colab_Notebooks\data"
#CLEAN_ROOT = r"G:\My Drive\OCT_Project\Colab_Notebooks\local_data_clean"
#OUT_DIR = "Qualitative_Visualizations"

CLASSES = ["DryAMD", "WetAMD", "NonAMD"]
IMG_EXT = (".bmp", ".png", ".jpg")

os.makedirs(OUT_DIR, exist_ok=True)

# ==========================
# PLOT SETTINGS (MDPI)
# ==========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype"],
    "font.size": 14
})

# ==========================
# HELPERS
# ==========================
def load_slice(folder, idx):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)])
    if idx >= len(files):
        return None
    img = cv2.imread(os.path.join(folder, files[idx]), cv2.IMREAD_GRAYSCALE)
    return img

def center_crop(img, crop_size=128):
    h, w = img.shape
    ch, cw = crop_size, crop_size
    y1 = h//2 - ch//2
    x1 = w//2 - cw//2
    return img[y1:y1+ch, x1:x1+cw]

# ==========================
# MAIN VISUALIZATION LOOP
# ==========================
for cls in CLASSES:
    noisy_cls = os.path.join(NOISY_ROOT, cls)
    clean_cls = os.path.join(CLEAN_ROOT, cls)

    patients = sorted(set(os.listdir(noisy_cls)) & set(os.listdir(clean_cls)))
    if not patients:
        continue

    # Pick first patient (or change index)
    patient = patients[0]

    noisy_path = os.path.join(noisy_cls, patient)
    clean_path = os.path.join(clean_cls, patient)

    # Pick middle slice
    slice_idx = len(os.listdir(clean_path)) // 2

    noisy = load_slice(noisy_path, slice_idx)
    clean = load_slice(clean_path, slice_idx)

    if noisy is None or clean is None:
        continue

    # Resize noisy to match clean
    noisy = cv2.resize(noisy, (clean.shape[1], clean.shape[0]))

    # Difference map
    diff = np.abs(noisy.astype(np.float32) - clean.astype(np.float32))

    # ==========================
    # FIGURE 1: SIDE-BY-SIDE
    # ==========================
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(noisy, cmap="gray")
    axes[0].set_title("Noisy OCT")
    axes[0].axis("off")

    axes[1].imshow(clean, cmap="gray")
    axes[1].set_title("Denoised OCT")
    axes[1].axis("off")

    axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("Absolute Difference")
    axes[2].axis("off")

    fig.suptitle(f"{cls} — Patient: {patient}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{cls}_SideBySide.png"), dpi=350)
    plt.close()

    # ==========================
    # FIGURE 2: ZOOMED ROI
    # ==========================
    noisy_roi = center_crop(noisy)
    clean_roi = center_crop(clean)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(noisy_roi, cmap="gray")
    axes[0].set_title("Noisy (ROI)")
    axes[0].axis("off")

    axes[1].imshow(clean_roi, cmap="gray")
    axes[1].set_title("Denoised (ROI)")
    axes[1].axis("off")

    fig.suptitle(f"{cls} — Zoomed Retinal Region", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{cls}_ZoomedROI.png"), dpi=300)
    plt.close()

print("Qualitative visualizations saved to:", OUT_DIR)
