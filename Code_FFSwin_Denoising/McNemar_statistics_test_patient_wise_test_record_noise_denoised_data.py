import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from model_architecture import FFSwinClassifier


# ------------------------------------------------------------------
# USER PATHS #sample data paths
# ------------------------------------------------------------------
 
NOISY_ROOT = r"33_Extracted_OCT_data"
CLEAN_ROOT = r"33_Extracted_OCT_data_denoised"
MODEL_PATH = r"models\classifier_best_100_used_here_as_final.pth"
OUTPUT_CSV = "McNemarTest_record_PatientWise_Evaluation_Experiment_Table_sample_test.csv"

# ------------------------------------------------------------------
# USER PATHS---Experimental data paths
# ------------------------------------------------------------------

#NOISY_ROOT = r"G:\My Drive\OCT_Project\Colab_Notebooks\data"
#CLEAN_ROOT = r"G:\My Drive\OCT_Project\Colab_Notebooks\local_data_clean"
#MODEL_PATH = r"models\classifier_best_100_used_here_as_final.pth"
#OUTPUT_CSV = "McNemarTest_record_PatientWise_Evaluation_Experiment_Table.csv"

# Resize images
IMG_SIZE = (256, 256)

# Expected class folders
CLASSES = ["DryAMD", "WetAMD", "NonAMD"]
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}


# ------------------------------------------------------------------
# Load classifier model
# ------------------------------------------------------------------
def load_classifier(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFSwinClassifier(num_classes=3).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model, device


# ------------------------------------------------------------------
# Load patient volume (BMP → Tensor)
# ------------------------------------------------------------------
def load_volume(path):
    """Load volume of BMP images and convert to shape (1,1,D,H,W)"""
    files = sorted(glob.glob(os.path.join(path, "*.bmp")))
    if len(files) == 0:
        return None, None

    vol = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        vol.append(img)

    if len(vol) == 0:
        return None, None

    vol = np.array(vol) / 255.0   # normalize
    original_depth = vol.shape[0]

    # Pad odd depth for Swin
    if original_depth % 2 != 0:
        vol = np.concatenate([vol, vol[-1:]], axis=0)

    tensor = torch.FloatTensor(vol).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    return tensor, original_depth


# ------------------------------------------------------------------
# Run classifier on volume → return predicted class ID
# ------------------------------------------------------------------
def classify_volume(model, device, vol_tensor):
    with torch.no_grad():
        vol_tensor = vol_tensor.to(device)
        logits = model(vol_tensor)
        pred = torch.argmax(logits, dim=1).item()
    return pred


# ------------------------------------------------------------------
# Patient evaluation (noisy vs clean)
# ------------------------------------------------------------------
def evaluate_all_patients():
    model, device = load_classifier(MODEL_PATH)

    results = []

    print("\n Evaluating all patients...\n")

    for cls in CLASSES:
        noisy_cls_path = os.path.join(NOISY_ROOT, cls)
        clean_cls_path = os.path.join(CLEAN_ROOT, cls)

        if not os.path.exists(noisy_cls_path):
            print(f"⚠ Class folder missing in noisy dataset: {cls}")
            continue

        for patient in sorted(os.listdir(noisy_cls_path)):
            noisy_patient_path = os.path.join(noisy_cls_path, patient)
            clean_patient_path = os.path.join(clean_cls_path, patient)

            if not os.path.isdir(noisy_patient_path):
                continue

            # Load volumes
            noisy_vol, d_noisy = load_volume(noisy_patient_path)
            if noisy_vol is None:
                continue

            clean_vol, d_clean = load_volume(clean_patient_path)
            if clean_vol is None:
                continue

            # Run classifier
            noisy_pred = classify_volume(model, device, noisy_vol)
            clean_pred = classify_volume(model, device, clean_vol)

            gt = CLASS_TO_ID[cls]

            # correctness flags
            noisy_correct = int(noisy_pred == gt)
            clean_correct = int(clean_pred == gt)

            changed = ""
            if noisy_correct == 0 and clean_correct == 1:
                changed = "0 → 1 (Improved)"
            elif noisy_correct == 1 and clean_correct == 0:
                changed = "1 → 0 (Degraded)"
            else:
                changed = "No Change"

            # save row
            results.append({
                "Patient ID": patient,
                "Ground Truth Class": cls,
                "GT_ID": gt,
                "Noisy Prediction": CLASSES[noisy_pred],
                "Noisy Pred ID": noisy_pred,
                "Noisy Correct": noisy_correct,
                "Clean Prediction": CLASSES[clean_pred],
                "Clean Pred ID": clean_pred,
                "Clean Correct": clean_correct,
                "Changed Correctness?": changed,
                "Slices_Noise": d_noisy,
                "Slices_Clean": d_clean,
                "Noisy_Path": noisy_patient_path,
                "Clean_Path": clean_patient_path
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n Patient-wise evaluation saved to:\n{OUTPUT_CSV}\n")
    print(df.head())
    return df


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    evaluate_all_patients()
