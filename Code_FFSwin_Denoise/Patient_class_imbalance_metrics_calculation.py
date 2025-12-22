import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score
)
"""
# ------------------------------------------------------------------
# USER PATHS #sample data paths
# ------------------------------------------------------------------
EXCEL_PATH = "McNemarTest_record_PatientWise_Evaluation_Experiment_Table_sample_test.xlsx"
GT_COL = "Ground Truth Class"
NOISY_COL = "Noisy Prediction"
CLEAN_COL = "Clean Prediction"

CLASS_ORDER = ["DryAMD", "WetAMD", "NonAMD"]  # IMPORTANT
# ------------------------------------------------------------------
# USER CONFIGURATION PATHS---Experimental data paths
# ------------------------------------------------------------------
"""
#Sample data paths
#EXCEL_PATH = "McNemarTest_record_PatientWise_Evaluation_Experiment_Table_sample_test.xlsx"
#Experimental data paths
EXCEL_PATH = "McNemarTest_record PatientWise_Evaluation_Table_without_patient_name.xlsx"

GT_COL = "Ground Truth Class"
NOISY_COL = "Noisy Prediction"
CLEAN_COL = "Clean Prediction"



CLASS_ORDER = ["DryAMD", "WetAMD", "NonAMD"]  # IMPORTANT

# ================================
# LOAD DATA
# ================================
df = pd.read_excel(EXCEL_PATH)

y_true = df[GT_COL].astype(str)
y_noisy = df[NOISY_COL].astype(str)
y_clean = df[CLEAN_COL].astype(str)

# ================================
# METRIC FUNCTION
# ================================
def compute_metrics(y_true, y_pred, label):
    print(f"\n{'='*70}")
    print(f"METRICS FOR: {label}")
    print(f"{'='*70}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)

    print("\nConfusion Matrix:")
    print(cm_df)

    # Classification Report (macro, weighted, per-class)
    report = classification_report(
        y_true, y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()

    # Balanced Accuracy
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # MCC
    mcc = matthews_corrcoef(y_true, y_pred)

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    print("\nClass-wise Metrics:")
    print(report_df.loc[CLASS_ORDER][["precision", "recall", "f1-score", "support"]])

    print("\nMacro / Weighted Metrics:")
    print(report_df.loc[["macro avg", "weighted avg"]][["precision", "recall", "f1-score"]])

    print("\nAdditional Metrics:")
    print(f"Balanced Accuracy : {bal_acc:.4f}")
    print(f"MCC               : {mcc:.4f}")
    print(f"Cohen's Kappa     : {kappa:.4f}")
    print(f"Overall Accuracy  : {report_df.loc['accuracy', 'precision']:.4f}")

    return {
        "confusion_matrix": cm_df,
        "report": report_df,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "kappa": kappa
    }

# ================================
# RUN METRICS
# ================================
metrics_noisy = compute_metrics(y_true, y_noisy, "NOISY DATA")
metrics_clean = compute_metrics(y_true, y_clean, "DENOISED DATA")

# ================================
# DELTA (IMPROVEMENT)
# ================================
print("\n" + "="*70)
print("IMPROVEMENT AFTER DENOISING (CLEAN − NOISY)")
print("="*70)

delta_bal_acc = metrics_clean["balanced_accuracy"] - metrics_noisy["balanced_accuracy"]
delta_mcc = metrics_clean["mcc"] - metrics_noisy["mcc"]
delta_kappa = metrics_clean["kappa"] - metrics_noisy["kappa"]

print(f"Δ Balanced Accuracy : {delta_bal_acc:+.4f}")
print(f"Δ MCC               : {delta_mcc:+.4f}")
print(f"Δ Cohen's Kappa     : {delta_kappa:+.4f}")
