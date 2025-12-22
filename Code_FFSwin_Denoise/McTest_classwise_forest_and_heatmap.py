import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib as mpl
from matplotlib import font_manager

# =====================================
# FONT CONFIGURATION (PALATINO)
# =====================================
FONT_NAME = "Palatino Linotype"

available_fonts = [f.name for f in font_manager.fontManager.ttflist]
if FONT_NAME in available_fonts:
    mpl.rcParams["font.family"] = FONT_NAME
else:
    mpl.rcParams["font.family"] = "serif"  # safe fallback

mpl.rcParams["font.size"] = 13
mpl.rcParams["axes.titlesize"] = 15
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 13
mpl.rcParams["ytick.labelsize"] = 13
mpl.rcParams["legend.fontsize"] = 13

# =====================================
# USER CONFIGURATION
# =====================================

#EXCEL_PATH = "McNemarTest_record_PatientWise_Evaluation_Experiment_Table_sample_test.xlsx"
#OUTPUT_DIR = "mcnemar_figures_sample_test"

EXCEL_PATH = "McNemarTest_record PatientWise_Evaluation_Table_without_patient_name.xlsx"
OUTPUT_DIR = "mcnemar_figures_without_patient_name"


Path(OUTPUT_DIR).mkdir(exist_ok=True)

CLASS_LABELS = ["DryAMD", "WetAMD", "NonAMD"]

# =====================================
# LOAD DATA
# =====================================
df = pd.read_excel(EXCEL_PATH)
df.columns = [c.strip() for c in df.columns]

# =====================================
# VERIFY REQUIRED COLUMNS
# =====================================
required_cols = [
    "Patient ID",
    "Ground Truth Class",
    "Noisy Correct",
    "Clean Correct"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f" Missing required columns: {missing}")

df["Noisy Correct"] = df["Noisy Correct"].astype(bool)
df["Clean Correct"] = df["Clean Correct"].astype(bool)

# =====================================
# 1 PATIENT-LEVEL CORRECTNESS HEATMAP
# =====================================
heatmap_counts = pd.crosstab(
    df["Noisy Correct"],
    df["Clean Correct"]
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    heatmap_counts,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    annot_kws={"size": 12}
)

plt.xlabel("Clean (Denoised) Correct")
plt.ylabel("Noisy Correct")
plt.title("Patient-level Diagnostic Correctness Transition")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Mc_Nemar_patient_correctness_heatmap.png", dpi=400)
plt.close()

# =====================================
# 2 CLASS-WISE FOREST PLOT
# =====================================
forest_rows = []

for cls in CLASS_LABELS:
    sub = df[df["Ground Truth Class"] == cls]

    if len(sub) == 0:
        continue

    b = np.sum((~sub["Noisy Correct"]) & (sub["Clean Correct"]))  # Improved
    c = np.sum((sub["Noisy Correct"]) & (~sub["Clean Correct"]))  # Degraded

    if (b + c) == 0:
        continue

    forest_rows.append({
        "Class": cls,
        "Improved": b / (b + c),
        "Degraded": c / (b + c),
        "b": b,
        "c": c
    })

forest_df = pd.DataFrame(forest_rows)

# =====================================
# PLOT FOREST
# =====================================
fig, ax = plt.subplots(figsize=(8, 4))
y_pos = np.arange(len(forest_df))

ax.hlines(
    y=y_pos,
    xmin=forest_df["Degraded"],
    xmax=forest_df["Improved"],
    color="gray",
    linewidth=2
)

ax.scatter(
    forest_df["Improved"],
    y_pos,
    color="darkgreen",
    s=100,
    label="Improved"
)

ax.scatter(
    forest_df["Degraded"],
    y_pos,
    color="darkred",
    s=100,
    label="Degraded"
)

ax.axvline(0.5, linestyle="--", color="black", linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(
    [f"{r.Class} (b={r.b}, c={r.c})" for _, r in forest_df.iterrows()]
)

ax.set_xlim(0, 1)
ax.set_xlabel("Proportion of Discordant Patients")
ax.set_title("Class-wise Diagnostic Change After Denoising")

ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Mc_Nemar_classwise_forest_plot.png", dpi=400)
plt.close()

# =====================================
# DONE
# =====================================
print(" Figures generated successfully with Palatino Linotype:")
print(f" - {OUTPUT_DIR}/Mc_Nemar_patient_correctness_heatmap.png")
print(f" - {OUTPUT_DIR}/Mc_Nemar_classwise_forest_plot.png")
