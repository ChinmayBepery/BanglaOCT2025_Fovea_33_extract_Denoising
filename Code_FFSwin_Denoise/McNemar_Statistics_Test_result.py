import pandas as pd
from scipy.stats import binomtest, chi2


# ------------------------------------------------------------------
# USER PATH
# ------------------------------------------------------------------
#EXCEL_PATH = r"McNemarTest_record_PatientWise_Evaluation_Experiment_Table_sample_test.xlsx"   # update if needed
EXCEL_PATH = r"McNemarTest_record_PatientWise_Evaluation_Experiment_Table.xlsx"   # update if needed

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
df = pd.read_excel(EXCEL_PATH)

# Ensure required columns exist
required_cols = ["Noisy Correct", "Clean Correct"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' missing from your file!")

# ------------------------------------------------------------------
# BUILD CONTINGENCY TABLE
# ------------------------------------------------------------------
# a = correct on both noisy and clean
a = len(df[(df["Noisy Correct"] == 1) & (df["Clean Correct"] == 1)])

# b = noisy incorrect → clean correct (improved)
b = len(df[(df["Noisy Correct"] == 0) & (df["Clean Correct"] == 1)])

# c = noisy correct → clean incorrect (degraded)
c = len(df[(df["Noisy Correct"] == 1) & (df["Clean Correct"] == 0)])

# d = incorrect on both
d = len(df[(df["Noisy Correct"] == 0) & (df["Clean Correct"] == 0)])

# ------------------------------------------------------------------
# DISPLAY TABLE
# ------------------------------------------------------------------
print("\n====================================")
print("  McNemar Contingency Table (2x2)")
print("====================================")
print(f"                   Clean Correct")
print(f"                  Yes       No")
print(f"Noisy  : Yes     {a:3d}      {c:3d}")
print(f"Correct: No      {b:3d}      {d:3d}")
print("====================================\n")

# ------------------------------------------------------------------
# McNemar’s Chi-square Test
# ------------------------------------------------------------------

if (b + c) == 0:
    print("b + c = 0 → No discordance. McNemar test cannot be performed.")
else:

    # Continuity-corrected chi-square
    chi_square_cc = ((abs(b - c) - 1)**2) / (b + c)
    p_value_cc = 1 - chi2.cdf(chi_square_cc, df=1)

    # Uncorrected chi-square
    chi_square_no_cc = (b - c)**2 / (b + c)
    p_value_no_cc = 1 - chi2.cdf(chi_square_no_cc, df=1)

    # Exact binomial test (recommended)
    exact_test = binomtest(k=min(b, c), n=b + c, p=0.5)
    p_exact = exact_test.pvalue

    print("McNemar Test Results")
    print("-----------------------------")
    print(f"b (improved)          = {b}")
    print(f"c (degraded)          = {c}")
    print(f"b + c (discordant)    = {b + c}\n")

    print("Continuity-corrected χ²:")
    print(f"  χ² = {chi_square_cc:.4f}")
    print(f"  p-value = {p_value_cc:.6f}\n")

    print("Uncorrected χ²:")
    print(f"  χ² = {chi_square_no_cc:.4f}")
    print(f"  p-value = {p_value_no_cc:.6f}\n")

    print("Exact Binomial Test (recommended):")
    print(f"  p-value = {p_exact:.6f}\n")

    # Interpretation
    if p_exact < 0.05:
        print("🟢 Interpretation: The improvement from noisy → clean data is statistically significant (p < 0.05).")
    else:
        print("🔵 Interpretation: No statistically significant difference detected (p ≥ 0.05).")
