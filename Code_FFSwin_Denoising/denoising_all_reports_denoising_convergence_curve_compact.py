import matplotlib.pyplot as plt
import re

# ==========================================
# 0. PUBLICATION FONT CONFIGURATION
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino Linotype']
plt.rcParams['font.size'] = 12

# ==========================================
# 1. INPUT: Your Training Log Data
# ==========================================
log_data = """
Training Denoiser on cuda...
Starting Training Loop...
Epoch 1/50, Loss: 0.00867
Epoch 2/50, Loss: 0.00798
Epoch 3/50, Loss: 0.00793
Epoch 4/50, Loss: 0.00795
Epoch 5/50, Loss: 0.00787
Epoch 6/50, Loss: 0.00785
Epoch 7/50, Loss: 0.00785
Epoch 8/50, Loss: 0.00781
Epoch 9/50, Loss: 0.00781
Epoch 10/50, Loss: 0.00779
Epoch 11/50, Loss: 0.00778
Epoch 12/50, Loss: 0.00776
Epoch 13/50, Loss: 0.00777
Epoch 14/50, Loss: 0.00775
Epoch 15/50, Loss: 0.00774
Epoch 16/50, Loss: 0.00774
Epoch 17/50, Loss: 0.00773
Epoch 18/50, Loss: 0.00773
Epoch 19/50, Loss: 0.00773
Epoch 20/50, Loss: 0.00772
Epoch 21/50, Loss: 0.00772
Epoch 22/50, Loss: 0.00772
Epoch 23/50, Loss: 0.00771
Epoch 24/50, Loss: 0.00771
Epoch 25/50, Loss: 0.00770
Epoch 26/50, Loss: 0.00770
Epoch 27/50, Loss: 0.00770
Epoch 28/50, Loss: 0.00770
Epoch 29/50, Loss: 0.00770
Epoch 30/50, Loss: 0.00770
Epoch 31/50, Loss: 0.00769
Epoch 32/50, Loss: 0.00770
Epoch 33/50, Loss: 0.00769
Epoch 34/50, Loss: 0.00769
Epoch 35/50, Loss: 0.00769
Epoch 36/50, Loss: 0.00769
Epoch 37/50, Loss: 0.00769
Epoch 38/50, Loss: 0.00769
Epoch 39/50, Loss: 0.00769
Epoch 40/50, Loss: 0.00768
Epoch 41/50, Loss: 0.00768
Epoch 42/50, Loss: 0.00768
Epoch 43/50, Loss: 0.00769
Epoch 44/50, Loss: 0.00768
Epoch 45/50, Loss: 0.00768
Epoch 46/50, Loss: 0.00768
Epoch 47/50, Loss: 0.00768
Epoch 48/50, Loss: 0.00768
Epoch 49/50, Loss: 0.00768
Epoch 50/50, Loss: 0.00768
"""

# ==========================================
# 2. PARSING THE LOGS
# ==========================================
epochs = []
losses = []

# Regex pattern to find "Epoch X/50" and "Loss: Y"
pattern = r"Epoch (\d+)/50, Loss: ([0-9.]+)"

for line in log_data.strip().split('\n'):
    match = re.search(pattern, line)
    if match:
        epochs.append(int(match.group(1)))
        losses.append(float(match.group(2)))

# ==========================================
# 3. PLOTTING FOR MDPI JOURNAL (Compact)
# ==========================================
# Figure size (6, 5) fits nicely in a quarter-page column
plt.figure(figsize=(6, 5))

# Plot line
plt.plot(epochs, losses, color='navy', linewidth=2, linestyle='-', marker='o', markersize=4, label='Training Loss')

# Axis Labels
plt.xlabel("Epochs", fontsize=14, fontweight='bold', family='Palatino Linotype')
plt.ylabel("Reconstruction Loss (MSE + L\u2081)", fontsize=14, fontweight='bold', family='Palatino Linotype')

# Title
plt.title("Restoration Model Convergence", fontsize=16, fontweight='bold', family='Palatino Linotype', pad=15)

# Ticks
plt.xticks(fontsize=12, family='Palatino Linotype')
plt.yticks(fontsize=12, family='Palatino Linotype')

# Grid and Legend
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'family': 'Palatino Linotype', 'size': 12})

# Annotations (Start and End)
# Using 'family' in dictionary to ensure font consistency in annotations
plt.annotate(f'Start: {losses[0]:.5f}', 
             xy=(epochs[0], losses[0]), 
             xytext=(epochs[0]+5, losses[0]+0.0003),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=12, family='Palatino Linotype', fontweight='bold')

plt.annotate(f'Converged: {losses[-1]:.5f}', 
             xy=(epochs[-1], losses[-1]), 
             xytext=(epochs[-1]-20, losses[-1]+0.0003),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=12, family='Palatino Linotype', fontweight='bold')

# Tight layout to prevent cutting off labels
plt.tight_layout()

# Save the plot
plt.savefig("denoising_convergence_curve_compact.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot generated successfully: 'denoising_convergence_curve_compact.png'")
print(f"Starting Loss: {losses[0]}")
print(f"Final Loss: {losses[-1]}")