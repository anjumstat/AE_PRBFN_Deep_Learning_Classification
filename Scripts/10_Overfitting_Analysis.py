# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 19:55:00 2025

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Overfitting Analysis for AE-RBFN Model - 10-Fold Cross Validation
@author: H.A.R
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
output_dir = 'E:/AE_RBFN/RBFN_Standard_new1/'
model_name = 'Standard RBFN'

# Load the full history data
all_histories = np.load(os.path.join(output_dir, 'all_histories.npy'), allow_pickle=True)

# Create a comprehensive overfitting analysis plot
plt.figure(figsize=(15, 10))

# ===== Plot 1: All 10 Folds Training vs Validation Accuracy =====
plt.subplot(2, 2, 1)
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for fold_idx, history in enumerate(all_histories):
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Plot training accuracy (lighter, transparent)
    plt.plot(epochs, history['accuracy'], color=colors[fold_idx], 
             alpha=0.4, linewidth=1, label=f'Fold {fold_idx+1} Train' if fold_idx == 0 else "")
    
    # Plot validation accuracy (darker)
    plt.plot(epochs, history['val_accuracy'], color=colors[fold_idx], 
             alpha=0.8, linewidth=2, label=f'Fold {fold_idx+1} Val' if fold_idx == 0 else "")

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('a) Training vs Validation Accuracy (All 10 Folds)')
plt.grid(True, alpha=0.3)
plt.legend()

# ===== Plot 2: Average Training vs Validation Accuracy =====
plt.subplot(2, 2, 2)

# Load averaged data
avg_train_acc = np.load(os.path.join(output_dir, 'avg_train_acc.npy'))
avg_val_acc = np.load(os.path.join(output_dir, 'avg_val_acc.npy'))
epochs_avg = range(1, len(avg_train_acc) + 1)

plt.plot(epochs_avg, avg_train_acc, 'b-', linewidth=3, label='Average Training Accuracy')
plt.plot(epochs_avg, avg_val_acc, 'r-', linewidth=3, label='Average Validation Accuracy')

# Calculate and display final gap
final_gap = abs(avg_train_acc[-1] - avg_val_acc[-1])
plt.axhline(y=avg_train_acc[-1], color='blue', linestyle=':', alpha=0.7)
plt.axhline(y=avg_val_acc[-1], color='red', linestyle=':', alpha=0.7)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(f'b) Average Accuracy Across Folds\nFinal Gap: {final_gap:.4f}')
plt.grid(True, alpha=0.3)
plt.legend()

# ===== Plot 3: Accuracy Gap (Train - Val) for Each Fold =====
plt.subplot(2, 2, 3)

max_epochs = max(len(h['accuracy']) for h in all_histories)
all_gaps = []

for fold_idx, history in enumerate(all_histories):
    train_acc = np.array(history['accuracy'])
    val_acc = np.array(history['val_accuracy'])
    
    # Pad to same length for consistent plotting
    if len(train_acc) < max_epochs:
        train_acc = np.pad(train_acc, (0, max_epochs - len(train_acc)), mode='edge')
        val_acc = np.pad(val_acc, (0, max_epochs - len(val_acc)), mode='edge')
    
    gap = train_acc - val_acc
    all_gaps.append(gap)
    epochs = range(1, len(gap) + 1)
    
    # Every fold gets a label
    plt.plot(epochs, gap, color=colors[fold_idx], alpha=0.7, 
             label=f'Fold {fold_idx+1}')

plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.xlabel('Epochs')
plt.ylabel('Accuracy Gap (Train - Val)')
plt.title('c) Overfitting Gap for Each Fold')
plt.grid(True, alpha=0.3)
plt.legend(ncol=2, fontsize=8, loc='lower right')  # compact legend in 2 columns

# ===== Plot 4: Statistical Summary of Overfitting Gap =====
plt.subplot(2, 2, 4)

# Calculate final gaps for each fold
final_gaps = []
for history in all_histories:
    train_final = history['accuracy'][-1]
    val_final = history['val_accuracy'][-1]
    final_gaps.append(train_final - val_final)

# Create box plot
plt.boxplot(final_gaps, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'),
            whiskerprops=dict(color='blue'),
            capprops=dict(color='blue'))

plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.xticks([1], ['Final Epoch Gap'])
plt.ylabel('Accuracy Gap (Train - Val)')
plt.title('d) Statistical Summary of Final Overfitting Gap')
plt.grid(True, alpha=0.3)

# Add text with statistics
mean_gap = np.mean(final_gaps)
std_gap = np.std(final_gaps)
plt.text(0.7, max(final_gaps) * 0.9, f'Mean: {mean_gap:.4f}\nStd: {std_gap:.4f}', 
         bbox=dict(facecolor='white', alpha=0.8))

# ===== Final Touches =====
plt.tight_layout()
plt.suptitle(f'Overfitting Analysis: {model_name} (10-Fold Cross Validation)', 
             fontsize=16, fontweight='bold', y=1.02)

# Save the figure
plt.savefig(os.path.join(output_dir, 'comprehensive_overfitting_analysis.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

# ===== Print summary statistics =====
print("=" * 60)
print(f"OVERFITTING ANALYSIS SUMMARY - {model_name}")
print("=" * 60)

print(f"\nFinal Epoch Statistics (Across 10 Folds):")
print(f"Average Training Accuracy: {np.mean([h['accuracy'][-1] for h in all_histories]):.4f}")
print(f"Average Validation Accuracy: {np.mean([h['val_accuracy'][-1] for h in all_histories]):.4f}")
print(f"Average Final Gap (Train - Val): {mean_gap:.4f} ± {std_gap:.4f}")
print(f"Maximum Final Gap: {max(final_gaps):.4f}")
print(f"Minimum Final Gap: {min(final_gaps):.4f}")

print(f"\nEarly Stopping Analysis:")
epochs_trained = [len(h['accuracy']) for h in all_histories]
print(f"Average Epochs Trained: {np.mean(epochs_trained):.1f} ± {np.std(epochs_trained):.1f}")
print(f"Early Stopping Range: {min(epochs_trained)} - {max(epochs_trained)} epochs")

# ===== Overfitting warnings =====
if mean_gap > 0.05:  # threshold for significant overfitting
    print(f"\n⚠️  WARNING: Potential overfitting detected (mean gap > 0.05)")
elif mean_gap > 0.02:
    print(f"\n⚠️  NOTE: Minor overfitting detected (mean gap > 0.02)")
else:
    print(f"\n✅ GOOD: Minimal overfitting detected (mean gap ≤ 0.02)")
