# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 10:23:45 2025

@author: H.A.R
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

# Load data
file_path = "E:/novel_MLP2/4_Species_gene.csv"
df = pd.read_csv(file_path)

# 1. Separate Metadata and Features
# First three columns are metadata: Sequence_ID, Species, Label
metadata = df.iloc[:, :3] 
# All columns from the 4th onward are codon counts (features)
features = df.iloc[:, 3:] 

# Extract specific metadata for plotting
sequence_ids = metadata['Sequence_ID']
species_names = metadata['Species']  # This is crucial for color-coding
numeric_labels = metadata['Label']

# 2. Preprocess the Feature Data
# Normalize by row (gene) to get relative frequencies
# This ensures each gene's codon usage sums to 1, making them comparable
features_normalized = features.div(features.sum(axis=1), axis=0)

# Standardize the data (mean=0, variance=1) for PCA and UMAP
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_normalized)

# 3. Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=principal_components, 
                      columns=['PC1', 'PC2'])
pca_df['Species'] = species_names.values # Add the species names for coloring

# 4. Perform UMAP
umap_reducer = umap.UMAP(n_neighbors=15, 
                         min_dist=0.1, 
                         n_components=2, 
                         random_state=42, 
                         n_jobs=-1) # n_jobs=-1 uses all CPU cores
umap_results = umap_reducer.fit_transform(features_scaled)

# Create a DataFrame for the UMAP results
umap_df = pd.DataFrame(data=umap_results, 
                       columns=['UMAP1', 'UMAP2'])
umap_df['Species'] = species_names.values # Add the species names for coloring

# 5. Create the Combined Figure (PCA + UMAP)
plt.figure(figsize=(16, 7)) # Width, Height in inches

# --- Plot 1: PCA ---
plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st panel
# Create the scatter plot
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Species', 
                palette='tab10', alpha=0.8, s=50, edgecolor='w', linewidth=0.2)
# Add titles and labels with variance explained
plt.title(f"PCA of Gene Codon Usage", fontsize=16, fontweight='bold')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)", fontsize=14)
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)", fontsize=14)
plt.legend(title='Species', title_fontsize=12, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

# --- Plot 2: UMAP ---
plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd panel
sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Species', 
                palette='tab10', alpha=0.8, s=50, edgecolor='w', linewidth=0.2)
plt.title("UMAP Visualization of Gene Codon Usage", fontsize=16, fontweight='bold')
plt.xlabel("UMAP Dimension 1", fontsize=14)
plt.ylabel("UMAP Dimension 2", fontsize=14)
plt.legend(title='Species', title_fontsize=12, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

# Final layout adjustment to prevent overlapping
plt.tight_layout(pad=3.0)

# Save the figure as a high-resolution PNG for your paper
plt.savefig('E:/novel_MLP2/Figure_2_PCA_UMAP_Comparison.png', 
            dpi=300, 
            bbox_inches='tight')

# Display the figure
plt.show()

# (Optional) Print out the total variance explained by the first two PCs
total_variance = pca.explained_variance_ratio_.sum() * 100
print(f"Total variance captured by PC1 and PC2: {total_variance:.2f}%")