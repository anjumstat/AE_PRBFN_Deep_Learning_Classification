# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 16:20:32 2025

@author: H.A.R
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the averaged attention weights
avg_attention_weights = np.load('E:/AE_RBFN/AE_RBFN_new/avg_attention_weights.npy')

# Define your species names (ensure this order matches the rows in your data)
species_names = ['Triticum aestivum', 'Oryza sativa', 'Hordeum vulgare', 'Brachypodium']

# Define codon names (standard order of 64 codons)
codons = [
    'AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT',
    'ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT',
    'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT', 'GAA', 'GAC', 'GAG', 'GAT',
    'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT',
    'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT',
    'TTA', 'TTC', 'TTG', 'TTT'
]

# Create a DataFrame for better labeling
attention_df = pd.DataFrame(avg_attention_weights, 
                           index=species_names, 
                           columns=codons)

# Create the heatmap
plt.figure(figsize=(16, 10))
heatmap = sns.heatmap(
    attention_df,
    cmap="YlOrRd",  # Yellow-Orange-Red colormap
    xticklabels=codons,
    yticklabels=species_names,
    cbar_kws={'label': 'Attention Weight'}
)

# Customize the plot
plt.title("Species-Specific Codon Attention Weights", fontsize=18, pad=20)
plt.xlabel("Codon", fontsize=14)
plt.ylabel("Species", fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=90, fontsize=14)
plt.yticks(rotation=0, fontsize=14)

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Save the figure
plt.savefig('E:/AE_RBFN/AE_RBFN_new/Species_Attention_Heatmap1.png', 
            dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

print("Heatmap saved as 'Species_Attention_Heatmap.png'")