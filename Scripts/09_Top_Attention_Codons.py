# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 16:09:16 2025

@author: H.A.R
"""

import numpy as np
import pandas as pd

# Load the averaged attention weights
avg_attention_weights = np.load('E:/AE_RBFN/AE_RBFN_new/avg_attention_weights.npy')

# Define your species names (ensure this order matches the rows in your heatmap)
species_names = ['Triticum aestivum', 'Oryza sativa', 'Hordeum vulgare', 'Brachypodium'] # Adjust if needed

# Define codon names (This is the standard order of 64 codons)
codons = [
    'AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT',
    'ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT',
    'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT', 'GAA', 'GAC', 'GAG', 'GAT',
    'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT',
    'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT',
    'TTA', 'TTC', 'TTG', 'TTT'
]

# Create a DataFrame for easy analysis
attention_df = pd.DataFrame(avg_attention_weights, index=species_names, columns=codons)

# Analyze and print top 10 codons for each species
print("Top 10 Codons per Species based on Attention Weights:")
print("="*60)
for species in species_names:
    print(f"\n--- {species} ---")
    # Get the row for the species, sort values descending, and take top 10
    top_codons = attention_df.loc[species].sort_values(ascending=False).head(10)
    
    for codon, weight in top_codons.items():
        print(f"{codon}: {weight:.4f}")

# Optional: Save this table to a CSV for your paper
top_codons_per_species = {}
for species in species_names:
    top_codons_per_species[species] = attention_df.loc[species].nlargest(10).index.tolist()

# Creates a DataFrame where each column is a species and each row is a rank (1st to 10th)
results_df = pd.DataFrame(top_codons_per_species)
results_df.to_csv('E:/AE_RBFN/AE_RBFN_new/Top_Attention_Codons11.csv')
print("\n\nResults saved to 'Top_Attention_Codons.csv'")