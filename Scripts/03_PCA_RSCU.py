import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = "E:/AE_RBFN/4_Species_gene1.csv"
df = pd.read_csv(file_path)

# Drop non-numeric columns for PCA
X = df.drop(columns=['Sequence_ID', 'Species', 'Label'])
y = df['Species']  # For coloring in the plot

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

# Create a DataFrame for plotting
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Species'] = y

# Plotting
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Species', palette='Set2', alpha=0.7, edgecolor='w', s=60)

# Title and axis labels
plt.title("PCA of Relative Synonymous Codon Usage (RSCU) Values", fontsize=14)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
plt.legend(title='Species', fontsize=10, title_fontsize=11)
plt.grid(True)
plt.tight_layout()
plt.show()
