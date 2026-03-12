import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
df = pd.DataFrame(np.c_[iris.data, iris.target], columns=iris.feature_names + ['target'])

X = df.drop('target', axis=1)
y = df.target

# Scale data
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
var = pca.explained_variance_ratio_

plt.plot(np.cumsum(var), marker='o', linestyle='--')
plt.title('Explained Variance Ratio')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.show()

# Components needed for 95% variance
n = np.argmax(np.cumsum(var) >= 0.95) + 1
print(f"Number of principal components to explain 95% variance: {n}")

# Reduced dataset
X_red = PCA(n_components=n).fit_transform(X_scaled)
plt.scatter(X_red[:,0], X_red[:,1], c=y, cmap='viridis', s=50, alpha=0.5)
plt.title('Data in Reduced-dimensional Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Target')
plt.show()