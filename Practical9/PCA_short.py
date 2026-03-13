import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
p = PCA().fit(X)

plt.plot(np.cumsum(p.explained_variance_ratio_)); plt.show()

X = PCA(2).fit_transform(X)
plt.scatter(X[:,0], X[:,1], c=y); plt.show()