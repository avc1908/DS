import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv("wholesale.csv")

cat = ['Channel', 'Region']
cont = ['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']

data[cont].describe()

# One-hot encoding
for c in cat:
    data = pd.concat([data, pd.get_dummies(data[c], prefix=c)], axis=1).drop(c, axis=1)

# Scaling
data_scaled = MinMaxScaler().fit_transform(data)

# Elbow method
dist = []
K = range(1,15)

for k in K:
    dist.append(KMeans(n_clusters=k).fit(data_scaled).inertia_)

plt.plot(K, dist, 'bx-')
plt.xlabel('k')
plt.ylabel('sum_of_squared_distances')
plt.title('Elbow Method for optimal k')
plt.show()