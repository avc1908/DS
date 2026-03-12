import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Load wine dataset
df = pd.read_csv("wine.csv")
print("\nWine CSV :-\n", df)

# Separate target and features
y = df['Wine']
X = df.drop('Wine', axis=1)

# MinMax Scaling
print("\nWine Features after MinMax scaling (first 5 rows):-\n")
print(pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns).head())

# Standard Scaling
print("\nWine Features after Standard scaling (first 5 rows):-\n")
print(pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns).head())


# Label Encoding
iris = pd.read_csv("iris.csv")
print("\niris.head() :-\n", iris.head())

iris['code'] = LabelEncoder().fit_transform(iris['species'])
print("\n\nPerforming label encoding...")
print("\niris.head() :-\n", iris.head())
print("\niris.tail() :-\n", iris.tail())

# One Hot Encoding (dummification)
print("\niris with dummies:\n")
print(pd.get_dummies(iris, columns=['species']).head())