
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load wine dataset
df = pd.read_csv("wine.csv")
print("\nWine CSV :-\n")
print(df)

# Separate features from target (Wine column is the target/class label)
target = df['Wine']  # Save target variable
features = df.drop('Wine', axis=1)  # Only scale features, not target

# MinMax Scaling (only on features)
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
features_minmax = minmax_scaler.fit_transform(features)
print("\nWine Features after MinMax scaling (first 5 rows):-\n")
print(pd.DataFrame(features_minmax, columns=features.columns).head())

# Standard Scaling (only on features)
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
features_standard = standard_scaler.fit_transform(features)
print("\nWine Features after Standard scaling (first 5 rows):-\n")
print(pd.DataFrame(features_standard, columns=features.columns).head())


## Label Encoding (not dummification)

iris = pd.read_csv("iris.csv")
print("\niris.head() :-\n")
print(iris.head())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
iris['code'] = encoder.fit_transform(iris.species)
print("\n\nPerforming label encoding...")
print("\niris.head() :-\n")
print(iris.head())

print("\niris.tail() :-\n")
print(iris.tail())

# To perform actual dummification (one-hot encoding), use:
iris_dummies = pd.get_dummies(iris, columns=['species'])
print("\niris with dummies:\n")
print(iris_dummies.head())
