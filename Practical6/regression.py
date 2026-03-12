#!/usr/bin/env python

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print("\nValue of 'df' :-\n", df)
print("\nValue of 'y' :-\n", y)

def evaluate(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(Xtr, ytr)
    pred = model.predict(Xte)

    print("Mean Squared Error:", mean_squared_error(yte, pred))
    print("R-squared:", r2_score(yte, pred))
    print("Intercept:", model.intercept_)
    print("Coefficient:", model.coef_)

print("\n# Single Linear Regression :-\n")
evaluate(df[["AveRooms"]], y)

print("\n# Multiple Linear Regression :-\n")
evaluate(df, y)