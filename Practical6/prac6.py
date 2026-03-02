#!/usr/bin/env python
# same as prac5_simple-linear-regression.ipynb


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
print("\nValue of 'df' :-\n", df)


y = housing.target
print("\nValue of 'y' :-\n", y)


def evaluate(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R-squared:", r2_score(y_test, y_pred))
    print("Intercept:", model.intercept_)
    print("Coefficient:", model.coef_)


x = df[["AveRooms"]]
print("\n# Single Linear Regression :-\n")
evaluate(x, y)


x = df
print("\n# Multiple Linear Regression :-\n")
evaluate(x, y)
