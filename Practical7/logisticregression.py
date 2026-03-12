import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Load dataset and create binary classification
iris = load_iris()
df = pd.DataFrame(np.c_[iris.data, iris.target], columns=iris.feature_names + ['target'])
df = df[df.target != 2]

X = df.drop('target', axis=1)
y = df.target

# Train-test split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate(model, name):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    print(f"\n{name} Metrics")
    print("Accuracy:", accuracy_score(yte, pred))
    print("Precision:", precision_score(yte, pred))
    print("Recall:", recall_score(yte, pred))
    print("\nClassification Report")
    print(classification_report(yte, pred))

# Logistic Regression
evaluate(LogisticRegression(), "Logistic Regression")

# Decision Tree
evaluate(DecisionTreeClassifier(), "Decision Tree")