import numpy as np 
import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report 
# Load the Iris dataset and create a binary classification problem 
iris = load_iris() 
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target']) 
binary_df = iris_df[iris_df['target'] != 2] 
X = binary_df.drop('target', axis=1) 
y = binary_df['target'] 
# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Train a logistic regression model and evaluate its performance 
logistic_model = LogisticRegression() 
logistic_model.fit(X_train, y_train) 
y_pred_logistic = logistic_model.predict(X_test) 
 
print("Logistic Regression Metrics") 
print("Accuracy: 	", 	accuracy_score(y_test, y_pred_logistic)) 	
print("Precision:", precision_score(y_test, 	y_pred_logistic)) 
print("Recall: 	", 	recall_score(y_test, y_pred_logistic)) 
print("\nClassification Report") 
print(classification_report(y_test, y_pred_logistic)) 
# Train a decision tree model and evaluate its performance 
decision_tree_model = DecisionTreeClassifier() 
decision_tree_model.fit(X_train, y_train) 
y_pred_tree = decision_tree_model.predict(X_test) 
print("\nDecision Tree Metrics") 
print("Accuracy: ", accuracy_score(y_test, y_pred_tree)) 
print("Precision:", precision_score(y_test, y_pred_tree)) 
print("Recall: ", recall_score(y_test, y_pred_tree)) 
print("\nClassification Report") 
print(classification_report(y_test, y_pred_tree)) 
