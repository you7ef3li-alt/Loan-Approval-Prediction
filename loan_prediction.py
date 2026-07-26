
# (( Import Libraries ))

import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import logging


# Read Dataset
df = pd.read_csv("loan_approval_dataset.csv")


# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print(df.head())

print("\nDataset Information:\n")

print(df.info())

print("\nDataset Shape:")

print(df.shape)

print("\nColumns:")

print(df.columns)

print("\nMissing Values:")

print(df.isnull().sum())


# Drop loan_id column

df.drop("loan_id", axis=1, inplace=True)


# Encode categorical columns

encoder = LabelEncoder()

categorical_columns = ["education", "self_employed", "loan_status"]

for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column])



    # Features and Target

X = df.drop("loan_status", axis=1)

y = df["loan_status"]


# Split the data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



print("\nTraining Data Shape:", X_train.shape)

print("Testing Data Shape:", X_test.shape)


# ==========================
# Logistic Regression Model
# ==========================

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)


# Prediction
y_pred_log = log_model.predict(X_test)


print("\n===== Logistic Regression Results =====")

print("Accuracy :", accuracy_score(y_test, y_pred_log))

print("Precision:", precision_score(y_test, y_pred_log))

print("Recall   :", recall_score(y_test, y_pred_log))

print("F1 Score :", f1_score(y_test, y_pred_log))


print("\nClassification Report:\n")

print(classification_report(y_test, y_pred_log))


print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred_log))


# ==========================
# Decision Tree Model
# ==========================

tree_model = DecisionTreeClassifier(random_state=42)

tree_model.fit(X_train, y_train)

# Prediction

y_pred_tree = tree_model.predict(X_test)


print("\n===== Decision Tree Results =====")

print("Accuracy :", accuracy_score(y_test, y_pred_tree))

print("Precision:", precision_score(y_test, y_pred_tree))

print("Recall   :", recall_score(y_test, y_pred_tree))

print("F1 Score :", f1_score(y_test, y_pred_tree))


print("\nClassification Report:\n")

print(classification_report(y_test, y_pred_tree))


print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred_tree))

print("\n==============================")
print("Model Comparison")
print("==============================")

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, y_pred_log))

print("Decision Tree Accuracy:",
      accuracy_score(y_test, y_pred_tree))


# ==========================
# Logging
# ==========================

logging.basicConfig(
    filename="loan_predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

logging.info("Loan prediction models executed successfully.")

logging.info(
    f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.4f}"
)

logging.info(
    f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_tree):.4f}"
)

print("\nLog file created successfully.")


# =========================
# Input Validation
# ==========================

def validate_input(income, loan_amount):

    if income <= 0:
        print("Invalid Income!")
        return False

    if loan_amount <= 0:
        print("Invalid Loan Amount!")
        return False

    return True

print("\nInput Validation Example:")

validate_input(5000000, 2000000)


# ==========================
# Accuracy Comparison
# ==========================

log_accuracy = accuracy_score(y_test, y_pred_log)
tree_accuracy = accuracy_score(y_test, y_pred_tree)

models = ["Logistic Regression", "Decision Tree"]
accuracies = [log_accuracy, tree_accuracy]

plt.figure(figsize=(7,5))

plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.ylim(0,1)

for i, value in enumerate(accuracies):
    plt.text(i, value + 0.01, f"{value:.2f}", ha='center')

plt.show()


# Logistic Regression Confusion Matrix

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_log
)

plt.title("Logistic Regression Confusion Matrix")
plt.show()


# Decision Tree Confusion Matrix

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_tree
)

plt.title("Decision Tree Confusion Matrix")
plt.show()


joblib.dump(log_model, "loan_model.pkl")

print("Model saved successfully!")