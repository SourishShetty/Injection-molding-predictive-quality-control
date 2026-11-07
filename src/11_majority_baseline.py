
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# Load training and validation datasets
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/validation.csv")

# Define six input features
features = [
    "Tinj", "tinj", "Pinj",
    "Ph", "Bp", "th"
]

X_train = train[features]
y_train = train["label_viability"]

X_val = val[features]
y_val = val["label_viability"]

# Create a classifier that always predicts
# the most common training class
model = DummyClassifier(strategy="most_frequent")

# Fit using training data only
model.fit(X_train, y_train)

# Predict validation labels
y_pred = model.predict(X_val)

# Evaluate predictions
accuracy = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(
    y_val, y_pred,
    average="macro",
    zero_division=0
)

print("Majority-Class Baseline Results")
print("-------------------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_val, y_pred,
    zero_division=0
))
