
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# Load datasets
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/validation.csv")

# Select input features
features = [
    "Tinj", "tinj", "Pinj",
    "Ph", "Bp", "th"
]

X_train = train[features]
y_train = train["label_viability"]

X_val = val[features]
y_val = val["label_viability"]

# Build leakage-safe pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Predict validation dataset
y_pred = model.predict(X_val)

# Evaluate model
accuracy = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)

print("Logistic Regression Results")
print("---------------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_val,
    y_pred,
    zero_division=0
))
