
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# Load training and validation datasets
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/validation.csv")

# Select six machine-setting features
features = [
    "Tinj", "tinj", "Pinj",
    "Ph", "Bp", "th"
]

X_train = train[features]
y_train = train["label_viability"]

X_val = val[features]
y_val = val["label_viability"]

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)


os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/random_forest_model.pkl")

print("Random Forest model saved successfully!")

# Predict validation dataset
y_pred = model.predict(X_val)

# Evaluate model
accuracy = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(
    y_val, y_pred,
    average="macro",
    zero_division=0
)

print("Random Forest Results")
print("---------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_val, y_pred,
    zero_division=0
))
