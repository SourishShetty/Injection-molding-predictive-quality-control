
import pandas as pd

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# Load training and validation data
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/validation.csv")

features = [
    "Tinj", "tinj", "Pinj",
    "Ph", "Bp", "th"
]

X_train = train[features]
X_val = val[features]

y_train = train["label_viability"]
y_val = val["label_viability"]

# Encode target labels
encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)

# Best hyperparameters from GridSearchCV
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# Train on training data only
model.fit(X_train, y_train_encoded)

# Predict validation observations
pred_encoded = model.predict(X_val)

predictions = encoder.inverse_transform(
    pred_encoded.astype(int)
)

# Calculate evaluation metrics
accuracy = accuracy_score(y_val, predictions)

macro_f1 = f1_score(
    y_val,
    predictions,
    average="macro",
    zero_division=0
)

print("Tuned XGBoost Validation Results")
print("--------------------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_val,
    predictions,
    labels=["G", "R", "Y"],
    zero_division=0
))
