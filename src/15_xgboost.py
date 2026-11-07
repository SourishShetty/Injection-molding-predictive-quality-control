
import pandas as pd

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# 1. Load training and validation datasets
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/validation.csv")

# 2. Select six machine-setting features
features = [
    "Tinj", "tinj", "Pinj",
    "Ph", "Bp", "th"
]

X_train = train[features]
X_val = val[features]

y_train = train["label_viability"]
y_val = val["label_viability"]

# 3. Encode target labels using training data
encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)
y_val_encoded = encoder.transform(y_val)

print("Class encoding:")
for index, label in enumerate(encoder.classes_):
    print(f"{label} -> {index}")

# 4. Create XGBoost classifier
model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# 5. Train using training data only
model.fit(X_train, y_train_encoded)

# 6. Predict validation data
predictions_encoded = model.predict(X_val)

# Convert predictions back to G, R and Y
predictions = encoder.inverse_transform(
    predictions_encoded.astype(int)
)

# 7. Evaluate performance
accuracy = accuracy_score(y_val, predictions)

macro_f1 = f1_score(
    y_val,
    predictions,
    average="macro",
    zero_division=0
)

print("\nXGBoost Results")
print("----------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_val,
    predictions,
    labels=["G", "R", "Y"],
    zero_division=0
))
