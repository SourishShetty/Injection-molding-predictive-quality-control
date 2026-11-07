
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

# 1. Load training and validation datasets
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/validation.csv")

# 2. Select the six machine-setting features
features = [
    "Tinj", "tinj", "Pinj",
    "Ph", "Bp", "th"
]

X_train = train[features]
y_train = train["label_viability"]

X_val = val[features]
y_val = val["label_viability"]

# 3. Train our leading Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 4. Predict validation observations
predictions = model.predict(X_val)

# 5. Calculate metrics for each class
labels = ["G", "R", "Y"]

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_val,
        predictions,
        labels=labels,
        zero_division=0
    )
)

# 6. Create a results table
metrics = pd.DataFrame({
    "Class": labels,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "Support": support
})

print("Per-Class Evaluation Metrics")
print("----------------------------")
print(metrics.round(4).to_string(index=False))

# 7. Calculate overall metrics
accuracy = accuracy_score(y_val, predictions)

balanced_accuracy = balanced_accuracy_score(
    y_val, predictions
)

macro_f1 = f1_score(
    y_val,
    predictions,
    average="macro"
)

print("\nOverall Evaluation Metrics")
print("--------------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

# 8. Save results
metrics.to_csv(
    "results/random_forest_class_metrics.csv",
    index=False
)

print("\nClass metrics saved successfully!")
