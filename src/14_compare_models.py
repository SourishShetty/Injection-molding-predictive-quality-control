
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score
)

# Load datasets
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/validation.csv")

features = [
    "Tinj", "tinj", "Pinj",
    "Ph", "Bp", "th"
]

X_train = train[features]
y_train = train["label_viability"]

X_val = val[features]
y_val = val["label_viability"]

# Define our three baseline models
models = {
    "Majority Baseline": DummyClassifier(
        strategy="most_frequent"
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}

results = []

# Train and evaluate each model
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(
            y_val, predictions
        ),
        "Macro F1": f1_score(
            y_val, predictions,
            average="macro",
            zero_division=0
        ),
        "Balanced Accuracy": balanced_accuracy_score(
            y_val, predictions
        ),
        "R Recall": recall_score(
            y_val, predictions,
            labels=["R"],
            average="macro",
            zero_division=0
        ),
        "Y Recall": recall_score(
            y_val, predictions,
            labels=["Y"],
            average="macro",
            zero_division=0
        )
    })

# Create comparison table
comparison = pd.DataFrame(results)

print("\nModel Comparison:")
print(comparison.round(4).to_string(index=False))

# Save results
comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")
