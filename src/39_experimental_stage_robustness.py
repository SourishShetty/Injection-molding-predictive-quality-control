import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. Load original dataset
# --------------------------------------------------

df = pd.read_csv("data/Dataset.csv")

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]
classes = ["G", "R", "Y"]

# --------------------------------------------------
# 2. Separate experimental stages
# --------------------------------------------------

stage1 = df[df["batch"].between(1, 768)].copy()
stage2 = df[df["batch"].between(769, 1000)].copy()

X_train = stage1[features]
y_train = stage1["label_viability"]

X_test = stage2[features]
y_test = stage2["label_viability"]

print("\nEXPERIMENTAL-STAGE ROBUSTNESS AUDIT")
print("=" * 50)

print(f"Stage 1 training observations: {len(stage1)}")
print(f"Stage 2 evaluation observations: {len(stage2)}")

print("\nStage 1 class distribution:")
print(y_train.value_counts().reindex(classes, fill_value=0))

print("\nStage 2 class distribution:")
print(y_test.value_counts().reindex(classes, fill_value=0))

# --------------------------------------------------
# 3. Train a separate Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 4. Evaluate on Stage 2
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test,
    predictions,
    labels=classes,
    average="macro",
    zero_division=0
)

print("\nSTAGE 1 -> STAGE 2 RESULTS")
print("=" * 50)

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=classes,
        zero_division=0
    )
)

print("\nConfusion matrix (G, R, Y):")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=classes
    )
)

# --------------------------------------------------
# 5. Check feature ranges
# --------------------------------------------------

print("\nSTAGE 2 OBSERVATIONS OUTSIDE STAGE 1 RANGES")
print("=" * 50)

for feature in features:
    minimum = X_train[feature].min()
    maximum = X_train[feature].max()

    outside = (
        (X_test[feature] < minimum) |
        (X_test[feature] > maximum)
    ).sum()

    print(
        f"{feature}: {outside}/{len(X_test)} "
        f"outside [{minimum}, {maximum}]"
    )

print("\nAudit completed.")
print("Original saved Random Forest model was not modified.")
