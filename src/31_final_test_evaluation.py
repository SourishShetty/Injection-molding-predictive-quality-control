
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load frozen model and untouched test data
model = joblib.load("models/random_forest_model.pkl")
test = pd.read_csv("data/test.csv")

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

X_test = test[features]
y_test = test["label_viability"]

# Original G/R/Y predictions
predictions = model.predict(X_test)

print("\nFINAL HELD-OUT TEST RESULTS")
print("=" * 45)

print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(f"Macro F1: {f1_score(y_test, predictions, average='macro'):.4f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=["G", "R", "Y"],
    zero_division=0
))

print("\nConfusion matrix (G, R, Y):")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["G", "R", "Y"]
))

# Frozen combined review policy
probabilities = model.predict_proba(X_test)
classes = list(model.classes_)

p_r = probabilities[:, classes.index("R")]
p_y = probabilities[:, classes.index("Y")]

review_flags = (p_r >= 0.30) | (p_y >= 0.10)
actual_problem = y_test.isin(["R", "Y"]).to_numpy()

tp = int((review_flags & actual_problem).sum())
fp = int((review_flags & ~actual_problem).sum())
fn = int((~review_flags & actual_problem).sum())
tn = int((~review_flags & ~actual_problem).sum())

print("\nFROZEN COMBINED REVIEW POLICY")
print("=" * 45)

print("True positives:", tp)
print("False positives:", fp)
print("False negatives:", fn)
print("True negatives:", tn)

print(f"Problem recall: {tp / (tp + fn):.2%}")
print(f"Review precision: {tp / (tp + fp):.2%}" if tp + fp else
      "Review precision: Undefined")
print("Total review flags:", int(review_flags.sum()))

# Save test predictions for documentation
results = test.copy()
results["Predicted"] = predictions
results["P_G"] = probabilities[:, classes.index("G")]
results["P_R"] = p_r
results["P_Y"] = p_y
results["Review_Flag"] = review_flags

results.to_csv(
    "results/final_test_predictions.csv",
    index=False
)

print("\nTest predictions saved successfully.")
