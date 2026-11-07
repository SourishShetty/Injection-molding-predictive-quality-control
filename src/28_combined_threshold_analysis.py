
import pandas as pd
import joblib

model = joblib.load("models/random_forest_model.pkl")
validation = pd.read_csv("data/validation.csv")

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

X = validation[features]
y = validation["label_viability"]

probabilities = model.predict_proba(X)
classes = list(model.classes_)

p_r = probabilities[:, classes.index("R")]
p_y = probabilities[:, classes.index("Y")]

# Provisional thresholds
r_threshold = 0.30
y_threshold = 0.10

# Flag if either risk threshold is reached
review_flags = (
    (p_r >= r_threshold) |
    (p_y >= y_threshold)
)

# Actual problematic conditions: R or Y
actual_problem = y.isin(["R", "Y"])

tp = int((review_flags & actual_problem).sum())
fp = int((review_flags & ~actual_problem).sum())
fn = int((~review_flags & actual_problem).sum())
tn = int((~review_flags & ~actual_problem).sum())

recall = tp / (tp + fn)
precision = tp / (tp + fp) if tp + fp else 0

print("\nCombined review policy:")
print(f"R threshold: {r_threshold:.0%}")
print(f"Y threshold: {y_threshold:.0%}")

print("\nConfusion matrix counts:")
print("True positives:", tp)
print("False positives:", fp)
print("False negatives:", fn)
print("True negatives:", tn)

print(f"\nProblem detection recall: {recall:.2%}")
print(f"Review precision: {precision:.2%}")
print("Total configurations flagged:", int(review_flags.sum()))

print("\nReview flags by actual class:")
print(pd.crosstab(
    y,
    pd.Series(review_flags, name="Flagged_for_Review")
))
