
import pandas as pd
import joblib

# Load model and validation data
model = joblib.load("models/random_forest_model.pkl")
validation = pd.read_csv("data/validation.csv")

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

X = validation[features]
y = validation["label_viability"]

# Obtain probability estimates
probabilities = model.predict_proba(X)

# Find the probability column for class R
r_index = list(model.classes_).index("R")
p_r = probabilities[:, r_index]

# Apply illustrative review threshold
threshold = 0.30
review_flags = p_r >= threshold

# Compare review flags with actual R cases
actual_r = y == "R"

tp = int((review_flags & actual_r).sum())
fp = int((review_flags & ~actual_r).sum())
fn = int((~review_flags & actual_r).sum())
tn = int((~review_flags & ~actual_r).sum())

print("\nR review threshold:", threshold)
print("True positives:", tp)
print("False positives:", fp)
print("False negatives:", fn)
print("True negatives:", tn)

recall = tp / (tp + fn)
precision = tp / (tp + fp) if tp + fp else 0

print(f"\nR recall: {recall:.2%}")
print(f"R review precision: {precision:.2%}")
print(f"Total configurations flagged: {review_flags.sum()}")
