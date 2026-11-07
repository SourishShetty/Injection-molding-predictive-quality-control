
import pandas as pd
import joblib

model = joblib.load("models/random_forest_model.pkl")
validation = pd.read_csv("data/validation.csv")

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

X = validation[features]
y = validation["label_viability"]

probabilities = model.predict_proba(X)
r_index = list(model.classes_).index("R")
p_r = probabilities[:, r_index]

actual_r = y == "R"

results = []

for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:

    flags = p_r >= threshold

    tp = int((flags & actual_r).sum())
    fp = int((flags & ~actual_r).sum())
    fn = int((~flags & actual_r).sum())
    tn = int((~flags & ~actual_r).sum())

    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if tp + fp else 0

    results.append({
        "Threshold": threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Recall": round(recall, 4),
        "Precision": round(precision, 4),
        "Total_Flags": int(flags.sum())
    })

results_df = pd.DataFrame(results)

print("\nR threshold comparison:\n")
print(results_df.to_string(index=False))

results_df.to_csv(
    "results/r_threshold_comparison.csv",
    index=False
)
