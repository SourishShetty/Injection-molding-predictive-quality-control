
import pandas as pd
import joblib

model = joblib.load("models/random_forest_model.pkl")
validation = pd.read_csv("data/validation.csv")

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

X = validation[features]
y = validation["label_viability"]

probabilities = model.predict_proba(X)

y_index = list(model.classes_).index("Y")
p_y = probabilities[:, y_index]

actual_y = y == "Y"

results = []

for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:

    flags = p_y >= threshold

    tp = int((flags & actual_y).sum())
    fp = int((flags & ~actual_y).sum())
    fn = int((~flags & actual_y).sum())
    tn = int((~flags & ~actual_y).sum())

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

print("\nY threshold comparison:\n")
print(results_df.to_string(index=False))

results_df.to_csv(
    "results/y_threshold_comparison.csv",
    index=False
)
