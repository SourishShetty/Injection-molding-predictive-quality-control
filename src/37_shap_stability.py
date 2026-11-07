
import pandas as pd
import numpy as np
import joblib
import shap

model = joblib.load("models/random_forest_model.pkl")

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

train = pd.read_csv("data/train.csv")
validation = pd.read_csv("data/validation.csv")

explainer = shap.TreeExplainer(model)

results = pd.DataFrame({"Feature": features})

for name, data in [
    ("Training", train),
    ("Validation", validation)
]:
    X = data[features]

    shap_values = explainer(X).values

    importance = np.abs(shap_values).mean(axis=(0, 2))

    results[name] = importance

results["Training_Rank"] = (
    results["Training"].rank(ascending=False).astype(int)
)

results["Validation_Rank"] = (
    results["Validation"].rank(ascending=False).astype(int)
)

results = results.sort_values("Validation_Rank")

print("\nSHAP feature importance stability:\n")
print(results.round(5).to_string(index=False))

results.to_csv(
    "results/shap_stability.csv",
    index=False
)
