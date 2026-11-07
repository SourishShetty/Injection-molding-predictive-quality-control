
import pandas as pd
import numpy as np
import joblib
import shap

# Load saved model and validation data
model = joblib.load("models/random_forest_model.pkl")
validation = pd.read_csv("data/validation.csv")

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]
X = validation[features]

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# Shape: observations x features x classes
values = shap_values.values

# Mean absolute SHAP contribution across
# all observations and all three classes
importance = np.abs(values).mean(axis=(0, 2))

results = pd.DataFrame({
    "Feature": features,
    "Mean_Absolute_SHAP": importance
}).sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

print("\nGlobal SHAP feature importance:\n")
print(results.to_string(index=False))

results.to_csv(
    "results/shap_global_importance.csv",
    index=False
)
