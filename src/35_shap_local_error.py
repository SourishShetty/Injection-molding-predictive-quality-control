
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and validation data
model = joblib.load("models/random_forest_model.pkl")
validation = pd.read_csv("data/validation.csv")

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

# Validation row 18 corresponds to index 17
X = validation.loc[[17], features]

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# Select class R
r_index = list(model.classes_).index("R")

# Create local waterfall plot
shap.plots.waterfall(
    shap_values[0, :, r_index],
    show=False
)

plt.title("SHAP Explanation — Missed R (Validation Row 18)")
plt.tight_layout()

plt.savefig(
    "results/shap_local_error_R.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Local SHAP plot saved successfully!")
