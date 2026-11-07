
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load trained model and validation data
model = joblib.load("models/random_forest_model.pkl")
validation = pd.read_csv("data/validation.csv")

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]
X = validation[features]

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# Select SHAP values for class R
r_index = list(model.classes_).index("R")

# Generate SHAP beeswarm plot
shap.plots.beeswarm(
    shap_values[:, :, r_index],
    show=False
)

plt.title("SHAP Summary — Infeasible Process (R)")
plt.tight_layout()

plt.savefig(
    "results/shap_summary_R.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("SHAP summary plot saved successfully!")
