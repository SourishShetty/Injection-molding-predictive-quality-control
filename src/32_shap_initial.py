
import pandas as pd
import joblib
import shap

# Load our saved Random Forest model
model = joblib.load("models/random_forest_model.pkl")

# Load validation data
validation = pd.read_csv("data/validation.csv")

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

# Select five observations for our first analysis
X = validation[features].head(5)

# Create a Tree SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer(X)

print("\nSHAP calculation successful!")
print("Input shape:", X.shape)
print("SHAP values shape:", shap_values.values.shape)
print("Model classes:", model.classes_)

print("\nBaseline values:")
print(shap_values.base_values[0])

print("\nSHAP values for the first observation:")
print(shap_values.values[0])
