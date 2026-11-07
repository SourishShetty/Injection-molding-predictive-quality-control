
import pandas as pd
import joblib

# Load validation data and saved model
validation = pd.read_csv("data/validation.csv")

model = joblib.load("models/random_forest_model.pkl")

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

X = validation[features]

# Generate class probabilities
probabilities = model.predict_proba(X)

# Create probability table
probability_df = pd.DataFrame(
    probabilities,
    columns=[f"P_{label}" for label in model.classes_]
)

probability_df["Actual"] = validation["label_viability"]
probability_df["Predicted"] = model.predict(X)

# Identify incorrect predictions
errors = probability_df[
    probability_df["Actual"] != probability_df["Predicted"]
].copy()

errors["Validation_Row"] = errors.index + 1

print("\nProbabilities for misclassified observations:\n")

print(
    errors[
        ["Validation_Row", "Actual", "Predicted",
         "P_G", "P_R", "P_Y"]
    ].round(4).to_string(index=False)
)

errors.to_csv(
    "results/random_forest_error_probabilities.csv",
    index=False
)
