
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Load datasets
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/validation.csv")

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

# 2. Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(
    train[features],
    train["label_viability"]
)

# 3. Predict validation data
predictions = model.predict(val[features])

# 4. Identify incorrectly classified observations
errors = val.copy()
errors["Predicted"] = predictions

errors = errors[
    errors["label_viability"] != errors["Predicted"]
].copy()

# 5. Categorize the mistakes


def classify_error(row):
    actual = row["label_viability"]
    predicted = row["Predicted"]

    if actual == "R" and predicted in ["G", "Y"]:
        return "Infeasible process missed"

    if actual in ["G", "Y"] and predicted == "R":
        return "Feasible process rejected"

    if actual == "Y" and predicted == "G":
        return "Short-shot condition missed"

    if actual == "G" and predicted == "Y":
        return "Short-shot condition falsely flagged"

    return "Other classification error"


errors["Error_Type"] = errors.apply(
    classify_error,
    axis=1
)

# 6. Display results
# Identify each observation by its validation row number
errors.insert(0, "Validation_Row", errors.index + 1)

columns = [
    "Validation_Row",
    "label_viability",
    "Predicted",
    "Error_Type"
] + features

print("\nMisclassified Validation Observations")
print("-------------------------------------")
print(errors[columns].to_string(index=False))

print(f"\nTotal errors: {len(errors)}")

# 7. Save results
errors[columns].to_csv(
    "results/random_forest_errors.csv",
    index=False
)

print("\nError analysis saved successfully!")
