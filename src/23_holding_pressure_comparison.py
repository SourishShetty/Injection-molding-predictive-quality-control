
import pandas as pd

validation = pd.read_csv("data/validation.csv")
errors = pd.read_csv("results/random_forest_errors.csv")

# Identify the validation rows that were misclassified
error_rows = errors["Validation_Row"].astype(int) - 1

# Select observations at holding pressure = 5 bar
low_pressure = validation[validation["Ph"] == 5].copy()

# Mark incorrect predictions
low_pressure["Model_Error"] = low_pressure.index.isin(error_rows)

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

print("\nFeature averages at 5 bar:")
print(
    low_pressure.groupby("Model_Error")[features]
    .mean()
    .round(2)
)

print("\nClass distribution by prediction correctness:")
print(
    pd.crosstab(
        low_pressure["label_viability"],
        low_pressure["Model_Error"]
    )
)

print("\nThree misclassified configurations:")
print(
    low_pressure.loc[
        low_pressure["Model_Error"],
        features + ["label_viability"]
    ]
)
