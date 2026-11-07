
import pandas as pd

validation = pd.read_csv("data/validation.csv")

# Compare class counts at 5 bar and other pressures
validation["Pressure_Group"] = validation["Ph"].apply(
    lambda x: "5 bar" if x == 5 else "Other pressures"
)

counts = pd.crosstab(
    validation["Pressure_Group"],
    validation["label_viability"]
)

percentages = pd.crosstab(
    validation["Pressure_Group"],
    validation["label_viability"],
    normalize="index"
) * 100

print("\nClass counts by holding-pressure group:")
print(counts)

print("\nClass percentages within each group:")
print(percentages.round(2))
