
import pandas as pd

# Load validation data and previously saved model errors
validation = pd.read_csv("data/validation.csv")
errors = pd.read_csv("results/random_forest_errors.csv")

# Count validation observations for each holding pressure
total_counts = validation.groupby("Ph").size()

# Count model errors for each holding pressure
error_counts = errors.groupby("Ph").size()

# Combine the counts
summary = pd.DataFrame({
    "Total_Observations": total_counts,
    "Model_Errors": error_counts
}).fillna(0)

summary["Model_Errors"] = summary["Model_Errors"].astype(int)

# Calculate error percentage
summary["Error_Rate_Percent"] = (
    summary["Model_Errors"]
    / summary["Total_Observations"]
    * 100
)

print("\nRandom Forest error rate by holding pressure:\n")
print(summary.to_string())

# Save results
summary.to_csv("results/holding_pressure_error_analysis.csv")
