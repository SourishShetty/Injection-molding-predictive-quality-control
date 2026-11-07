
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/Dataset.csv")

# Select only the six pre-production input features
features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

# Calculate Pearson correlations
correlation_matrix = df[features].corr()

print("Feature correlation matrix:")
print(correlation_matrix.round(2))

# Visualize correlations
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    fmt=".2f"
)

plt.title("Correlation Between Process Parameters")
plt.tight_layout()

plt.savefig("results/feature_correlation.png", dpi=300)
plt.show()


# Define the inputs available before manufacturing
features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

# Define the prediction target
target = "label_viability"

# Columns that must not be used as predictive inputs
excluded = [
    "batch",
    "P1_weight", "P2_weight", "P3_weight",
    "P4_weight", "P5_weight",
    "weight_avg", "energy", "cycle_time"
]

print("\nSelected model inputs:")
print(features)

print("\nPrediction target:")
print(target)

print("\nExcluded columns:")
print(excluded)

# Confirm that no excluded columns appear in our inputs
assert not set(features).intersection(excluded)

# Confirm that the target is not an input
assert target not in features

print("\nFeature selection checks passed!")
