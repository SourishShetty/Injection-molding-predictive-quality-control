
import matplotlib.pyplot as plt
import pandas as pd

# Load the injection-molding dataset
df = pd.read_csv("data/Dataset.csv")

# Display dataset dimensions
print("Dataset shape:", df.shape)

# Display column names and data types
print("\nColumn information:")
df.info()

# Display basic statistics
print("\nStatistical summary:")
print(df.describe())


# Step 6: Count missing values in each column
print("\nMissing values per column:")
print(df.isnull().sum())

# Step 7: Count completely duplicated rows
print("\nDuplicate rows:", df.duplicated().sum())

# Step 8: Check duplicate batch identifiers
print("Duplicate batch IDs:", df["batch"].duplicated().sum())


# Step 9: Define the six machine settings
features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

# Step 10: Check for zero or negative measurements
print("\nZero or negative values:")
print((df[features] <= 0).sum())

# Step 11: Check the observed operating ranges
print("\nMinimum and maximum values:")
print(df[features].agg(["min", "max"]))


# Step 12: Count observations in each viability class
class_counts = df["label_viability"].value_counts()

print("\nViability class distribution:")
print(class_counts)

# Step 13: Calculate percentage of each class
class_percentages = (
    df["label_viability"].value_counts(normalize=True) * 100
)

print("\nClass percentages:")
print(class_percentages.round(2))


# Step 14: Visualize class distribution

counts = df["label_viability"].value_counts()
counts = counts.reindex(["G", "Y", "R"])

plt.figure(figsize=(7, 5))
plt.bar(counts.index, counts.values)

plt.xlabel("Viability class")
plt.ylabel("Number of batches")
plt.title("Injection-Molding Viability Distribution")

plt.tight_layout()
plt.savefig("results/class_distribution.png", dpi=300)
plt.show()
