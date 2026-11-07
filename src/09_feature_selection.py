
import pandas as pd

# Load training dataset
train = pd.read_csv("data/train.csv")

# Original machine-setting features
features = [
    "Tinj",
    "tinj",
    "Pinj",
    "Ph",
    "Bp",
    "th"
]

# Select inputs and target
X_train = train[features]
y_train = train["label_viability"]

print("Selected input features:")
for feature in features:
    print("-", feature)

print("\nNumber of input features:", X_train.shape[1])
print("Number of training samples:", X_train.shape[0])

# Verify that no post-production measurements are included
excluded = [
    "P1_weight", "P2_weight", "P3_weight",
    "P4_weight", "P5_weight", "weight_avg",
    "energy", "cycle_time", "batch"
]

assert not set(features).intersection(excluded)

print("\nFeature selection verified successfully!")
