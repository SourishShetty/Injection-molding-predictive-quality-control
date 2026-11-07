
import pandas as pd

# Define our six predictive input features
features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

# Load the three saved datasets
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/validation.csv")
test = pd.read_csv("data/test.csv")

# Check missing values in each dataset
for name, df in [
    ("Training", train),
    ("Validation", val),
    ("Test", test)
]:
    missing = df[features].isnull().sum().sum()

    print(f"{name} missing input values: {missing}")

    # Stop execution if missing values are detected
    assert missing == 0, f"Missing values found in {name}!"

print("\nAll datasets passed the missing-value check!")
