
import pandas as pd

# Load training dataset
train = pd.read_csv("data/train.csv")

# Select the six input features
features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

# Display data types
print("Input feature data types:")
print(train[features].dtypes)

# Verify all input features are numerical
assert all(
    pd.api.types.is_numeric_dtype(train[col])
    for col in features
)

print("\nAll six input features are numerical!")
