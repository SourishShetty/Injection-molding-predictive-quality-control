
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/Dataset.csv")

# Define the six pre-production features
features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

# Separate input features and target labels
X = df[features]
y = df["label_viability"]

# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split: temporary data into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Display the resulting dataset sizes
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

# Display class distributions
for name, labels in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Test", y_test)
]:
    print(f"\n{name} class distribution:")
    print(labels.value_counts())


# Save each dataset with its target label
train_df = X_train.copy()
train_df["label_viability"] = y_train

val_df = X_val.copy()
val_df["label_viability"] = y_val

test_df = X_test.copy()
test_df["label_viability"] = y_test

# Save datasets as CSV files
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/validation.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("\nAll three datasets saved successfully!")
