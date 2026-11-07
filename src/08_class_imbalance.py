
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Load training dataset only
train = pd.read_csv("data/train.csv")

# Extract target labels
y_train = train["label_viability"]

# Identify classes
classes = np.unique(y_train)

# Calculate balanced class weights
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

# Display results
print("Training class counts:")
print(y_train.value_counts())

print("\nCalculated class weights:")
for label, weight in zip(classes, weights):
    print(f"{label}: {weight:.3f}")
