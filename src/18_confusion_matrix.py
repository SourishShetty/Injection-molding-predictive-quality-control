
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load training and validation datasets
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/validation.csv")

features = [
    "Tinj", "tinj", "Pinj",
    "Ph", "Bp", "th"
]

X_train = train[features]
y_train = train["label_viability"]

X_val = val[features]
y_val = val["label_viability"]

# Train our leading model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict validation data
predictions = model.predict(X_val)

# Calculate confusion matrix
labels = ["G", "R", "Y"]

cm = confusion_matrix(
    y_val,
    predictions,
    labels=labels
)

print("Confusion Matrix:")
print(cm)

# Visualize the matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot(cmap="Blues", values_format="d")

plt.title("Random Forest - Validation Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/random_forest_confusion_matrix.png",
    dpi=300
)

plt.show()
