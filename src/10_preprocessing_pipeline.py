
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load training dataset
train = pd.read_csv("data/train.csv")

# Define input features
features = [
    "Tinj", "tinj", "Pinj",
    "Ph", "Bp", "th"
]

X_train = train[features]
y_train = train["label_viability"]

# Create a preprocessing + model pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

print("Preprocessing pipeline created successfully!")
print(pipeline)
