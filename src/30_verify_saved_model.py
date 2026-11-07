
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, f1_score

model = joblib.load("models/random_forest_model.pkl")
validation = pd.read_csv("data/validation.csv")

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

X = validation[features]
y = validation["label_viability"]

predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)
macro_f1 = f1_score(y, predictions, average="macro")

print("\nSaved Random Forest verification:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nExpected validation results:")
print("Accuracy: 0.9800")
print("Macro F1: 0.9492")

assert abs(accuracy - 0.9800) < 0.0001
assert abs(macro_f1 - 0.9492) < 0.0001

print("\nVerification PASSED.")
