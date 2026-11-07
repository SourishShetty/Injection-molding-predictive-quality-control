
import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load trained model and validation data
model = joblib.load("models/random_forest_model.pkl")
validation = pd.read_csv("data/validation.csv")

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]
X = validation[features]

os.makedirs("results/shap_errors", exist_ok=True)

# Calculate SHAP values for all validation observations
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

classes = list(model.classes_)

# Actual validation rows (1-based numbering)
error_rows = [18, 65, 104]

for row_number in error_rows:

    index = row_number - 1

    actual = validation.iloc[index]["label_viability"]
    predicted = model.predict(X.iloc[[index]])[0]

    # Explain the actual class when missed;
    # otherwise explain the incorrectly predicted class
    class_to_explain = (
        actual if actual != "G" else predicted
    )

    class_index = classes.index(class_to_explain)

    # Generate waterfall plot
    shap.plots.waterfall(
        shap_values[index, :, class_index],
        show=False
    )

    plt.title(
        f"Validation Row {row_number}: "
        f"Actual {actual}, Predicted {predicted}"
    )

    plt.tight_layout()

    filename = (
        f"results/shap_errors/"
        f"row_{row_number}_{class_to_explain}.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Display actual and predicted probabilities
    probabilities = model.predict_proba(
        X.iloc[[index]]
    )[0]

    print(f"\nValidation row {row_number}")
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}")

    for label, probability in zip(classes, probabilities):
        print(f"P({label}): {probability:.2%}")

    print(f"SHAP plot saved: {filename}")

print("\nAll three SHAP explanations completed!")
