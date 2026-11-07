
import pandas as pd

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# 1. Load training data only
train = pd.read_csv("data/train.csv")

features = [
    "Tinj", "tinj", "Pinj",
    "Ph", "Bp", "th"
]

X_train = train[features]
y_train = train["label_viability"]

# 2. Encode target labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_train)

# 3. Create XGBoost model
model = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=1
)

# 4. Define hyperparameter combinations
param_grid = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 200]
}

# 5. Define stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 6. Create grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

# 7. Run hyperparameter tuning
grid_search.fit(X_train, y_encoded)

# 8. Display best results
print("\nBest hyperparameters:")
print(grid_search.best_params_)

print("\nBest cross-validation Macro F1:")
print(round(grid_search.best_score_, 4))

# 9. Save all configuration results
results = pd.DataFrame(grid_search.cv_results_)

results.to_csv(
    "results/xgboost_tuning_results.csv",
    index=False
)

print("\nTuning results saved successfully!")
