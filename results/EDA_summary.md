
# Exploratory Data Analysis — Manufacturing Quality AI

## Dataset
- Source: Zenodo polypropylene injection-molding dataset
- Observations: 1,000 experimental batches
- Columns: 16
- Prediction target: label_viability

## Model Inputs
Tinj, tinj, Pinj, Ph, Bp, th

## Target Distribution
- G: 691 (69.1%)
- Y: 49 (4.9%)
- R: 260 (26.0%)

## Data Quality
- No missing values in the six selected inputs
- No duplicate rows or batch identifiers
- No zero or negative process settings
- Missing values exist in post-production weight and energy measurements

## Preliminary Findings
- Feasible batches generally have higher injection temperatures and pressures.
- Short-shot batches are concentrated at lower injection temperatures.
- Holding pressure, back pressure, and holding time show substantial overlap across classes.
- The six input variables have very low pairwise linear correlations.
- These observations are associations, not proof of causation.

## Data Leakage Prevention
Excluded from model inputs:
- batch
- P1_weight through P5_weight
- weight_avg
- energy
- cycle_time

## Modeling Considerations
- Use multiclass classification.
- Prioritize macro F1 and class-specific recall.
- Pay particular attention to minority class Y.
- Evaluate on unseen experimental configurations.
- Investigate potential interactions between process parameters.
- Avoid interpreting SHAP explanations as causal effects.

## Limitations
- Dataset contains only 1,000 experimental batches.
- The G label indicates feasibility, not guaranteed defect-free quality.
- Results may not generalize to other machines, materials, or molds.
- Industrial deployment requires additional validation.


## Phase 6: XGBoost Model Comparison

### Models Evaluated

1. Random Forest (200 trees)
2. Original XGBoost (depth=3, learning rate=0.05)
3. Tuned XGBoost (depth=4, learning rate=0.05)

### Validation Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest | 0.9800 | 0.9492 |
| Original XGBoost | 0.9533 | 0.8690 |
| Tuned XGBoost | 0.9533 | 0.8678 |

### Key Findings

- Random Forest achieved the highest validation macro F1.
- Hyperparameter tuning did not improve XGBoost's validation macro F1.
- Tuned XGBoost improved R recall but reduced Y recall.
- Only eight Y observations were available in validation.
- Performance on new experimental campaigns remains uncertain.

### Current Model Selection

Random Forest is the leading candidate based on
validation performance.

This is a provisional selection, not a final
production-readiness conclusion.

The held-out test dataset has not been evaluated.
