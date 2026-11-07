# Model Limitations and Industrial Applicability

**Project:** AI-Based Predictive Quality Control for Polypropylene Injection Molding

## 1. Project Scope

This project investigates whether six pre-cycle machine settings can predict polypropylene injection-molding process viability using a Random Forest classifier.

The model predicts three classes:

- **G:** Feasible process
- **R:** Infeasible process
- **Y:** Feasible process with short shots

The system includes SHAP explainability, a Streamlit dashboard, and a model-based industrial review recommendation.

The system is a research prototype and has not been validated for industrial production control.

## 2. Dataset Limitations

The dataset contains 1,000 experimental observations, including:

- 691 G observations
- 260 R observations
- 49 Y observations

The Y class represents only 4.9% of the dataset, making reliable detection and evaluation of short-shot conditions particularly challenging.

The dataset represents a limited experimental setting. Generalization to other injection-molding machines, polypropylene grades, molds, and operating environments has not been established.

## 3. Random Held-Out Test Performance

The Random Forest was evaluated using a stratified training, validation, and test split.

The final held-out test contained 150 observations.

| Metric | Result |
|---|---|
| Accuracy | 98.67% |
| Macro F1 | 0.9466 |
| R recall | 100% (39/39) |
| Y recall | 85.71% (6/7) |

These results demonstrate strong predictive performance on the selected random test split.

However, one held-out test split cannot establish consistent performance across different experimental conditions or production environments.

## 4. Experimental-Stage Generalization

An additional robustness experiment evaluated a Random Forest trained on Stage 1 observations against 232 Stage 2 observations.

The results were:

| Metric | Result |
|---|---|
| Accuracy | 84.48% |
| Macro F1 | Approximately 0.54 |
| R recall | 60% (30/50) |
| Y recall | 0% (0/5) |

This represents substantially weaker generalization than the original random held-out test.

The two evaluations use different training and testing strategies and should not be interpreted as directly equivalent performance estimates.

### Feature-value coverage

All 232 Stage 2 observations contained at least one individual feature value absent from the Stage 1 training data.

However, none of the Stage 2 feature values exceeded the minimum or maximum ranges observed in Stage 1.

This indicates a change in experimental feature-value coverage rather than extrapolation beyond the individual training feature ranges.

The feature-coverage differences may contribute to the observed performance reduction, but the analysis does not establish them as the sole cause.

## 5. Model Probability Limitations

Random Forest class probabilities are estimated from the trained ensemble.

These probabilities have not been independently calibrated.

Consequently, a model output of 90% should not automatically be interpreted as a verified 90% real-world likelihood of the corresponding process condition.

Probability calibration and independent reliability testing would be required before using these estimates for high-stakes production decisions.

## 6. Industrial Review Policy

The dashboard uses validation-selected review thresholds to identify potentially problematic process settings.

The combined review policy achieved the following results on the original held-out test:

- Problematic conditions detected: 46/46
- Missed problematic conditions: 0
- False review flags: 12
- Total review flags: 58

These results apply to the evaluated held-out test observations.

They do not establish that the review policy will achieve the same detection rate on new experimental designs or real manufacturing lines.

The review thresholds are decision-support settings, not validated industrial safety limits.

## 7. Explainability Limitations

SHAP provides local and global explanations of the Random Forest's predictions.

The analysis identified injection pressure (`Pinj`) as the most influential feature according to mean absolute SHAP values.

However, SHAP describes how features contribute to model predictions. It does not establish physical causality.

Model-based suggestions for alternative machine settings must therefore be treated as hypothetical scenarios rather than verified process corrections.

## 8. Requirements Before Industrial Deployment

Before considering industrial use, additional work would be necessary:

1. Collect representative data across machines, molds, material batches, and operating conditions.
2. Evaluate performance across independent production runs and experimental designs.
3. Improve and validate minority-class detection, particularly short-shot conditions.
4. Assess probability calibration and review thresholds using realistic manufacturing costs.
5. Validate suggested parameter adjustments through engineering review and controlled experiments.
6. Establish monitoring, retraining, and operational safeguards.

## 9. Conclusion

The project demonstrates a complete machine-learning workflow combining process-viability prediction, explainable AI, and an interactive engineering decision-support interface.

The strong random held-out test performance demonstrates the potential of the approach within the evaluated dataset.

The experimental-stage robustness assessment also reveals important limitations in generalization, particularly for the minority Y class.

The system should therefore be regarded as a research and engineering demonstration rather than a production-validated quality-control solution.