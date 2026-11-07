# AI-Based Predictive Quality Control for Polypropylene Injection Molding

Predicting whether a molding setup is likely to produce a viable part is useful only if the prediction holds up beyond the examples used to train the model. This project began with a straightforward question: **can we identify potentially problematic injection-molding settings before running a cycle, using only the machine parameters available at that point?**

I built a three-class classification workflow around an experimental polypropylene injection-molding dataset. It compares conventional machine-learning approaches, uses SHAP to examine what drives the selected model's predictions, and presents the results in a Streamlit interface intended for engineering review. A second evaluation, separating earlier and later experimental stages, tests how much confidence we should place in the initially strong results.

This is a **research and portfolio prototype**, not a validated production quality-control system.

## The engineering problem

Injection molding involves interacting settings: temperature, injection pressure, holding pressure, and timing can all affect whether a process is feasible and whether the resulting part is complete. Trial-and-error adjustments consume material, machine time, and engineering effort.

Rather than attempting to predict a measured part property after the cycle, I framed the task as **pre-cycle process-viability classification**. The model receives six settings and predicts one of three labels supplied by the dataset:

| Label | Dataset interpretation |
|---|---|
| **G** | Feasible process |
| **R** | Infeasible, incomplete, or erratic process |
| **Y** | Feasible process with short shots |

These labels describe the dataset's experimental outcomes; **G should not be read as a guarantee that a part is defect-free**.

## Dataset and inputs

The dataset contains **1,000 experimental observations**. The class distribution is uneven: 691 G, 260 R, and 49 Y. In particular, the short-shot class represents just **4.9%** of the observations, so a high overall accuracy can conceal poor performance on a condition that matters to an engineer.

Only six machine settings were used as model inputs:

| Feature | Meaning |
|---|---|
| `Tinj` | Injection temperature |
| `tinj` | Injection time |
| `Pinj` | Injection pressure |
| `Ph` | Holding pressure |
| `Bp` | Back pressure |
| `th` | Holding time |

I deliberately excluded measurements such as part weight, energy use, and cycle time from the predictor set. Those measurements may be informative, but using information that becomes available only during or after a cycle would undermine the purpose of making a decision **before** the cycle begins. The batch identifier was also excluded as a predictive feature; it was retained for the later experimental-stage evaluation.

The dataset is experimental rather than a representative sample of day-to-day factory production. That distinction matters when interpreting every result below.

## How I approached the problem

### 1. Explore the data and define the evaluation

I checked class balance, feature distributions, and relationships between the available process measurements. The initial evaluation used a stratified split: **700 training**, **150 validation**, and **150 held-out test** observations. The test set was reserved until model selection and the review policy had been established.

### 2. Establish baselines before choosing a model

I compared a majority-class baseline, logistic regression, Random Forest, and XGBoost variants. On the validation split, the Random Forest offered the best combination of macro F1 and recall for the two problematic classes.

| Model | Validation accuracy | Validation macro F1 |
|---|---:|---:|
| Majority-class baseline | 68.67% | 0.2714 |
| Logistic regression | 90.67% | 0.6606 |
| Random Forest | **98.00%** | **0.9492** |
| Initial XGBoost | 95.33% | 0.8690 |
| Tuned XGBoost | 95.33% | 0.8678 |

The selected model was a `RandomForestClassifier` with **200 trees** and a fixed random seed. The saved model powers the dashboard.

### 3. Evaluate the held-out test set

On the original stratified test set, the Random Forest correctly classified **148 of 150** observations.

| Metric | Held-out test result |
|---|---:|
| Accuracy | **98.67%** |
| Macro F1 | **0.9466** |
| R recall | **100% (39/39)** |
| Y recall | **85.71% (6/7)** |

These results were encouraging, but the Y recall is based on just seven test examples. A small number of observations cannot establish how reliably the model will recognize short shots in a different experimental setting.

### 4. Turn predictions into a review signal

For the dashboard, I introduced a simple engineering review rule: flag a setting when the model estimates **P(R) ≥ 0.30** or **P(Y) ≥ 0.10**. These thresholds were chosen using validation results, not tuned on the final test set.

On the original held-out test, the rule flagged all **46** observations belonging to R or Y, alongside **12** G observations. This gives **46 true review flags, 12 false review flags, and no missed R/Y cases** in that particular test set.

The output is a **recommendation to review**, not an automatic instruction to stop or modify a machine. Random Forest probabilities were not independently calibrated, and these thresholds have not been validated against real production costs or risks.

### 5. Examine the model with SHAP

SHAP helps answer a question that accuracy alone cannot: *which inputs are contributing to this prediction?* In the global validation analysis, **injection pressure (`Pinj`)** had the largest mean absolute SHAP value. I also inspected local explanations for misclassified validation examples and checked whether the feature-importance ranking remained stable between the training and validation sets.

These explanations describe the **model's learned relationships**, not proven physical causes. A high SHAP value for injection pressure does not, by itself, establish that changing injection pressure will fix a molding problem.

### 6. Test generalization across experimental stages

The most revealing result came from a separate robustness experiment. I trained a **new Random Forest** on the earlier experimental stage (observations 1–768) and evaluated it on the later stage (observations 769–1,000). The original saved dashboard model was not replaced.

| Metric | Random held-out test | Experimental-stage transfer |
|---|---:|---:|
| Accuracy | **98.67%** | **84.48%** |
| Macro F1 | **0.9466** | **~0.54** |
| R recall | **100% (39/39)** | **60% (30/50)** |
| Y recall | **85.71% (6/7)** | **0% (0/5)** |

**These are different evaluation designs, not two scores from the same trained model.** The comparison shows why a random split can provide a more optimistic picture than testing on a distinct experimental stage.

All **232** later-stage observations contained at least one individual feature value not seen in the earlier-stage training data, although their feature values remained within the earlier stage's individual minimum–maximum ranges. This suggests a change in experimental coverage and possible interpolation difficulty, but it does not prove why the model's performance declined.

The short-shot result is particularly important: the stage-transfer model recognized **none of the five Y observations**. That is not sufficient for a production-facing short-shot detector.

## Interactive dashboard

The Streamlit application provides four views:

- **Prediction:** adjust six machine settings, inspect the G/R/Y probabilities, and see the review recommendation.
- **Explainability:** inspect SHAP-based feature contributions and model behavior.
- **Model Performance:** review the original held-out test and the separate experimental-stage robustness results.
- **About Project:** understand the scope, terminology, and intended use.

For problematic predictions, the interface also explores alternative settings suggested by the model. These are **hypothetical model scenarios**, not experimentally verified process recipes. They should not be applied to production equipment without engineering review and controlled trials.

## Running the application locally

The project was developed with **Python 3.13**. From the project root, create and activate a virtual environment, then install the recorded dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the dashboard:

```powershell
python -m streamlit run app/main.py
```

The application expects the saved model at `models/random_forest_model.pkl`. The dependency list currently records the full development environment; a clean-machine installation and end-to-end reproduction have **not yet been independently verified**.

## Repository layout

```text
Manufacturing_Quality_AI/
├── app/
│   └── main.py                # Streamlit interface
├── data/                      # Experimental dataset and split files
├── models/
│   └── random_forest_model.pkl
├── results/                   # Evaluation tables and visualizations
├── src/                       # Data analysis, training, and audit scripts
├── LIMITATIONS.md             # Detailed scope and generalization limitations
├── requirements.txt
└── README.md
```

Some files and directories may be excluded from the public repository pending a review of dataset redistribution permissions and file sizes. The source dataset and its citation/license details should be added here once the original record has been verified.

## What the results mean

The initial results show that a Random Forest can learn useful relationships between the selected molding settings and the dataset's viability labels. The explainability work makes those predictions easier to inspect, and the dashboard turns the model into something an engineer can interact with.

The stage-transfer experiment sets an equally important boundary on that conclusion. Performance falls substantially when the evaluation moves to a later experimental stage, especially for the underrepresented short-shot class. For me, that is the main lesson of the project: **a convincing random-test score is a starting point for validation, not the end of it**.

Before this approach could be considered for industrial use, it would need additional data across machines, molds, material batches, and production conditions; independent stage- or run-based validation; better minority-class performance; probability calibration; and controlled verification of any suggested setting changes.

A more detailed discussion is available in [`LIMITATIONS.md`](LIMITATIONS.md).

## Dataset and Attribution

This project uses the experimental polypropylene injection-molding dataset published by Pérez-González and colleagues at the Universidad de León.

The dataset contains 1,000 injection batches, including machine settings, process viability labels, part-weight measurements, energy consumption, and cycle time. For this project, I selected six pre-cycle machine parameters to predict process viability without relying on measurements collected after manufacturing.

**Original dataset:**

Pérez-González, J., Sánchez-Calleja, I., Fernandez-Gorgojo, A., Ferrero-Guillén, R., Martínez-Gutiérrez, A., & Díez-González, J. (2026). *Injection-Molded Polypropylene Parts: Experimental Process Dataset with Part Weights, Energy Consumption and CAD Geometry*. Zenodo.

**DOI:** https://doi.org/10.5281/zenodo.20309380

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0) — https://creativecommons.org/licenses/by/4.0/

The original dataset was created by the cited researchers. The machine-learning pipeline, model evaluation, explainability analysis, and interactive dashboard presented in this repository are separate project work.