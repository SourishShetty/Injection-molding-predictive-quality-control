
import pandas as pd

# Actual validation results
policies = {
    "Original Random Forest": {
        "FN": 2,
        "FP": 1
    },
    "Combined Review Policy": {
        "FN": 0,
        "FP": 9
    }
}

# Illustrative industrial costs
cost_missed_problem = 100
cost_false_review = 10

results = []

for name, metrics in policies.items():

    missed_cost = (
        metrics["FN"] * cost_missed_problem
    )

    review_cost = (
        metrics["FP"] * cost_false_review
    )

    total_cost = missed_cost + review_cost

    results.append({
        "Policy": name,
        "Missed_Problems": metrics["FN"],
        "False_Reviews": metrics["FP"],
        "Missed_Problem_Cost": missed_cost,
        "False_Review_Cost": review_cost,
        "Total_Cost_EUR": total_cost
    })

results_df = pd.DataFrame(results)

print("\nIndustrial cost comparison:\n")
print(results_df.to_string(index=False))

results_df.to_csv(
    "results/industrial_cost_comparison.csv",
    index=False
)
