import pandas as pd

df = pd.read_csv("data/Dataset.csv")

stage1 = df[df["batch"].between(1, 768)]
stage2 = df[df["batch"].between(769, 1000)]

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

print("\nSTAGE 2 FEATURE-VALUE COVERAGE")
print("=" * 55)

unseen_any = pd.Series(False, index=stage2.index)

for feature in features:
    known_values = set(stage1[feature].unique())
    unseen = ~stage2[feature].isin(known_values)
    unseen_any |= unseen

    print(
        f"{feature}: {unseen.sum()}/{len(stage2)} "
        f"Stage 2 observations use a value absent from Stage 1"
    )

print("\nStage 2 observations with at least one unseen value:")
print(f"{unseen_any.sum()}/{len(stage2)}")

print("\nClass distribution of observations with unseen values:")
print(
    stage2.loc[unseen_any, "label_viability"]
    .value_counts()
    .reindex(["G", "R", "Y"], fill_value=0)
)
