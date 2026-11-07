import pandas as pd

# Load the three datasets
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/validation.csv")
test = pd.read_csv("data/test.csv")

features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]


def configuration_set(df):
    return set(
        map(tuple, df[features].to_numpy())
    )


train_configs = configuration_set(train)
val_configs = configuration_set(val)
test_configs = configuration_set(test)

print("\nDATASET CONFIGURATION OVERLAP AUDIT")
print("=" * 45)

print("Unique training configurations:", len(train_configs))
print("Unique validation configurations:", len(val_configs))
print("Unique test configurations:", len(test_configs))

print("\nOverlapping configurations:")

print(
    "Train vs Validation:",
    len(train_configs & val_configs)
)

print(
    "Train vs Test:",
    len(train_configs & test_configs)
)

print(
    "Validation vs Test:",
    len(val_configs & test_configs)
)

print("\nRows with configurations seen during training:")

for name, df in [("Validation", val), ("Test", test)]:
    seen = df[features].apply(
        lambda row: tuple(row) in train_configs,
        axis=1
    )

    print(
        f"{name}: {seen.sum()} / {len(df)} "
        f"({seen.mean() * 100:.2f}%)"
    )
