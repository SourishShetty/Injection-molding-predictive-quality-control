
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/Dataset.csv")

# Compare injection temperature across viability classes
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="label_viability",
    y="Tinj",
    order=["G", "Y", "R"]
)

plt.xlabel("Viability Class")
plt.ylabel("Injection Temperature (°C)")
plt.title("Injection Temperature vs. Process Viability")

plt.tight_layout()
plt.savefig("results/temperature_boxplot.png", dpi=300)
plt.show()


# Compare all six machine settings across viability classes
features = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for ax, feature in zip(axes.flat, features):
    sns.boxplot(
        data=df,
        x="label_viability",
        y=feature,
        order=["G", "Y", "R"],
        ax=ax
    )

    ax.set_title(feature)
    ax.set_xlabel("Viability Class")

plt.suptitle("Process Parameters vs. Viability", fontsize=15)
plt.tight_layout()

plt.savefig("results/all_features_boxplots.png", dpi=300)
plt.show()
