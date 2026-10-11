
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sources = [
    "CEAS_08",
    "Enron",
    "Ling",
    "SpamAssasin"
]

results = []

for source in sources:
    df = pd.read_csv(Path("data") / f"{source}.csv")

    results.append({
        "source": source,
        "label_0": (df["label"] == 0).sum(),
        "label_1": (df["label"] == 1).sum()
    })

summary = pd.DataFrame(results)

print("\n===== LABEL DISTRIBUTION BY SOURCE =====")
print(summary.to_string(index=False))

# Create a stacked bar chart.
fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    summary["source"],
    summary["label_0"],
    label="Original label 0"
)

ax.bar(
    summary["source"],
    summary["label_1"],
    bottom=summary["label_0"],
    label="Original label 1"
)

ax.set_title(
    "Email Distribution and Original Labels by Source"
)
ax.set_xlabel("Email source")
ax.set_ylabel("Number of emails")
ax.legend()

plt.xticks(rotation=25, ha="right")
plt.tight_layout()

# Save the graph.
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

plt.savefig(
    results_dir / "figure1_source_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved results/figure1_source_distribution.png")

plt.show()
