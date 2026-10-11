
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sources = [
    "CEAS_08",
    "Enron",
    "Ling",
    "Nazario",
    "Nigerian_Fraud",
    "SpamAssasin"
]

results = []
plot_data = []

for source in sources:
    df = pd.read_csv(Path("data") / f"{source}.csv")

    # Count the number of characters in each email body.
    lengths = df["body"].fillna("").astype(str).str.len()

    results.append({
        "source": source,
        "emails": len(df),
        "mean_length": round(lengths.mean(), 2),
        "median_length": round(lengths.median(), 2),
        "minimum_length": lengths.min(),
        "maximum_length": lengths.max(),
        "p95_length": round(lengths.quantile(0.95), 2)
    })

    plot_data.append(lengths.to_numpy())

summary = pd.DataFrame(results)

print("\n===== EMAIL BODY LENGTH BY SOURCE =====")
print(summary.to_string(index=False))

# Create a boxplot showing the distribution of email lengths.
fig, ax = plt.subplots(figsize=(11, 6))

ax.boxplot(
    plot_data,
    tick_labels=sources,
    showfliers=False
)

ax.set_title("Email Body Length Distribution by Source")
ax.set_xlabel("Email Source")
ax.set_ylabel("Email Body Length (Characters)")

plt.xticks(rotation=25, ha="right")
plt.tight_layout()

# Save the graph and the results.
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

plt.savefig(
    results_dir / "figure2_email_length.png",
    dpi=300,
    bbox_inches="tight"
)

summary.to_csv(
    results_dir / "email_length_summary.csv",
    index=False
)

print("\nSaved results/figure2_email_length.png")
print("Saved results/email_length_summary.csv")

plt.show()
