
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sources = [
    "CEAS_08",
    "SpamAssasin"
]

results = []

for source in sources:
    df = pd.read_csv(Path("data") / f"{source}.csv")

    total = len(df)
    url_present = (df["urls"] == 1).sum()
    url_absent = (df["urls"] == 0).sum()

    results.append({
        "source": source,
        "total_emails": total,
        "url_present": url_present,
        "url_absent": url_absent,
        "url_present_percent": round(
            url_present / total * 100, 2
        )
    })

summary = pd.DataFrame(results)

print("\n===== URL PRESENCE BY SOURCE =====")
print(summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    summary["source"],
    summary["url_present_percent"]
)

ax.set_title("Percentage of Emails with URL Indicator by Source")
ax.set_xlabel("Email Source")
ax.set_ylabel("Emails with URL Indicator = 1 (%)")
ax.set_ylim(0, 100)

for bar, value in zip(
    bars,
    summary["url_present_percent"]
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f"{value:.1f}%",
        ha="center"
    )

plt.xticks(rotation=20, ha="right")
plt.tight_layout()

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

plt.savefig(
    results_dir / "figure3_url_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

summary.to_csv(
    results_dir / "url_distribution_summary.csv",
    index=False
)

print("\nSaved results/figure3_url_distribution.png")
print("Saved results/url_distribution_summary.csv")

plt.show()
