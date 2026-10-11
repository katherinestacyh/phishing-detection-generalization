
from pathlib import Path
import hashlib
import pandas as pd

project_dir = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    project_dir / "data/spam_modeling_candidates.csv"
)

def normalize(text):
    if pd.isna(text):
        return ""
    return " ".join(str(text).lower().split())

df["normalized_hash"] = [
    hashlib.sha256(
        (normalize(subject) + "\0" + normalize(body))
        .encode("utf-8")
    ).hexdigest()
    for subject, body in zip(df["subject"], df["body"])
]

groups = df.groupby("normalized_hash").agg(
    records=("project_label", "size"),
    sources=("source_id", "nunique"),
    labels=("project_label", "nunique")
)

duplicates = groups[groups["records"] > 1]
cross_source = duplicates[duplicates["sources"] > 1]
conflicts = duplicates[duplicates["labels"] > 1]

summary = {
    "total_emails": len(df),
    "unique_normalized_emails": len(groups),
    "duplicate_groups": len(duplicates),
    "records_in_duplicate_groups": int(duplicates["records"].sum()),
    "extra_duplicate_copies": int(
        (duplicates["records"] - 1).sum()
    ),
    "cross_source_duplicate_groups": len(cross_source),
    "conflicting_label_groups": len(conflicts)
}

print("\n===== SPAM DUPLICATE AUDIT =====")

for metric, value in summary.items():
    print(f"{metric}: {value:,}")

results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)

pd.DataFrame(
    list(summary.items()),
    columns=["metric", "count"]
).to_csv(
    results_dir / "spam_duplicate_summary.csv",
    index=False
)

print("\nSaved results/spam_duplicate_summary.csv")


# Assign the same group ID to emails with
# identical normalized subject and body text.
df["duplicate_group_id"] = df["normalized_hash"]

# Save locally without changing the original files.
output_path = (
    project_dir / "data/spam_with_duplicate_groups.csv"
)

df.drop(columns=["normalized_hash"]).to_csv(
    output_path,
    index=False
)

print("\n===== DUPLICATE GROUP ASSIGNMENT =====")
print(f"Total records: {len(df):,}")
print(f"Unique groups: {df['duplicate_group_id'].nunique():,}")
print(f"Saved {output_path.name}")
