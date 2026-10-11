
from pathlib import Path
import pandas as pd

# Project folders
project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
results_dir = project_dir / "results"

results_dir.mkdir(exist_ok=True)

# Only the four spam/ham corpora
sources = [
    "CEAS_08",
    "Enron",
    "Ling",
    "SpamAssasin"
]

all_emails = []

for source in sources:
    df = pd.read_csv(data_dir / f"{source}.csv")

    # Preserve original source and row number
    df["source_id"] = source
    df["source_row"] = range(len(df))
    df["original_label"] = df["label"]

    # Provisional spam-detection labels
    # 0 = legitimate, 1 = spam
    df["project_label"] = df["original_label"]

    # Standardize columns across sources
    columns = [
        "source_id",
        "source_row",
        "original_label",
        "project_label",
        "subject",
        "body",
        "sender",
        "receiver",
        "date",
        "urls"
    ]

    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA

    all_emails.append(df[columns])

# Combine the four corpora
spam_data = pd.concat(
    all_emails,
    ignore_index=True
)

# Save original content locally
spam_data.to_csv(
    data_dir / "spam_modeling_candidates.csv",
    index=False
)

# Create an aggregate summary
summary = spam_data.groupby("source_id").agg(
    total_emails=("project_label", "size"),
    legitimate=("project_label", lambda x: (x == 0).sum()),
    spam=("project_label", lambda x: (x == 1).sum())
).reset_index()

summary.to_csv(
    results_dir / "spam_dataset_summary.csv",
    index=False
)

print("\n===== SPAM DETECTION DATASET =====")
print(f"Total emails: {len(spam_data):,}")

print("\n===== CLASS DISTRIBUTION =====")
print(spam_data["project_label"].value_counts().sort_index())

print("\n===== BY SOURCE =====")
print(summary.to_string(index=False))

print("\n===== DATA QUALITY =====")
print(f"Missing subjects: {spam_data['subject'].isna().sum():,}")
print(f"Missing bodies: {spam_data['body'].isna().sum():,}")

print("\nSaved data/spam_modeling_candidates.csv")
print("Saved results/spam_dataset_summary.csv")
