
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
results_dir = project_dir / "results"

results_dir.mkdir(exist_ok=True)

sources = [
    "CEAS_08",
    "Enron",
    "Ling",
    "Nazario",
    "Nigerian_Fraud",
    "SpamAssasin"
]

prepared_files = []

for source in sources:
    df = pd.read_csv(data_dir / f"{source}.csv")

    # Keep track of where every email came from.
    df["source_id"] = source
    df["source_row"] = range(len(df))

    # Preserve the original labels.
    df["original_label"] = df["label"]

    # Create provisional categories.
    # These are NOT verified project labels.
    if source == "Nazario":
        df["provisional_category"] = "phishing_candidate"

    elif source == "Nigerian_Fraud":
        df["provisional_category"] = "fraud_candidate"

    else:
        df["provisional_category"] = df["label"].map({
            0: "legitimate_candidate",
            1: "general_spam"
        }).fillna("uncertain")

    # Final phishing labels are not assigned yet.
    df["project_label"] = pd.NA

    # Make sure all sources have the same columns.
    columns = [
        "source_id",
        "source_row",
        "original_label",
        "provisional_category",
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

    prepared_files.append(df[columns])

# Combine all six sources.
prepared = pd.concat(
    prepared_files,
    ignore_index=True
)

# Save locally, NOT to public GitHub.
prepared.to_csv(
    data_dir / "prepared_email_records.csv",
    index=False
)

# Create a summary without private email contents.
summary = pd.crosstab(
    prepared["source_id"],
    prepared["provisional_category"]
)

summary.to_csv(
    results_dir / "provisional_label_summary.csv"
)

print("\n===== PREPARED DATASET =====")
print(f"Total emails: {len(prepared):,}")

print("\n===== PROVISIONAL CATEGORIES =====")
print(
    prepared["provisional_category"]
    .value_counts()
    .to_string()
)

print("\n===== CATEGORIES BY SOURCE =====")
print(summary.to_string())

print("\nSaved data/prepared_email_records.csv")
print("Saved results/provisional_label_summary.csv")
