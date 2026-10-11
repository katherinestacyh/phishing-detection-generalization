
from pathlib import Path
import pandas as pd

# Project folders
project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
results_dir = project_dir / "results"

results_dir.mkdir(exist_ok=True)

# Load our prepared spam dataset with duplicate group IDs
df = pd.read_csv(
    data_dir / "spam_with_duplicate_groups.csv",
    low_memory=False
)

print("\n===== BEFORE CLEANING =====")
print(f"Total emails: {len(df):,}")

# Preserve the original subject and body columns.
# Create separate cleaned versions for modeling.

def clean_text(value):
    if pd.isna(value):
        return ""

    # Remove extra spaces, tabs, and line breaks.
    return " ".join(str(value).split())

# Identify missing or empty text before cleaning.
subject_missing = (
    df["subject"].fillna("").astype(str).str.strip() == ""
)

body_missing = (
    df["body"].fillna("").astype(str).str.strip() == ""
)

print(f"Missing or empty subjects: {subject_missing.sum():,}")
print(f"Missing or empty bodies: {body_missing.sum():,}")

# Clean subject and body separately.
df["clean_subject"] = df["subject"].apply(clean_text)
df["clean_body"] = df["body"].apply(clean_text)

# Combine subject and body into one modeling field.
df["model_text"] = (
    df["clean_subject"] + " " + df["clean_body"]
).str.strip()

# Identify records with no usable text.
unusable = df["model_text"].eq("")

print(f"Emails with no usable text: {unusable.sum():,}")

# Keep track of how many records were excluded.
original_count = len(df)
excluded_count = int(unusable.sum())

# Keep emails with at least some usable text.
cleaned_df = df.loc[~unusable].copy()

# Check that duplicate group IDs were preserved.
assert cleaned_df["duplicate_group_id"].notna().all()

# Save cleaned dataset locally.
cleaned_df.to_csv(
    data_dir / "spam_cleaned.csv",
    index=False
)

# Create a summary for our report.
summary = [
    {"metric": "original_records", "count": original_count},
    {"metric": "missing_or_empty_subjects", "count": int(subject_missing.sum())},
    {"metric": "missing_or_empty_bodies", "count": int(body_missing.sum())},
    {"metric": "records_excluded_no_text", "count": excluded_count},
    {"metric": "cleaned_records", "count": len(cleaned_df)},
    {"metric": "legitimate_records", "count": int((cleaned_df["project_label"] == 0).sum())},
    {"metric": "spam_records", "count": int((cleaned_df["project_label"] == 1).sum())},
    {"metric": "unique_duplicate_groups", "count": cleaned_df["duplicate_group_id"].nunique()}
]

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    results_dir / "spam_cleaning_summary.csv",
    index=False
)

print("\n===== AFTER CLEANING =====")
print(f"Original emails: {original_count:,}")
print(f"Excluded emails: {excluded_count:,}")
print(f"Cleaned emails: {len(cleaned_df):,}")

print("\n===== CLEANED CLASS DISTRIBUTION =====")
print(cleaned_df["project_label"].value_counts().sort_index())

print("\n===== DUPLICATE GROUPS =====")
print(f"Unique groups preserved: {cleaned_df['duplicate_group_id'].nunique():,}")

print("\n===== FILES SAVED =====")
print("data/spam_cleaned.csv")
print("results/spam_cleaning_summary.csv")
