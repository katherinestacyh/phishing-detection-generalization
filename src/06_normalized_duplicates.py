
from pathlib import Path
import hashlib
import re
import pandas as pd

sources = [
    "CEAS_08",
    "Enron",
    "Ling",
    "Nazario",
    "Nigerian_Fraud",
    "SpamAssasin"
]

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

all_data = []

for source in sources:
    df = pd.read_csv(Path("data") / f"{source}.csv")

    df["source_id"] = source

    subjects = df["subject"].apply(normalize_text)
    bodies = df["body"].apply(normalize_text)

    df["normalized_hash"] = [
        hashlib.sha256(
            (subject + "\0" + body).encode("utf-8")
        ).hexdigest()
        for subject, body in zip(subjects, bodies)
    ]

    all_data.append(
        df[["source_id", "label", "normalized_hash"]]
    )

combined = pd.concat(all_data, ignore_index=True)

groups = combined.groupby("normalized_hash").agg(
    records=("label", "size"),
    source_count=("source_id", "nunique"),
    label_count=("label", "nunique")
)

duplicates = groups[groups["records"] > 1]
cross_source = duplicates[duplicates["source_count"] > 1]
conflicting = duplicates[duplicates["label_count"] > 1]

print("\n===== NORMALIZED DUPLICATE ANALYSIS =====")
print(f"Total emails: {len(combined):,}")
print(f"Unique normalized emails: {len(groups):,}")
print(f"Duplicate groups: {len(duplicates):,}")
print(f"Records in duplicate groups: {duplicates['records'].sum():,}")
print(f"Extra duplicate copies: {(duplicates['records'] - 1).sum():,}")

print("\n===== CROSS-SOURCE DUPLICATES =====")
print(f"Groups appearing in multiple sources: {len(cross_source):,}")
print(f"Records in cross-source groups: {cross_source['records'].sum():,}")

print("\n===== CONFLICTING LABELS =====")
print(f"Groups with conflicting original labels: {len(conflicting):,}")


########################################################################################
print("\n===== CROSS-SOURCE DUPLICATE DETAILS =====")

for email_hash in cross_source.index:
    matching = combined[
        combined["normalized_hash"] == email_hash
    ]

    print("\nDuplicate group:")
    print("Sources:", ", ".join(
        sorted(matching["source_id"].unique())
    ))
    print("Records:", len(matching))
    print("Original labels:", sorted(
        matching["label"].unique().tolist()
    ))


########################################################################################
# Save aggregate findings for the interim report.
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

summary = pd.DataFrame([
    {"metric": "total_emails", "count": len(combined)},
    {"metric": "unique_normalized_emails", "count": len(groups)},
    {"metric": "duplicate_groups", "count": len(duplicates)},
    {"metric": "records_in_duplicate_groups", "count": int(duplicates["records"].sum())},
    {"metric": "extra_duplicate_copies", "count": int((duplicates["records"] - 1).sum())},
    {"metric": "cross_source_duplicate_groups", "count": len(cross_source)},
    {"metric": "cross_source_duplicate_records", "count": int(cross_source["records"].sum())},
    {"metric": "conflicting_label_groups", "count": len(conflicting)}
])

summary.to_csv(
    results_dir / "duplicate_summary.csv",
    index=False
)

print("\nSaved results/duplicate_summary.csv")
