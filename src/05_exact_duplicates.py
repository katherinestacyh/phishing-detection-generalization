
from pathlib import Path
from itertools import combinations
import hashlib
import pandas as pd

sources = [
    "CEAS_08",
    "Enron",
    "Ling",
    "Nazario",
    "Nigerian_Fraud",
    "SpamAssasin"
]

all_data = []

for source in sources:
    df = pd.read_csv(Path("data") / f"{source}.csv")

    # Preserve the original dataset name and labels.
    df["source_id"] = source

    # Compare the exact subject and body text.
    # Missing subjects or bodies are treated as empty strings.
    subjects = df["subject"].fillna("").astype(str)
    bodies = df["body"].fillna("").astype(str)

    df["email_hash"] = [
        hashlib.sha256(
            (subject + "\0" + body).encode("utf-8")
        ).hexdigest()
        for subject, body in zip(subjects, bodies)
    ]

    all_data.append(
        df[["source_id", "label", "email_hash"]]
    )

combined = pd.concat(all_data, ignore_index=True)

# Group emails with exactly matching subject and body.
groups = combined.groupby("email_hash").agg(
    records=("label", "size"),
    sources=("source_id", lambda x: tuple(sorted(set(x)))),
    source_count=("source_id", "nunique"),
    label_count=("label", "nunique")
)

duplicates = groups[groups["records"] > 1]
cross_source = duplicates[duplicates["source_count"] > 1]
conflicting = duplicates[duplicates["label_count"] > 1]

print("\n===== EXACT DUPLICATE ANALYSIS =====")
print(f"Total emails: {len(combined):,}")
print(f"Unique email texts: {len(groups):,}")
print(f"Duplicate groups: {len(duplicates):,}")
print(f"Records in duplicate groups: {duplicates['records'].sum():,}")
print(f"Extra duplicate copies: {(duplicates['records'] - 1).sum():,}")

print("\n===== CROSS-SOURCE DUPLICATES =====")
print(f"Groups appearing in multiple sources: {len(cross_source):,}")
print(f"Records in cross-source groups: {cross_source['records'].sum():,}")

print("\n===== LABEL CONFLICTS =====")
print(f"Duplicate groups with conflicting labels: {len(conflicting):,}")

# Count duplicate groups shared by each pair of sources.
pair_counts = {}

for source_group in cross_source["sources"]:
    for pair in combinations(source_group, 2):
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

print("\n===== OVERLAPPING SOURCE PAIRS =====")
if pair_counts:
    for pair, count in sorted(
        pair_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{pair[0]} / {pair[1]}: {count:,} groups")
else:
    print("No exact cross-source matches found.")
