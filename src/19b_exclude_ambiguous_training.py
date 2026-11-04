from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"

INPUT = DATA / "controlled_train.csv"
OUTPUT = DATA / "controlled_train_label_checked.csv"
SUMMARY = RESULTS / "ambiguous_label_exclusion_summary.csv"


def main():

    print("Loading controlled training dataset...")

    train = pd.read_csv(INPUT, low_memory=False)

    # Check that the required columns exist.
    required = {"leakage_group_id", "project_label"}

    if not required.issubset(train.columns):
        raise ValueError("Required columns are missing.")

    if train[list(required)].isna().any().any():
        raise ValueError("Missing group IDs or labels found.")

    # Step 1: Find groups containing different labels.
    label_counts = (
        train.groupby("leakage_group_id")["project_label"]
        .nunique()
    )

    conflicting_groups = label_counts[
        label_counts > 1
    ].index

    # Step 2: Separate ambiguous emails from other emails.
    ambiguous = train[
        train["leakage_group_id"].isin(conflicting_groups)
    ]

    clean_train = train[
        ~train["leakage_group_id"].isin(conflicting_groups)
    ].copy()

    # Step 3: Verify no conflicting-label groups remain.
    remaining_conflicts = (
        clean_train.groupby("leakage_group_id")["project_label"]
        .nunique() > 1
    ).sum()

    assert remaining_conflicts == 0

    # Step 4: Save the corrected training dataset.
    clean_train.to_csv(OUTPUT, index=False)

    # Step 5: Save statistics for our report.
    summary = {
        "original_training_emails": len(train),
        "conflicting_label_groups": len(conflicting_groups),
        "ambiguous_emails_excluded": len(ambiguous),
        "training_emails_remaining": len(clean_train),
        "remaining_conflicting_groups": remaining_conflicts
    }

    pd.DataFrame(
        summary.items(),
        columns=["metric", "count"]
    ).to_csv(SUMMARY, index=False)

    print("\n===== LABEL QUALITY CHECK =====")

    for metric, count in summary.items():
        print(f"{metric}: {count:,}")

    print("\nSaved data/controlled_train_label_checked.csv")
    print("Saved results/ambiguous_label_exclusion_summary.csv")

    print("\nOriginal training dataset remains unchanged.")
    print("Validation and test datasets remain unchanged.")
    print("Ambiguous emails were excluded from model fitting only.")


if __name__ == "__main__":
    main()
