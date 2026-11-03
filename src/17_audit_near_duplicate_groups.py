from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

INPUT = DATA / "spam_near_duplicate_grouped.csv"
EXISTING_SUMMARY = RESULTS / "near_duplicate_summary.csv"
GROUP_REPORT = RESULTS / "near_duplicate_groups_to_review.csv"
AUDIT_SUMMARY = RESULTS / "near_duplicate_audit_summary.csv"
PRIVATE_EXAMPLES = DATA / "near_duplicate_review_examples_PRIVATE.csv"

# Review rules only; these flags do not prove a group is incorrect.
LARGE_GROUP_MIN_RECORDS = 100
SUBJECT_VARIATION_MIN_RECORDS = 10
REVIEW_TOP_N_CROSS_SOURCE = 20
REVIEW_TOP_N_SUBJECT_VARIANTS = 10
REVIEW_TOP_N_LARGEST = 10
REVIEW_RANDOM_SMALL_GROUPS = 25
REVIEW_RANDOM_SEED = 42


def compact_preview(value, limit=220):
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())[:limit]


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(
            f"Cannot find {INPUT}. Run script 16 first from your repository."
        )

    fields = [
        "near_duplicate_group_id", "duplicate_group_id", "project_label",
        "source_id", "source_row", "subject", "model_text",
    ]
    df = pd.read_csv(INPUT, usecols=fields, low_memory=False)
    if df[fields[:5]].isna().any().any():
        raise ValueError("Missing group IDs, labels, or source identifiers; inspect the data.")

    print(f"Auditing {len(df):,} email records...")
    df["text_char_length"] = df["model_text"].fillna("").str.len()
    families = (
        df.groupby("near_duplicate_group_id", sort=False)
        .agg(
            record_count=("project_label", "size"),
            exact_groups=("duplicate_group_id", "nunique"),
            distinct_labels=("project_label", "nunique"),
            sources=("source_id", "nunique"),
            distinct_subjects=("subject", "nunique"),
            min_text_length=("text_char_length", "min"),
            max_text_length=("text_char_length", "max"),
        )
        .reset_index()
    )
    families["has_label_conflict"] = families["distinct_labels"] > 1
    families["is_large_group"] = families["record_count"] >= LARGE_GROUP_MIN_RECORDS
    families["is_cross_source"] = families["sources"] > 1
    families["varying_subjects"] = (
        (families["record_count"] >= SUBJECT_VARIATION_MIN_RECORDS)
        & (families["distinct_subjects"] > 1)
    )

    # Focus review on contradictory labels, unusually large families,
    # the biggest cross-source families, and some groups with varied subjects.
    selected_ids = set(families.loc[
        families["has_label_conflict"] | families["is_large_group"],
        "near_duplicate_group_id",
    ])
    selected_ids.update(
        families.nlargest(REVIEW_TOP_N_LARGEST, "record_count")
        ["near_duplicate_group_id"]
    )
    selected_ids.update(
        families[families["is_cross_source"]]
        .nlargest(REVIEW_TOP_N_CROSS_SOURCE, "record_count")
        ["near_duplicate_group_id"]
    )
    selected_ids.update(
        families[families["varying_subjects"]]
        .nlargest(REVIEW_TOP_N_SUBJECT_VARIANTS, "record_count")
        ["near_duplicate_group_id"]
    )

    # Also sample ordinary-sized matched families to estimate whether the
    # matching method is making false-positive merges beyond the outliers.
    ordinary_groups = families[
        (families["record_count"] >= 2)
        & (families["record_count"] < LARGE_GROUP_MIN_RECORDS)
        & (~families["near_duplicate_group_id"].isin(selected_ids))
    ]
    if not ordinary_groups.empty:
        random_review = ordinary_groups.sample(
            n=min(REVIEW_RANDOM_SMALL_GROUPS, len(ordinary_groups)),
            random_state=REVIEW_RANDOM_SEED,
        )
        selected_ids.update(random_review["near_duplicate_group_id"])

    review = families[families["near_duplicate_group_id"].isin(selected_ids)].copy()
    review = review.sort_values(
        ["has_label_conflict", "is_large_group", "record_count"],
        ascending=[False, False, False],
    )
    review.to_csv(GROUP_REPORT, index=False)

    # Store sample email snippets ONLY in the private, gitignored data folder.
    sample_rows = []
    selected = df[df["near_duplicate_group_id"].isin(selected_ids)].copy()
    for group_id, subset in selected.groupby("near_duplicate_group_id", sort=False):
        subset = subset.sort_values(["source_id", "source_row"])
        # Include one example of each label if the group has conflicting labels.
        picks = list(subset.groupby("project_label", sort=True).head(1).index)
        # Then cover the range of IDs, rather than just the first adjacent rows.
        for pos in (0, len(subset) // 2, len(subset) - 1):
            picks.append(subset.index[pos])
        for idx in dict.fromkeys(picks):
            row = subset.loc[idx]
            sample_rows.append({
                "near_duplicate_group_id": group_id,
                "source_id": row["source_id"],
                "source_row": row["source_row"],
                "project_label": row["project_label"],
                "subject_preview": compact_preview(row["subject"], 120),
                "text_preview": compact_preview(row["model_text"], 220),
            })

    pd.DataFrame(sample_rows, columns=[
        "near_duplicate_group_id", "source_id", "source_row",
        "project_label", "subject_preview", "text_preview",
    ]).to_csv(PRIVATE_EXAMPLES, index=False)

    summary = {
        "total_records": len(df),
        "total_combined_groups": len(families),
        "groups_with_multiple_records": int((families["record_count"] > 1).sum()),
        "groups_with_label_conflicts": int(families["has_label_conflict"].sum()),
        "groups_with_100_or_more_records": int(families["is_large_group"].sum()),
        "cross_source_groups": int(families["is_cross_source"].sum()),
        "groups_with_varied_subjects_10plus_records": int(families["varying_subjects"].sum()),
        "groups_selected_for_manual_review": len(review),
        "random_small_groups_selected": min(REVIEW_RANDOM_SMALL_GROUPS, len(ordinary_groups)),
        "largest_group_records": int(families["record_count"].max()),
    }
    if EXISTING_SUMMARY.is_file():
        original = pd.read_csv(EXISTING_SUMMARY).set_index("metric")["count"]
        for key in ["searches_truncated_by_safety_limit", "short_representatives_skipped"]:
            if key in original:
                summary[key] = int(original[key])

    pd.DataFrame(summary.items(), columns=["metric", "count"]).to_csv(
        AUDIT_SUMMARY, index=False
    )
    print("\n===== NEAR-DUPLICATE GROUP AUDIT =====")
    for key, value in summary.items():
        print(f"{key}: {value:,}")
    print(f"\nSaved {GROUP_REPORT.relative_to(PROJECT)}")
    print(f"Saved {AUDIT_SUMMARY.relative_to(PROJECT)}")
    print(f"Saved {PRIVATE_EXAMPLES.relative_to(PROJECT)} (PRIVATE; do not commit)")
    print("\nNEXT: review the flagged examples, then refine script 16 as needed.")
    print("Do not split or train using the current near-duplicate groups yet.")


if __name__ == "__main__":
    main()
