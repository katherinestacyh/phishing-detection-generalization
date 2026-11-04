from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

INPUT_EMAILS = DATA / "spam_near_duplicate_grouped.csv"
INPUT_PAIRS = RESULTS / "near_duplicate_pairs.csv"
INPUT_SUMMARY = RESULTS / "near_duplicate_summary.csv"

OUTPUT_EMAILS = DATA / "spam_near_duplicate_refined.csv"
OUTPUT_FLAGS = RESULTS / "near_duplicate_refinement_flags.csv"
OUTPUT_SUMMARY = RESULTS / "near_duplicate_refinement_summary.csv"

# Stricter similarity requirements
MIN_JACCARD = 0.90
MIN_CHAR_SIMILARITY = 97.0

# Rules for identifying suspicious families
LARGE_FAMILY_SIZE = 100
SUBJECT_REVIEW_MIN_SIZE = 10


class UnionFind:
    """Keep track of which emails belong to the same group."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)

        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def main():

    # STEP 1: Load the previous results
    for path in (INPUT_EMAILS, INPUT_PAIRS):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run Script 16 first."
            )

    print("Reading Script 16 results...")

    df = pd.read_csv(INPUT_EMAILS, low_memory=False)

    required = {
        "duplicate_group_id",
        "near_duplicate_group_id",
        "source_id",
        "source_row",
        "project_label",
        "subject",
        "model_text"
    }

    if missing := required - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df[list(required - {"subject"})].isna().any().any():
        raise ValueError(
            "Missing IDs, labels, or model text in the input data."
        )

    # STEP 2: Preserve existing exact-duplicate groups
    reps = df.drop_duplicates(
        "duplicate_group_id"
    ).reset_index(drop=True)

    key_to_rep = {
        (str(r.source_id), int(r.source_row)): i
        for i, r in enumerate(reps.itertuples(index=False))
    }

    if len(key_to_rep) != len(reps):
        raise ValueError(
            "Source ID + source row are not unique."
        )

    uf = UnionFind(len(reps))

    checked = 0
    accepted = 0
    too_weak = 0
    conflicting = 0

    # STEP 3: Recheck saved matches with stricter rules
    columns = [
        "source_a",
        "row_a",
        "source_b",
        "row_b",
        "jaccard",
        "char_similarity"
    ]

    for chunk in pd.read_csv(
        INPUT_PAIRS,
        usecols=columns,
        chunksize=100_000
    ):

        for edge in chunk.itertuples(index=False):
            checked += 1

            key_a = (str(edge.source_a), int(edge.row_a))
            key_b = (str(edge.source_b), int(edge.row_b))

            if key_a not in key_to_rep or key_b not in key_to_rep:
                raise ValueError(
                    f"Saved pair not found: {key_a}, {key_b}"
                )

            a = key_to_rep[key_a]
            b = key_to_rep[key_b]

            # Never automatically combine different labels
            if (
                reps.at[a, "project_label"]
                != reps.at[b, "project_label"]
            ):
                conflicting += 1
                continue

            # Reject weaker similarity matches
            if (
                edge.jaccard < MIN_JACCARD
                or edge.char_similarity < MIN_CHAR_SIMILARITY
            ):
                too_weak += 1
                continue

            uf.join(a, b)
            accepted += 1

    # STEP 4: Assign refined group IDs
    reps["refined_group_id"] = [
        f"RF_{uf.find(i):06d}"
        for i in range(len(reps))
    ]

    group_map = dict(
        zip(
            reps["duplicate_group_id"],
            reps["refined_group_id"]
        )
    )

    df["refined_group_id"] = (
        df["duplicate_group_id"].map(group_map)
    )

    # Keep the broader Script 16 families as split boundaries.
    # This helps prevent previously detected possible duplicates
    # from appearing in different train/test sets.
    df["leakage_group_id"] = df["near_duplicate_group_id"]

    # STEP 5: Examine characteristics of each original family
    df["subject_normalized"] = (
        df["subject"]
        .fillna("")
        .astype(str)
        .str.casefold()
        .str.strip()
    )

    family = df.groupby("leakage_group_id").agg(
        emails=("project_label", "size"),
        labels=("project_label", "nunique"),
        sources=("source_id", "nunique"),
        subjects=("subject_normalized", "nunique"),
        refined_parts=("refined_group_id", "nunique")
    )

    df.drop(columns=["subject_normalized"], inplace=True)

    # STEP 6: Identify families needing further review
    family["conflicting_labels"] = family["labels"] > 1

    family["very_large"] = (
        family["emails"] >= LARGE_FAMILY_SIZE
    )

    family["varied_subjects"] = (
        (family["emails"] >= SUBJECT_REVIEW_MIN_SIZE)
        & (family["subjects"] > 1)
    )

    family["split_by_stricter_rules"] = (
        (family["emails"] >= SUBJECT_REVIEW_MIN_SIZE)
        & (family["refined_parts"] > 1)
    )

    flags = [
        "conflicting_labels",
        "very_large",
        "varied_subjects",
        "split_by_stricter_rules"
    ]

    # Temporarily set questionable families aside.
    family["exclude_from_modeling"] = family[flags].any(axis=1)

    family["review_reason"] = family[flags].apply(
        lambda r: "; ".join(
            k for k in flags if r[k]
        ),
        axis=1
    )

    df["exclude_from_modeling"] = (
        df["leakage_group_id"].map(
            family["exclude_from_modeling"]
        )
    )

    df["review_reason"] = (
        df["leakage_group_id"].map(
            family["review_reason"]
        )
    )

    # STEP 7: Confirm exact duplicates were not separated
    assert (
        df.groupby("duplicate_group_id")[
            "refined_group_id"
        ].nunique() == 1
    ).all()

    assert (
        df.groupby("duplicate_group_id")[
            "leakage_group_id"
        ].nunique() == 1
    ).all()

    # STEP 8: Save the refined results
    # Contains email text: keep inside private data folder.
    df.to_csv(OUTPUT_EMAILS, index=False)

    family[
        family["exclude_from_modeling"]
    ].reset_index().to_csv(
        OUTPUT_FLAGS,
        index=False
    )

    # Read unresolved search limitations from Script 16
    truncated = "unavailable"
    skipped_short = "unavailable"

    if INPUT_SUMMARY.exists():
        old_summary = pd.read_csv(
            INPUT_SUMMARY
        ).set_index("metric")["count"]

        truncated = old_summary.get(
            "searches_truncated_by_safety_limit",
            truncated
        )

        skipped_short = old_summary.get(
            "short_representatives_skipped",
            skipped_short
        )

    # STEP 9: Create summary statistics
    summary = {
        "total_emails": len(df),
        "original_script16_families": family.shape[0],
        "stricter_refined_groups": df["refined_group_id"].nunique(),
        "saved_pairs_checked": checked,
        "pairs_kept_by_stricter_thresholds": accepted,
        "pairs_rejected_low_similarity": too_weak,
        "pairs_rejected_conflicting_labels": conflicting,
        "uncertain_families_set_aside": int(
            family["exclude_from_modeling"].sum()
        ),
        "emails_set_aside": int(
            df["exclude_from_modeling"].sum()
        ),
        "emails_remaining_for_later_splitting": int(
            (~df["exclude_from_modeling"]).sum()
        ),
        "cross_source_families": int(
            (family["sources"] > 1).sum()
        ),
        "script16_truncated_searches_UNRESOLVED": truncated,
        "script16_short_representatives_skipped_UNRESOLVED": skipped_short
    }

    pd.DataFrame(
        summary.items(),
        columns=["metric", "count"]
    ).to_csv(OUTPUT_SUMMARY, index=False)

    print("\n===== REFINEMENT RESULTS =====")

    for name, value in summary.items():
        if isinstance(value, int):
            print(f"{name}: {value:,}")
        else:
            print(f"{name}: {value}")

    print("\nSaved data/spam_near_duplicate_refined.csv (PRIVATE)")
    print("Saved results/near_duplicate_refinement_flags.csv")
    print("Saved results/near_duplicate_refinement_summary.csv")

    print("\nNOTE: No new matches were discovered.")
    print("Truncated searches are still unresolved.")
    print("NEXT: review the summary before creating controlled splits.")


if __name__ == "__main__":
    main()
