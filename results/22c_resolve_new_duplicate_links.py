
from pathlib import Path
import pandas as pd


# STEP 1: Set up project folders

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

LINKS = RESULTS / "22b_new_cross_family_links.csv"

SPLITS = {
    "train": DATA / "controlled_train.csv",
    "validation": DATA / "controlled_validation.csv",
    "test": DATA / "controlled_test.csv",
}

TRAIN_CHECKED = DATA / "controlled_train_label_checked.csv"

OUT_TEST = DATA / "controlled_test_leakage_checked.csv"
OUT_AUDIT = RESULTS / "22c_new_link_review.csv"
OUT_SUMMARY = RESULTS / "22c_resolution_summary.csv"


# STEP 2: Use the stricter thresholds from Script 18

STRICT_JACCARD = 0.90
STRICT_CHAR_SIMILARITY = 97.0


def row_keys(frame):
    """Identify records using source and original row number."""

    return list(zip(
        frame["source_id"].astype(str),
        frame["source_row"].astype(int)
    ))


def main():

    # STEP 3: Check required input files

    for path in [LINKS, *SPLITS.values(), TRAIN_CHECKED]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

    print("Loading new duplicate links...")

    links = pd.read_csv(LINKS)

    required = {
        "source_a", "row_a",
        "source_b", "row_b",
        "split_a", "split_b",
        "jaccard", "char_similarity",
        "old_family_a", "old_family_b",
        "label_conflict"
    }

    if not required.issubset(links.columns):
        raise ValueError(
            f"Missing columns: "
            f"{sorted(required - set(links.columns))}"
        )

    # STEP 4: Verify existing split assignments

    actual_splits = {}

    for split_name, path in SPLITS.items():

        ids = pd.read_csv(
            path,
            usecols=["source_id", "source_row"]
        )

        for key in row_keys(ids):

            if key in actual_splits:
                raise ValueError(
                    f"Email appears in multiple splits: {key}"
                )

            actual_splits[key] = split_name

    # This is the training data used by Script 20.
    train_checked = pd.read_csv(
        TRAIN_CHECKED,
        usecols=["source_id", "source_row"]
    )

    checked_keys = set(row_keys(train_checked))

    audit = []
    test_keys_to_quarantine = set()

    # STEP 5: Review every new duplicate connection

    for n, link in links.iterrows():

        a = (
            str(link["source_a"]),
            int(link["row_a"])
        )

        b = (
            str(link["source_b"]),
            int(link["row_b"])
        )

        if a not in actual_splits or b not in actual_splits:
            raise ValueError(
                f"Link {n + 1} contains a missing email."
            )

        split_a = actual_splits[a]
        split_b = actual_splits[b]

        # Make sure Script 19's splits have not changed.
        if (
            split_a != link["split_a"]
            or split_b != link["split_b"]
        ):
            raise ValueError(
                "Splits have changed since Script 22b. "
                "Recheck the data before continuing."
            )

        # Both strict thresholds must be satisfied.
        strict_match = (
            float(link["jaccard"]) >= STRICT_JACCARD
            and float(link["char_similarity"])
            >= STRICT_CHAR_SIMILARITY
        )

        cross_split = split_a != split_b
        involved_test = "test" in (split_a, split_b)

        strong_test_link = (
            strict_match
            and cross_split
            and involved_test
        )

        # Confirm that the matching training record
        # was actually included in model training.
        if strong_test_link:

            other_key = b if split_a == "test" else a
            other_split = actual_splits[other_key]

            if (
                other_split == "train"
                and other_key not in checked_keys
            ):
                strong_test_link = False

        action = "no_split_action"

        if strict_match and cross_split and not involved_test:

            action = "needs_train_validation_correction"

        elif strong_test_link:

            test_key = a if split_a == "test" else b

            test_keys_to_quarantine.add(test_key)

            action = "quarantine_from_reserved_test"

        elif cross_split:

            action = "review_weaker_cross_split_template_match"

        elif strict_match:

            action = "strong_match_within_same_split"

        # Save metadata only, not raw email text.
        audit.append({
            "link_number": n + 1,
            "source_a": a[0],
            "row_a": a[1],
            "split_a": split_a,
            "source_b": b[0],
            "row_b": b[1],
            "split_b": split_b,
            "old_family_a": str(link["old_family_a"]),
            "old_family_b": str(link["old_family_b"]),
            "jaccard": float(link["jaccard"]),
            "char_similarity": float(
                link["char_similarity"]
            ),
            "strict_match": strict_match,
            "cross_split": cross_split,
            "cross_source": a[0] != b[0],
            "label_conflict": (
                str(link["label_conflict"]).lower()
                == "true"
            ),
            "proposed_action": action
        })

    review = pd.DataFrame(audit)

    if review.empty:
        raise ValueError(
            "No new duplicate links were found."
        )

    # STEP 6: Create a corrected COPY of the test set

    original_test = pd.read_csv(
        SPLITS["test"],
        low_memory=False
    )

    test_keys = row_keys(original_test)

    missing_test = (
        test_keys_to_quarantine - set(test_keys)
    )

    if missing_test:
        raise ValueError(
            f"Expected test emails missing: {missing_test}"
        )

    keep = [
        key not in test_keys_to_quarantine
        for key in test_keys
    ]

    corrected_test = original_test.loc[keep].copy()

    expected_count = (
        len(original_test)
        - len(test_keys_to_quarantine)
    )

    if len(corrected_test) != expected_count:
        raise RuntimeError(
            "Corrected test count does not match."
        )

    # STEP 7: Save new files without overwriting originals

    review.to_csv(
        OUT_AUDIT,
        index=False
    )

    corrected_test.to_csv(
        OUT_TEST,
        index=False
    )

    weak_cross = review[
        review["cross_split"]
        & ~review["strict_match"]
    ]

    strong_cross = review[
        review["cross_split"]
        & review["strict_match"]
    ]

    summary = {
        "new_links_reviewed": len(review),
        "links_passing_strict_thresholds": int(
            review["strict_match"].sum()
        ),
        "new_links_crossing_existing_splits": int(
            review["cross_split"].sum()
        ),
        "strict_links_crossing_splits": len(strong_cross),
        "weaker_links_crossing_splits_still_to_review": len(
            weak_cross
        ),
        "reserved_test_before": len(original_test),
        "reserved_test_records_quarantined": len(
            test_keys_to_quarantine
        ),
        "reserved_test_after": len(corrected_test),
        "train_or_validation_records_changed": 0,
        "existing_model_results_changed": 0
    }

    pd.DataFrame(
        summary.items(),
        columns=["metric", "value"]
    ).to_csv(
        OUT_SUMMARY,
        index=False
    )

    # STEP 8: Print results

    print("\n===== SCRIPT 22C: LINK AUDIT =====")

    print(
        review[[
            "link_number",
            "source_a",
            "source_b",
            "split_a",
            "split_b",
            "strict_match",
            "proposed_action"
        ]].to_string(index=False)
    )

    print("\n===== SUMMARY =====")

    for label, value in summary.items():
        print(f"{label}: {value:,}")

    print("\n===== FILES SAVED =====")
    print("results/22c_new_link_review.csv")
    print("results/22c_resolution_summary.csv")
    print("data/controlled_test_leakage_checked.csv")

    print("\nOriginal datasets were NOT modified.")
    print("Existing models were NOT retrained.")
    print("The final test was NOT evaluated.")

    if len(weak_cross):
        print(
            "\nNOTE: Some weaker template matches "
            "still cross splits. They require a "
            "documented decision or sensitivity analysis."
        )


if __name__ == "__main__":
    main()
