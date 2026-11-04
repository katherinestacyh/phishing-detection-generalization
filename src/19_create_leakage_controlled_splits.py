from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

INPUT = DATA / "spam_near_duplicate_refined.csv"

# Desired split: 70% training, 15% validation, 15% testing
TARGET = np.array([0.70, 0.15, 0.15])

SEED = 42
CANDIDATE_SEEDS = 30
SPLITS = ("train", "validation", "test")


def make_candidate(df, seed):
    """Create a split without separating duplicate families."""

    groups = df["leakage_group_id"]

    # First separate approximately 15% of groups for testing.
    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=seed
    )

    remaining_idx, test_idx = next(
        outer.split(df, groups=groups)
    )

    # Split the remaining groups into training and validation.
    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15 / 0.85,
        random_state=seed + 1000
    )

    train_rel, val_rel = next(
        inner.split(
            df.iloc[remaining_idx],
            groups=groups.iloc[remaining_idx]
        )
    )

    # Assign numeric codes for each split.
    # 0 = training, 1 = validation, 2 = testing
    codes = np.full(len(df), 2, dtype=np.int8)

    codes[remaining_idx[train_rel]] = 0
    codes[remaining_idx[val_rel]] = 1

    return codes


def balance_score(codes, strata):
    """Measure how balanced the split is across labels and sources."""

    global_ratio = (
        np.bincount(codes, minlength=3) / len(codes)
    )

    score = 2 * np.abs(global_ratio - TARGET).sum()

    # Check the distribution within each source and label.
    score += np.mean([
        np.abs(
            np.bincount(codes[idx], minlength=3)
            / len(idx) - TARGET
        ).sum()
        for idx in strata
    ])

    return float(score)


def main():

    # STEP 1: Load our refined dataset.
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run Script 18 first."
        )

    print("Loading refined email dataset...")

    df = pd.read_csv(INPUT, low_memory=False)

    required_columns = [
        "source_id",
        "source_row",
        "project_label",
        "duplicate_group_id",
        "refined_group_id",
        "leakage_group_id",
        "exclude_from_modeling"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Missing identifiers or class labels found."
        )

    if not df["project_label"].isin([0, 1]).all():
        raise ValueError(
            "Expected labels: 0 = legitimate, 1 = spam."
        )

    if df.duplicated(["source_id", "source_row"]).any():
        raise ValueError(
            "Duplicate source identifiers detected."
        )

    # STEP 2: Prepare source and label information.
    # We are NOT removing the flagged emails.
    strata = list(
        df.groupby(
            ["source_id", "project_label"]
        ).indices.values()
    )

    # Find groups containing both legitimate and spam labels.
    label_counts = (
        df.groupby("leakage_group_id")[
            "project_label"
        ].nunique()
    )

    mixed_groups = label_counts[
        label_counts > 1
    ].index

    mixed_mask = (
        df["leakage_group_id"]
        .isin(mixed_groups)
        .to_numpy()
    )

    print(
        f"Loaded {len(df):,} emails across "
        f"{df['leakage_group_id'].nunique():,} families."
    )

    # STEP 3: Create several possible splits.
    # Choose the one with better source/class balance.
    print(
        "Selecting a reproducible group split "
        "with balanced labels and sources..."
    )

    best_score = float("inf")
    best_codes = None
    best_seed = None

    for seed in range(SEED, SEED + CANDIDATE_SEEDS):

        candidate = make_candidate(df, seed)

        # Keep conflicting-label families in training
        # so they are not treated as validation/test truth.
        candidate[mixed_mask] = 0

        score = balance_score(candidate, strata)

        if score < best_score:
            best_score = score
            best_codes = candidate
            best_seed = seed

    df["split"] = pd.Categorical.from_codes(
        best_codes,
        categories=SPLITS
    )

    # STEP 4: Check for leakage between splits.
    overlap_checks = {
        col: int(
            (
                df.groupby(col, observed=True)["split"]
                .nunique() > 1
            ).sum()
        )
        for col in (
            "duplicate_group_id",
            "refined_group_id",
            "leakage_group_id"
        )
    }

    # Stop if any detected duplicate group crosses splits.
    if any(overlap_checks.values()):
        raise RuntimeError(
            f"Duplicate-family leakage found: {overlap_checks}"
        )

    # Check that every email belongs to exactly one split.
    if len(df) != sum(
        (df["split"] == s).sum() for s in SPLITS
    ):
        raise RuntimeError(
            "Some records were not assigned correctly."
        )

    if not all(
        (df["split"] == s).any() for s in SPLITS
    ):
        raise RuntimeError("A split is empty.")

    # Keep conflicting-label families out of evaluation.
    if (
        df.loc[mixed_mask, "split"] != "train"
    ).any():
        raise RuntimeError(
            "Conflicting-label family entered validation/test."
        )

    # STEP 5: Save new datasets.
    # These contain email text, so keep them private.
    for name in SPLITS:

        part = df.loc[df["split"] == name].copy()

        output = DATA / f"controlled_{name}.csv"

        part.to_csv(output, index=False)

        print(
            f"Saved {output.relative_to(PROJECT)}: "
            f"{len(part):,} emails"
        )

    # STEP 6: Summarize the results.
    summary_rows = []

    for name, proportion in zip(SPLITS, TARGET):

        part = df[df["split"] == name]

        summary_rows.append({
            "split": name,
            "target_fraction": proportion,
            "actual_fraction": round(
                len(part) / len(df), 5
            ),
            "emails": len(part),
            "legitimate_0": int(
                (part["project_label"] == 0).sum()
            ),
            "spam_1": int(
                (part["project_label"] == 1).sum()
            ),
            "leakage_families": (
                part["leakage_group_id"].nunique()
            ),
            "flagged_emails_retained": int(
                part["exclude_from_modeling"]
                .astype(str)
                .str.lower()
                .eq("true")
                .sum()
            )
        })

    pd.DataFrame(summary_rows).to_csv(
        RESULTS / "controlled_split_summary.csv",
        index=False
    )

    # STEP 7: Check distribution by email source and label.
    source_counts = (
        df.groupby(
            ["source_id", "project_label", "split"],
            observed=True
        )
        .size()
        .unstack("split", fill_value=0)
        .reindex(columns=SPLITS, fill_value=0)
        .reset_index()
    )

    source_counts.to_csv(
        RESULTS / "controlled_source_label_counts.csv",
        index=False
    )

    # STEP 8: Save leakage verification results.
    checks = {
        "input_emails": len(df),
        "output_emails": sum(
            row["emails"] for row in summary_rows
        ),
        "mixed_label_families_training_only": len(
            mixed_groups
        ),
        "mixed_label_emails_training_only": int(
            mixed_mask.sum()
        ),
        "exact_group_overlap_across_splits": (
            overlap_checks["duplicate_group_id"]
        ),
        "refined_group_overlap_across_splits": (
            overlap_checks["refined_group_id"]
        ),
        "broad_leakage_group_overlap_across_splits": (
            overlap_checks["leakage_group_id"]
        ),
        "chosen_random_seed": best_seed,
        "candidate_seeds_compared": CANDIDATE_SEEDS,
        "undetected_near_duplicates_ruled_out": (
            "NO - approximate search was incomplete"
        )
    }

    pd.DataFrame(
        checks.items(),
        columns=["check", "value"]
    ).to_csv(
        RESULTS / "controlled_split_checks.csv",
        index=False
    )

    # STEP 9: Display the final results.
    print("\n===== CONTROLLED SPLIT SUMMARY =====")

    print(
        pd.DataFrame(summary_rows).to_string(index=False)
    )

    print("\n===== GROUP LEAKAGE CHECKS =====")

    for name, value in overlap_checks.items():
        print(f"{name} across splits: {value}")

    print(
        f"Mixed-label groups placed in train: "
        f"{len(mixed_groups)}"
    )

    print(f"Selected seed: {best_seed}")

    print("\nSaved results/controlled_split_summary.csv")
    print("Saved results/controlled_source_label_counts.csv")
    print("Saved results/controlled_split_checks.csv")

    print(
        "\nCAUTION: Group overlap is zero "
        "for DETECTED families only."
    )

    print(
        "Script 16 had truncated searches "
        "and skipped short emails."
    )

    print(
        "Keep data/ files private and reserve "
        "the test split for final evaluation."
    )


if __name__ == "__main__":
    main()
