
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MASTER = DATA / "spam_near_duplicate_master_v2.csv"
TRAIN = DATA / "controlled_train.csv"
TRAIN_CHECKED = DATA / "controlled_train_label_checked.csv"
VALIDATION = DATA / "controlled_validation.csv"
TEST_ORIGINAL = DATA / "controlled_test.csv"
TEST_CHECKED = DATA / "controlled_test_leakage_checked.csv"

OUTPUTS = {
    "train": DATA / "controlled_train_v2.csv",
    "train_checked": DATA / "controlled_train_label_checked_v2.csv",
    "validation": DATA / "controlled_validation_v2.csv",
    "test": DATA / "controlled_test_v2.csv",
}
QUARANTINE = RESULTS / "22e_quarantined_records.csv"
SUMMARY = RESULTS / "22e_split_summary.csv"

KEY = ["source_id", "source_row"]
GROUP = "leakage_group_id"
NEW_GROUP = "leakage_group_id_v2"


def get_keys(frame):
    return set(map(tuple, frame[KEY].itertuples(index=False, name=None)))


def check_unique(frame, name):
    if frame.duplicated(KEY).any():
        raise ValueError(f"Duplicate record IDs in {name}")


def update_groups(frame, lookup, name):
    if GROUP not in frame.columns:
        raise ValueError(f"{name} is missing {GROUP}")
    if NEW_GROUP in frame.columns or "leakage_group_id_original" in frame.columns:
        raise ValueError(f"{name} already contains v2 grouping columns; use original input")
    check_unique(frame, name)

    joined = frame.merge(lookup, on=KEY, how="left", validate="one_to_one")
    if joined[NEW_GROUP].isna().any():
        raise ValueError(f"Some {name} rows are missing from the updated master")
    if not (joined[GROUP].astype(str) == joined["master_old_group"].astype(str)).all():
        raise ValueError(f"Old groups in {name} disagree with the original master")

    joined["leakage_group_id_original"] = joined[GROUP]
    joined[GROUP] = joined[NEW_GROUP]  # downstream scripts use corrected boundaries
    joined = joined.drop(columns=["master_old_group"])
    return joined


def main():
    for path in [MASTER, TRAIN, TRAIN_CHECKED, VALIDATION, TEST_ORIGINAL, TEST_CHECKED]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    print("Loading proposed master grouping and original splits...")
    master = pd.read_csv(
        MASTER,
        usecols=KEY + [GROUP, NEW_GROUP],
        low_memory=False,
    )
    check_unique(master, "master")
    if master[KEY + [GROUP, NEW_GROUP]].isna().any().any():
        raise ValueError("Master identifiers or family IDs contain missing values")

    lookup = master.rename(columns={GROUP: "master_old_group"})
    raw_train = pd.read_csv(TRAIN, low_memory=False)
    raw_train_checked = pd.read_csv(TRAIN_CHECKED, low_memory=False)
    raw_val = pd.read_csv(VALIDATION, low_memory=False)
    raw_test = pd.read_csv(TEST_ORIGINAL, low_memory=False)
    raw_test_checked = pd.read_csv(TEST_CHECKED, low_memory=False)

    for name, frame in [
        ("train", raw_train), ("train_checked", raw_train_checked),
        ("validation", raw_val), ("original test", raw_test),
        ("checked test", raw_test_checked),
    ]:
        check_unique(frame, name)

    original_keys = [get_keys(raw_train), get_keys(raw_val), get_keys(raw_test)]
    if any(original_keys[i] & original_keys[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError("Original train, validation and test have duplicate record IDs")
    if set.union(*original_keys) != get_keys(master):
        raise ValueError("Original splits do not cover exactly the updated master")
    if not get_keys(raw_train_checked).issubset(original_keys[0]):
        raise ValueError("Label-checked train includes non-training records")
    if not get_keys(raw_test_checked).issubset(original_keys[2]):
        raise ValueError("Corrected test contains non-test records")

    train = update_groups(raw_train, lookup, "train")
    train_checked = update_groups(raw_train_checked, lookup, "train_checked")
    val = update_groups(raw_val, lookup, "validation")
    test = update_groups(raw_test_checked, lookup, "checked test")

    # We keep all existing training examples in training. We do NOT add any
    # previously evaluated validation examples to training.
    train_families = set(train[GROUP])
    val_overlap_mask = val[GROUP].isin(train_families)
    val_removed = val.loc[val_overlap_mask].copy()
    val = val.loc[~val_overlap_mask].copy()

    # Script 22c already quarantined the original test record with a strong
    # Enron-SpamAssasin overlap. Verify it actually removes all v2 test overlap.
    val_families = set(val[GROUP])
    test_overlap = set(test[GROUP]) & (train_families | val_families)
    if test_overlap:
        raise ValueError(
            f"Corrected test still shares {len(test_overlap)} updated family IDs "
            "with train/validation. Audit before proceeding."
        )

    if train_families & val_families:
        raise RuntimeError("Updated train and validation still share a family")

    train_label_conflicts = (
        train_checked.groupby(GROUP)["project_label"].nunique()
        if "project_label" in train_checked else pd.Series(dtype=int)
    )
    if (train_label_conflicts > 1).any():
        raise ValueError("Updated training family has conflicting labels; inspect first")

    # Document exclusions, without placing email text into results/.
    missing_test_keys = original_keys[2] - get_keys(raw_test_checked)
    missing_test = raw_test.loc[
        [k in missing_test_keys for k in map(tuple, raw_test[KEY].itertuples(index=False, name=None))]
    ].copy()
    missing_test = update_groups(missing_test, lookup, "quarantined test")

    quarantined_val = val_removed[KEY + ["leakage_group_id_original", GROUP]].copy()
    quarantined_val["original_split"] = "validation"
    quarantined_val["reason"] = "v2_family_already_in_training"

    quarantined_test = missing_test[KEY + ["leakage_group_id_original", GROUP]].copy()
    quarantined_test["original_split"] = "test"
    quarantined_test["reason"] = "removed_in_22c_checked_test"

    quarantine = pd.concat([quarantined_val, quarantined_test], ignore_index=True)

    # Save only after all checks pass.
    train.to_csv(OUTPUTS["train"], index=False)
    train_checked.to_csv(OUTPUTS["train_checked"], index=False)
    val.to_csv(OUTPUTS["validation"], index=False)
    test.to_csv(OUTPUTS["test"], index=False)
    quarantine.to_csv(QUARANTINE, index=False)

    summary = {
        "original_master_emails": len(master),
        "train_v2_rows": len(train),
        "train_label_checked_v2_rows": len(train_checked),
        "validation_original_rows": len(raw_val),
        "validation_v2_rows": len(val),
        "validation_quarantined": len(val_removed),
        "test_original_rows": len(raw_test),
        "test_v2_rows": len(test),
        "test_quarantined": len(missing_test_keys),
        "remaining_cross_split_v2_families": 0,
        "total_quarantined_from_evaluation": len(quarantine),
    }
    pd.DataFrame(summary.items(), columns=["metric", "value"]).to_csv(SUMMARY, index=False)

    print("\n===== SCRIPT 22E RESULTS =====")
    for metric, value in summary.items():
        print(f"{metric}: {value:,}")
    print("\nSaved corrected COPIES in data/: controlled_train_v2.csv, "
          "controlled_train_label_checked_v2.csv, controlled_validation_v2.csv, "
          "controlled_test_v2.csv")
    print("Saved reports in results/: 22e_quarantined_records.csv, 22e_split_summary.csv")
    print("No original dataset, model or prediction file was overwritten.")


if __name__ == "__main__":
    main()
