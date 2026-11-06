from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MASTER = DATA / "spam_near_duplicate_refined.csv"
LINKS = RESULTS / "22b_new_cross_family_links.csv"
SPLITS = {
    "train": DATA / "controlled_train.csv",
    "validation": DATA / "controlled_validation.csv",
    "test": DATA / "controlled_test.csv",
}
CHECKED_TEST = DATA / "controlled_test_leakage_checked.csv"

OUTPUT_MASTER = DATA / "spam_near_duplicate_master_v2.csv"
OUTPUT_AUDIT = RESULTS / "22d_updated_family_split_audit.csv"
OUTPUT_SUMMARY = RESULTS / "22d_updated_master_summary.csv"


def keys(df):
    """Stable email record identifiers, independent of row ordering."""
    return list(zip(df["source_id"].astype(str), df["source_row"].astype(int)))


class UnionFind:
    def __init__(self, families):
        self.parent = {family: family for family in families}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def main():
    inputs = [MASTER, LINKS, CHECKED_TEST, *SPLITS.values()]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")

    print("Loading the refined dataset and Script 22b links...")
    master = pd.read_csv(MASTER, low_memory=False)
    required = {"source_id", "source_row", "leakage_group_id", "project_label"}
    if not required.issubset(master.columns):
        raise ValueError(f"Missing master columns: {required - set(master.columns)}")
    if master[list(required)].isna().any().any():
        raise ValueError("Master identifiers/groups/labels contain missing values")
    if master.duplicated(["source_id", "source_row"]).any():
        raise ValueError("Master has repeated source/row identifiers")

    master["leakage_group_id"] = master["leakage_group_id"].astype(str)
    old_families = set(master["leakage_group_id"])
    lookup = dict(zip(keys(master), master["leakage_group_id"]))

    links = pd.read_csv(LINKS)
    required_links = {
        "source_a", "row_a", "source_b", "row_b",
        "old_family_a", "old_family_b", "jaccard", "char_similarity"
    }
    if not required_links.issubset(links.columns):
        raise ValueError(f"Missing link columns: {required_links - set(links.columns)}")
    if links.empty:
        raise ValueError("No additional links found")

    uf = UnionFind(old_families)
    for link in links.itertuples(index=False):
        a_key = (str(link.source_a), int(link.row_a))
        b_key = (str(link.source_b), int(link.row_b))
        if a_key not in lookup or b_key not in lookup:
            raise ValueError(f"A linked record is not in the master: {a_key}, {b_key}")
        a_family, b_family = lookup[a_key], lookup[b_key]
        if a_family != str(link.old_family_a) or b_family != str(link.old_family_b):
            raise ValueError("Script 22b families do not match this master version")
        if float(link.jaccard) < 0.78 or float(link.char_similarity) < 92:
            raise ValueError("A link does not meet Script 16's broad thresholds")
        # Conservative leakage-boundary policy: group ALL eight links that
        # satisfied the original broad rule, including newsletter templates.
        uf.union(a_family, b_family)

    group_mapping = {family: uf.find(family) for family in old_families}
    master["leakage_group_id_v2"] = master["leakage_group_id"].map(group_mapping)
    new_group_count = master["leakage_group_id_v2"].nunique()
    if new_group_count > len(old_families):
        raise RuntimeError("Unexpected increase in family count")

    # Inspect all original split boundaries without moving any records.
    split_lookup = {}
    for split, path in SPLITS.items():
        split_df = pd.read_csv(path, usecols=["source_id", "source_row"])
        for key in keys(split_df):
            if key in split_lookup:
                raise ValueError(f"Record appears in multiple original splits: {key}")
            split_lookup[key] = split
    master_keys = keys(master)
    if len(split_lookup) != len(master) or set(split_lookup) != set(master_keys):
        raise ValueError("Original split records do not exactly match the master")
    original_split = pd.Series(
        [split_lookup[k] for k in master_keys], index=master.index
    )
    counts = pd.crosstab(master["leakage_group_id_v2"], original_split)
    for split in SPLITS:
        if split not in counts:
            counts[split] = 0
    audit = counts.loc[(counts > 0).sum(axis=1) > 1,
                       ["train", "validation", "test"]].reset_index()
    audit = audit.rename(columns={"leakage_group_id_v2": "proposed_family"})
    audit["total_records"] = audit[["train", "validation", "test"]].sum(axis=1)

    # Verify the test quarantine from Script 22c addresses train/test overlap.
    corrected_test = pd.read_csv(CHECKED_TEST,
                                 usecols=["source_id", "source_row"])
    corrected_test_keys = set(keys(corrected_test))
    original_test_keys = {k for k, v in split_lookup.items() if v == "test"}
    if not corrected_test_keys.issubset(original_test_keys):
        raise ValueError("The checked test set contains non-test records")
    train_families = set(master.loc[original_split == "train", "leakage_group_id_v2"])
    corrected_test_families = set(master.loc[
        [k in corrected_test_keys for k in master_keys], "leakage_group_id_v2"
    ])
    train_test_overlap_after_quarantine = len(train_families & corrected_test_families)

    summary = {
        "original_emails": len(master),
        "updated_master_emails": len(master),
        "additional_links_included": len(links),
        "old_broad_families": len(old_families),
        "updated_broad_families": new_group_count,
        "families_spanning_original_splits": len(audit),
        "train_test_families_after_22c_test_quarantine": train_test_overlap_after_quarantine,
        "test_records_in_22c_checked_copy": len(corrected_test_keys),
        "original_split_rows_changed": 0,
    }

    # Save a proposed master; never overwrite the earlier master or splits.
    master.to_csv(OUTPUT_MASTER, index=False)
    audit.to_csv(OUTPUT_AUDIT, index=False)
    pd.DataFrame(summary.items(), columns=["metric", "value"]).to_csv(
        OUTPUT_SUMMARY, index=False
    )

    print("\n===== SCRIPT 22D SUMMARY =====")
    for name, value in summary.items():
        print(f"{name}: {value:,}")
    if not audit.empty:
        print("\nFamily groups spanning original splits:")
        print(audit.to_string(index=False))
    print("\nSaved:")
    print("data/spam_near_duplicate_master_v2.csv")
    print("results/22d_updated_family_split_audit.csv")
    print("results/22d_updated_master_summary.csv")
    print("\nNo original data, splits, predictions, or models were changed.")
    print("The new leakage_group_id_v2 is PROPOSED until splits are corrected.")


if __name__ == "__main__":
    main()
