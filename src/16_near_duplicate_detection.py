from collections import defaultdict
from pathlib import Path
import re
import zlib

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Record these choices in the Deliverable 3 report.
SEED = 42
NUM_HASHES = 64
BANDS = 8
ROWS_PER_BAND = NUM_HASHES // BANDS
MIN_WORDS = 12  # Very short emails remain in their existing exact-duplicate groups.
MIN_LENGTH_RATIO = 0.85
MIN_JACCARD = 0.78
MIN_CHAR_SIMILARITY = 92.0  # RapidFuzz ratio ranges from 0 to 100.
MAX_CANDIDATES_PER_EMAIL = 300  # Safety limit; any truncation is reported.


def normalize(text):
    return " ".join(str(text).casefold().split())


def shingles(text):
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    if len(words) < MIN_WORDS:
        return set()
    return {
        zlib.crc32(" ".join(words[i:i + 3]).encode("utf-8"))
        for i in range(len(words) - 2)
    }


# Reproducible MinHash parameters (no all-pairs comparisons).
rng = np.random.default_rng(SEED)
prime = np.uint64(4294967291)
a = rng.integers(1, int(prime) - 1, size=NUM_HASHES, dtype=np.uint64)
b = rng.integers(0, int(prime) - 1, size=NUM_HASHES, dtype=np.uint64)


def signature(shingle_hashes):
    values = np.fromiter(shingle_hashes, dtype=np.uint64)
    hashed = (a[:, None] * values[None, :] + b[:, None]) % prime
    return hashed.min(axis=1)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, x, y):
        a_root, b_root = self.find(x), self.find(y)
        if a_root != b_root:
            self.parent[max(a_root, b_root)] = min(a_root, b_root)


def main():
    input_path = DATA_DIR / "spam_cleaned.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Run script 13 first.")

    df = pd.read_csv(input_path, low_memory=False)
    required = {"duplicate_group_id", "model_text", "source_id", "source_row", "project_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Required columns contain missing values; inspect the data first.")

    original_groups = df["duplicate_group_id"].nunique()
    # One representative per existing normalized-exact duplicate group.
    reps = df.drop_duplicates("duplicate_group_id").reset_index(drop=True)
    reps["near_text"] = reps["model_text"].map(normalize)
    texts = reps["near_text"].tolist()
    uf = UnionFind(len(reps))

    index = defaultdict(list)
    accepted = []
    candidate_checks = 0
    truncated_searches = 0
    short_emails_skipped = 0

    print(f"Loaded {len(df):,} emails and {len(reps):,} existing exact groups.")
    print("Searching for likely near-duplicates (approximate candidate search)...")

    for i, text in enumerate(texts):
        sh = shingles(text)
        if not sh:
            short_emails_skipped += 1
            continue
        sig = signature(sh)
        candidate_ids = set()
        for band in range(BANDS):
            start = band * ROWS_PER_BAND
            key = (band, *map(int, sig[start:start + ROWS_PER_BAND]))
            candidate_ids.update(index.get(key, []))

        if len(candidate_ids) > MAX_CANDIDATES_PER_EMAIL:
            truncated_searches += 1
            # Prefer earlier representatives so the search is reproducible.
            candidate_ids = set(sorted(candidate_ids)[:MAX_CANDIDATES_PER_EMAIL])

        for j in sorted(candidate_ids):
            other = texts[j]
            length_ratio = min(len(text), len(other)) / max(len(text), len(other))
            if length_ratio < MIN_LENGTH_RATIO:
                continue
            candidate_checks += 1
            other_sh = shingles(other)
            jaccard = len(sh & other_sh) / len(sh | other_sh)
            if jaccard < MIN_JACCARD:
                continue
            char_similarity = ratio(text, other)
            if char_similarity < MIN_CHAR_SIMILARITY:
                continue

            uf.join(i, j)
            accepted.append({
                "source_a": reps.at[j, "source_id"],
                "row_a": reps.at[j, "source_row"],
                "label_a": reps.at[j, "project_label"],
                "source_b": reps.at[i, "source_id"],
                "row_b": reps.at[i, "source_row"],
                "label_b": reps.at[i, "project_label"],
                "jaccard": round(jaccard, 4),
                "char_similarity": round(char_similarity, 2),
                "label_conflict": reps.at[j, "project_label"] != reps.at[i, "project_label"],
            })

        for band in range(BANDS):
            start = band * ROWS_PER_BAND
            key = (band, *map(int, sig[start:start + ROWS_PER_BAND]))
            index[key].append(i)

        if (i + 1) % 10000 == 0:
            print(f"Checked {i + 1:,} / {len(reps):,} unique emails")

    # Map each original exact-duplicate group into a broader leakage-control family.
    group_to_rep = dict(zip(reps["duplicate_group_id"], reps.index))
    df["near_duplicate_group_id"] = df["duplicate_group_id"].map(
        lambda g: f"ND_{uf.find(group_to_rep[g]):06d}"
    )

    family_info = df.groupby("near_duplicate_group_id").agg(
        records=("project_label", "size"),
        labels=("project_label", "nunique"),
        sources=("source_id", "nunique"),
    )
    conflicted_families = set(family_info.index[family_info["labels"] > 1])
    df["near_duplicate_label_conflict"] = df["near_duplicate_group_id"].isin(conflicted_families)

    # Dataset contains original email text: save privately inside gitignored data/.
    df.to_csv(DATA_DIR / "spam_near_duplicate_grouped.csv", index=False)
    pd.DataFrame(accepted, columns=[
        "source_a", "row_a", "label_a", "source_b", "row_b", "label_b",
        "jaccard", "char_similarity", "label_conflict"
    ]).to_csv(RESULTS_DIR / "near_duplicate_pairs.csv", index=False)

    summary = {
        "total_emails": len(df),
        "original_exact_duplicate_groups": original_groups,
        "new_combined_groups": df["near_duplicate_group_id"].nunique(),
        "confirmed_near_duplicate_links": len(accepted),
        "groups_with_multiple_emails": int((family_info["records"] > 1).sum()),
        "groups_with_label_conflicts": len(conflicted_families),
        "cross_source_groups": int((family_info["sources"] > 1).sum()),
        "short_representatives_skipped": short_emails_skipped,
        "candidates_checked": candidate_checks,
        "searches_truncated_by_safety_limit": truncated_searches,
    }
    pd.DataFrame(summary.items(), columns=["metric", "count"]).to_csv(
        RESULTS_DIR / "near_duplicate_summary.csv", index=False
    )

    print("\n===== NEAR-DUPLICATE RESULTS =====")
    for name, value in summary.items():
        print(f"{name}: {value:,}")
    print("\nSaved data/spam_near_duplicate_grouped.csv")
    print("Saved results/near_duplicate_pairs.csv")
    print("Saved results/near_duplicate_summary.csv")
    if conflicted_families:
        print("WARNING: Some grouped emails have conflicting labels. Review before splitting.")
    if truncated_searches:
        print("WARNING: Some candidate searches were truncated. Results are not exhaustive.")
    print("Note: Approximate search can miss some near-duplicates; review examples and thresholds.")


if __name__ == "__main__":
    main()
