
from bisect import bisect_left
from collections import defaultdict, Counter
from functools import lru_cache
from pathlib import Path
import csv
import re
import zlib

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio


# STEP 1: Project paths
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
RESULTS.mkdir(exist_ok=True)

INPUT = DATA / "spam_near_duplicate_grouped.csv"

SPLIT_FILES = {
    "train": DATA / "controlled_train.csv",
    "validation": DATA / "controlled_validation.csv",
    "test": DATA / "controlled_test.csv",
}

OUTPUT_LINKS = RESULTS / "22b_new_cross_family_links.csv"
OUTPUT_MAP = RESULTS / "22b_proposed_family_mapping.csv"
OUTPUT_SUMMARY = RESULTS / "22b_search_summary.csv"


# STEP 2: Match Script 16's original settings
SEED = 42
NUM_HASHES = 64
BANDS = 8
MIN_WORDS = 12
MIN_LENGTH_RATIO = 0.85
ORIGINAL_LIMIT = 300

MIN_JACCARD = 0.78
MIN_CHAR_SIMILARITY = 92.0

# Stricter requirements for previously skipped short emails
SHORT_MIN_JACCARD = 0.90
SHORT_MIN_CHAR_SIMILARITY = 97.0


def normalize(value):
    """Normalize capitalization and whitespace."""
    return " ".join(str(value).casefold().split())


def word_trigrams(text):
    """Create three-word combinations for similarity checking."""
    words = re.findall(r"\w+", text, flags=re.UNICODE)

    return {
        zlib.crc32(
            " ".join(words[k:k + 3]).encode("utf-8")
        )
        for k in range(len(words) - 2)
    }


@lru_cache(maxsize=4000)
def get_shingles(i):
    """Cache word trigrams to avoid repeated processing."""
    return word_trigrams(TEXTS[i])


class UnionFind:
    """Track connections between duplicate families."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def main():

    # STEP 3: Load Script 16 results
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing Script 16 output: {INPUT}"
        )

    for path in SPLIT_FILES.values():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Script 19 output: {path}"
            )

    print("Loading Script 16 groups and Script 19 splits...")

    columns = [
        "source_id",
        "source_row",
        "duplicate_group_id",
        "near_duplicate_group_id",
        "project_label",
        "model_text",
    ]

    emails = pd.read_csv(
        INPUT,
        usecols=columns,
        low_memory=False
    )

    if emails[columns].isna().any().any():
        raise ValueError("Missing required values.")

    if emails.duplicated(["source_id", "source_row"]).any():
        raise ValueError("Repeated email identifiers found.")

    # STEP 4: Read existing train/validation/test assignments
    assignments = {}

    for split, path in SPLIT_FILES.items():

        part = pd.read_csv(
            path,
            usecols=["source_id", "source_row"]
        )

        for source, row in part.itertuples(
            index=False,
            name=None
        ):

            key = (str(source), int(row))

            if key in assignments:
                raise ValueError(
                    f"Record appears in multiple splits: {key}"
                )

            assignments[key] = split

    if len(assignments) != len(emails):
        raise ValueError(
            "Existing splits do not cover the entire dataset."
        )

    # STEP 5: Prepare one representative per exact group
    reps = emails.drop_duplicates(
        "duplicate_group_id"
    ).reset_index(drop=True)

    global TEXTS

    TEXTS = [
        normalize(s)
        for s in reps["model_text"].tolist()
    ]

    sizes = np.array(
        [len(s) for s in TEXTS],
        dtype=np.int64
    )

    is_short = np.array([
        len(re.findall(r"\w+", text, flags=re.UNICODE))
        < MIN_WORDS
        for text in TEXTS
    ], dtype=bool)

    family_ids = (
        reps["near_duplicate_group_id"]
        .astype(str)
        .tolist()
    )

    sources = reps["source_id"].astype(str).tolist()
    rows = reps["source_row"].astype(int).tolist()
    labels = reps["project_label"].astype(int).tolist()

    split_names = []

    for source, row in zip(sources, rows):
        key = (source, row)

        if key not in assignments:
            raise ValueError(
                f"Representative missing from splits: {key}"
            )

        split_names.append(assignments[key])

    unique_families, family_codes = np.unique(
        family_ids,
        return_inverse=True
    )

    union = UnionFind(len(unique_families))
    counts = Counter()

    # STEP 6: Rebuild Script 16's MinHash candidate index
    print(
        f"Rebuilding candidate index for "
        f"{len(reps):,} representatives..."
    )

    rng = np.random.default_rng(SEED)

    prime = np.uint64(4294967291)

    a = rng.integers(
        1, int(prime) - 1,
        size=NUM_HASHES,
        dtype=np.uint64
    )

    b = rng.integers(
        0, int(prime) - 1,
        size=NUM_HASHES,
        dtype=np.uint64
    )

    buckets = defaultdict(list)
    rep_keys = []

    band_size = NUM_HASHES // BANDS

    for i in range(len(reps)):

        if is_short[i]:
            rep_keys.append([])
            continue

        shingles = get_shingles(i)

        values = np.fromiter(
            shingles,
            dtype=np.uint64
        )

        sig = (
            (a[:, None] * values[None, :] + b[:, None])
            % prime
        ).min(axis=1)

        keys = [
            (
                band,
                *map(
                    int,
                    sig[
                        band * band_size:
                        (band + 1) * band_size
                    ]
                )
            )
            for band in range(BANDS)
        ]

        rep_keys.append(keys)

        for key in keys:
            buckets[key].append(i)

        if (i + 1) % 10000 == 0:
            print(
                f"Indexed {i + 1:,} / {len(reps):,}"
            )

    # STEP 7: Check newly discovered pairs
    def check_pair(i, j, method, writer):

        # Already in the same known duplicate family
        if family_codes[i] == family_codes[j]:
            counts["already_same_family_skipped"] += 1
            return

        small, large = sorted(
            (sizes[i], sizes[j])
        )

        if large == 0:
            return

        if small / large < MIN_LENGTH_RATIO:
            return

        counts["additional_pairs_evaluated"] += 1

        if method == "previously_truncated_LSH":

            # Original Script 16 similarity requirements
            sh_i = get_shingles(i)
            sh_j = get_shingles(j)

            jac = len(sh_i & sh_j) / len(sh_i | sh_j)

            if jac < MIN_JACCARD:
                return

            similarity = ratio(
                TEXTS[i],
                TEXTS[j]
            )

            if similarity < MIN_CHAR_SIMILARITY:
                return

        else:

            # Stricter checking for short emails
            similarity = ratio(
                TEXTS[i],
                TEXTS[j]
            )

            if similarity < SHORT_MIN_CHAR_SIMILARITY:
                return

            sh_i = get_shingles(i)
            sh_j = get_shingles(j)

            if not sh_i or not sh_j:
                counts["too_short_for_trigram_check"] += 1
                return

            jac = len(sh_i & sh_j) / len(sh_i | sh_j)

            if jac < SHORT_MIN_JACCARD:
                return

        # STEP 8: Record newly discovered connections
        counts["new_cross_family_matches"] += 1

        if split_names[i] != split_names[j]:
            counts["matches_crossing_existing_splits"] += 1

        if sources[i] != sources[j]:
            counts["matches_crossing_sources"] += 1

        if labels[i] != labels[j]:
            counts["matches_with_conflicting_labels"] += 1

        union.join(
            int(family_codes[i]),
            int(family_codes[j])
        )

        writer.writerow({
            "source_a": sources[i],
            "row_a": rows[i],
            "source_b": sources[j],
            "row_b": rows[j],
            "old_family_a": family_ids[i],
            "old_family_b": family_ids[j],
            "split_a": split_names[i],
            "split_b": split_names[j],
            "label_conflict": labels[i] != labels[j],
            "jaccard": round(jac, 4),
            "char_similarity": round(similarity, 2),
            "method": method
        })

    # STEP 9: Save newly discovered links
    with OUTPUT_LINKS.open("w", newline="") as stream:

        fields = [
            "source_a", "row_a",
            "source_b", "row_b",
            "old_family_a", "old_family_b",
            "split_a", "split_b",
            "label_conflict",
            "jaccard", "char_similarity",
            "method"
        ]

        writer = csv.DictWriter(
            stream,
            fieldnames=fields
        )

        writer.writeheader()

        # STEP 10: Complete previously truncated candidate checks
        print(
            "Checking all previously truncated "
            "LSH candidates..."
        )

        for i, keys in enumerate(rep_keys):

            if not keys:
                continue

            possible = set()

            for key in keys:

                members = buckets[key]

                possible.update(
                    members[:bisect_left(members, i)]
                )

            if len(possible) <= ORIGINAL_LIMIT:
                continue

            counts["truncated_searches_replayed"] += 1

            remaining = sorted(possible)[ORIGINAL_LIMIT:]

            counts["previously_omitted_candidates"] += (
                len(remaining)
            )

            for j in remaining:
                check_pair(
                    i, j,
                    "previously_truncated_LSH",
                    writer
                )

            if (
                counts["truncated_searches_replayed"] % 500
                == 0
            ):
                print(
                    "Replayed "
                    f"{counts['truncated_searches_replayed']:,} "
                    "truncated searches"
                )

        # STEP 11: Check previously skipped short emails
        print(
            "Checking short emails against "
            "comparable-length emails..."
        )

        sorted_ids = np.argsort(
            sizes,
            kind="stable"
        )

        sorted_lengths = sizes[sorted_ids]

        short_ids = np.flatnonzero(is_short)

        counts["short_representatives_rechecked"] = (
            len(short_ids)
        )

        for number, i in enumerate(short_ids, start=1):

            low = np.searchsorted(
                sorted_lengths,
                int(np.ceil(sizes[i] * MIN_LENGTH_RATIO)),
                side="left"
            )

            high = np.searchsorted(
                sorted_lengths,
                int(np.floor(sizes[i] / MIN_LENGTH_RATIO)),
                side="right"
            )

            for j in sorted_ids[low:high]:

                j = int(j)

                if j == i:
                    continue

                # Avoid checking short-short pairs twice
                if is_short[j] and j < i:
                    continue

                counts["short_length_filtered_candidates"] += 1

                check_pair(
                    int(i),
                    j,
                    "short_text_check",
                    writer
                )

            if number % 200 == 0:
                print(
                    f"Rechecked {number:,} / "
                    f"{len(short_ids):,} short representatives"
                )

    # STEP 12: Create proposed updated family mapping
    # This does not overwrite existing group IDs.
    proposed_ids = [
        f"ND22B_{union.find(i):06d}"
        for i in range(len(unique_families))
    ]

    mapping = pd.DataFrame({
        "old_near_duplicate_group_id": unique_families,
        "proposed_near_duplicate_group_id": proposed_ids
    })

    mapping.to_csv(
        OUTPUT_MAP,
        index=False
    )

    # STEP 13: Check whether proposed families cross splits
    proposed_split_members = defaultdict(set)

    for i in range(len(reps)):

        root = union.find(
            int(family_codes[i])
        )

        proposed_split_members[root].add(
            split_names[i]
        )

    counts["old_families"] = len(unique_families)

    counts["proposed_families"] = len(
        set(proposed_ids)
    )

    counts["proposed_families_spanning_splits"] = sum(
        len(s) > 1
        for s in proposed_split_members.values()
    )

    # Remaining detection limitations
    counts["remaining_LSH_candidate_generation_limit"] = (
        "MinHash LSH may miss pairs that share no bucket"
    )

    counts["short_messages_under_3_words_limit"] = (
        "Cannot confirm word-trigram similarity "
        "for fewer than 3 words"
    )

    # STEP 14: Save the audit summary
    pd.DataFrame(
        counts.items(),
        columns=["metric", "value"]
    ).to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    print("\n===== SCRIPT 22B RESULTS =====")

    for name, value in counts.items():

        if isinstance(value, int):
            print(f"{name}: {value:,}")
        else:
            print(f"{name}: {value}")

    print("\n===== FILES SAVED =====")

    print("results/22b_new_cross_family_links.csv")
    print("results/22b_proposed_family_mapping.csv")
    print("results/22b_search_summary.csv")

    print(
        "\nExisting data, splits, and model results "
        "were NOT modified."
    )

    if counts["proposed_families_spanning_splits"]:

        print(
            "\nWARNING: Newly connected families "
            "cross existing split boundaries."
        )

        print(
            "Current model evaluations should not "
            "be treated as fully leakage-controlled."
        )

    else:

        print(
            "\nNo split-crossing links found "
            "by these additional checks."
        )

    print(
        "This remains an approximate search. "
        "Complete near-duplicate recall is not guaranteed."
    )


if __name__ == "__main__":
    main()
