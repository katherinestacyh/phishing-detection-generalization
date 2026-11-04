from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    auc
)

# Project folders
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

INPUT = DATA / "loso_predictions_PRIVATE.csv"

OUTPUT = RESULTS / "loso_confidence_intervals.csv"
REPORT = RESULTS / "loso_confidence_interval_report_table.csv"
CHECKS = RESULTS / "loso_bootstrap_checks.csv"

SEED = 42
N_BOOTSTRAPS = 300
CONFIDENCE_LEVEL = 95

SOURCES = [
    "CEAS_08",
    "Enron",
    "Ling",
    "SpamAssasin"
]

MODELS = [
    "Logistic Regression",
    "Linear SVM"
]

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "average_precision",
    "pr_auc_trapezoid",
    "false_positive_rate",
    "false_negative_rate"
]


def calculate_metrics(y, pred, scores, weights):
    """Calculate metrics using bootstrap sample weights."""

    positive = y == 1
    negative = y == 0

    # Count weighted classification outcomes.
    tp = weights[(positive) & (pred == 1)].sum()
    tn = weights[(negative) & (pred == 0)].sum()
    fp = weights[(negative) & (pred == 1)].sum()
    fn = weights[(positive) & (pred == 0)].sum()

    # A bootstrap sample must contain both classes.
    if tp + fn == 0 or tn + fp == 0:
        return None

    total = tp + tn + fp + fn

    precision = (
        tp / (tp + fp) if tp + fp else 0.0
    )

    recall = tp / (tp + fn)

    f1 = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn else 0.0
    )

    # Average precision uses prediction scores,
    # not just the predicted 0/1 labels.
    ap = average_precision_score(
        y, scores, sample_weight=weights
    )

    curve_precision, curve_recall, _ = (
        precision_recall_curve(
            y, scores, sample_weight=weights
        )
    )

    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": ap,
        "pr_auc_trapezoid": auc(
            curve_recall, curve_precision
        ),
        "false_positive_rate": fp / (fp + tn),
        "false_negative_rate": fn / (fn + tp)
    }


def main():

    # STEP 1: Load Script 21 predictions.
    if not INPUT.exists():
        raise FileNotFoundError(
            "Missing LOSO predictions. Run Script 21 first."
        )

    print("Loading leave-one-source-out predictions...")

    required = [
        "source_id",
        "source_row",
        "project_label",
        "leakage_group_id",
        "model",
        "predicted_label",
        "score"
    ]

    df = pd.read_csv(
        INPUT,
        usecols=required,
        low_memory=False
    )

    if df[required].isna().any().any():
        raise ValueError("Missing prediction data found.")

    if not df["project_label"].isin([0, 1]).all():
        raise ValueError("Invalid true labels.")

    if not df["predicted_label"].isin([0, 1]).all():
        raise ValueError("Invalid predicted labels.")

    if not np.isfinite(df["score"]).all():
        raise ValueError("Invalid prediction scores.")

    if df.duplicated(
        ["model", "source_id", "source_row"]
    ).any():
        raise ValueError("Duplicate prediction records found.")

    print(f"Loaded {len(df):,} prediction records.")

    interval_rows = []
    report_rows = []
    check_rows = []

    # STEP 2: Evaluate each held-out source and model.
    for source_index, source in enumerate(SOURCES):

        for model_name in MODELS:

            subset = df[
                (df["source_id"] == source)
                & (df["model"] == model_name)
            ].copy()

            if subset.empty:
                raise ValueError(
                    f"No records for {model_name}, {source}"
                )

            y = subset["project_label"].to_numpy()
            pred = subset["predicted_label"].to_numpy()
            scores = subset["score"].to_numpy()

            # Give each detected duplicate family a number.
            group_codes, unique_groups = pd.factorize(
                subset["leakage_group_id"],
                sort=True
            )

            n_groups = len(unique_groups)
            n_emails = len(subset)

            print(
                f"\n===== {source}: {model_name} ====="
            )

            print(f"Emails: {n_emails:,}")
            print(f"Duplicate families: {n_groups:,}")

            # STEP 3: Calculate original observed metrics.
            original_weights = np.ones(n_emails)

            observed = calculate_metrics(
                y, pred, scores, original_weights
            )

            if observed is None:
                raise ValueError(
                    "Both classes are required for evaluation."
                )

            # STEP 4: Bootstrap duplicate families.
            # Use the same random seed for both models
            # within a source to support later comparisons.
            rng = np.random.default_rng(
                SEED + source_index
            )

            bootstrap_results = {
                metric: [] for metric in METRICS
            }

            skipped = 0

            for iteration in range(N_BOOTSTRAPS):

                # Randomly select families with replacement.
                sampled_groups = rng.integers(
                    0, n_groups, size=n_groups
                )

                # Count how many times each family is selected.
                family_counts = np.bincount(
                    sampled_groups,
                    minlength=n_groups
                )

                # All emails in a selected family receive
                # the same bootstrap multiplicity.
                weights = family_counts[group_codes]

                result = calculate_metrics(
                    y, pred, scores, weights
                )

                if result is None:
                    skipped += 1
                    continue

                for metric in METRICS:
                    bootstrap_results[metric].append(
                        result[metric]
                    )

            valid = N_BOOTSTRAPS - skipped

            if valid < N_BOOTSTRAPS * 0.8:
                raise ValueError(
                    "Too many invalid bootstrap samples."
                )

            # STEP 5: Calculate confidence intervals.
            alpha = (100 - CONFIDENCE_LEVEL) / 2

            row_for_report = {
                "held_out_source": source,
                "model": model_name,
                "emails": n_emails,
                "duplicate_families": n_groups
            }

            for metric in METRICS:

                values = np.array(
                    bootstrap_results[metric]
                )

                lower = np.percentile(values, alpha)
                upper = np.percentile(
                    values, 100 - alpha
                )

                interval_rows.append({
                    "held_out_source": source,
                    "model": model_name,
                    "metric": metric,
                    "estimate": observed[metric],
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": CONFIDENCE_LEVEL,
                    "valid_bootstrap_samples": valid,
                    "emails": n_emails,
                    "duplicate_families": n_groups
                })

                # Short version for the project report.
                row_for_report[metric] = (
                    f"{observed[metric]:.4f} "
                    f"[{lower:.4f}, {upper:.4f}]"
                )

            report_rows.append(row_for_report)

            check_rows.append({
                "held_out_source": source,
                "model": model_name,
                "emails": n_emails,
                "duplicate_families": n_groups,
                "requested_bootstraps": N_BOOTSTRAPS,
                "valid_bootstraps": valid,
                "skipped_bootstraps": skipped,
                "resampling_unit": "leakage_group_id"
            })

            f1_values = bootstrap_results["f1"]

            print(
                f"F1: {observed['f1']:.4f} "
                f"[{np.percentile(f1_values, 2.5):.4f}, "
                f"{np.percentile(f1_values, 97.5):.4f}]"
            )

            print(
                f"Valid bootstrap samples: {valid}"
            )

    # STEP 6: Save output files.
    pd.DataFrame(interval_rows).to_csv(
        OUTPUT, index=False
    )

    pd.DataFrame(report_rows).to_csv(
        REPORT, index=False
    )

    pd.DataFrame(check_rows).to_csv(
        CHECKS, index=False
    )

    # STEP 7: Print final summary.
    print("\n===== LOSO 95% CONFIDENCE INTERVALS =====")

    summary = pd.DataFrame(interval_rows)

    f1_summary = summary[
        summary["metric"] == "f1"
    ][[
        "held_out_source",
        "model",
        "estimate",
        "ci_lower",
        "ci_upper"
    ]]

    print(
        f1_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n===== FILES SAVED =====")
    print("results/loso_confidence_intervals.csv")
    print(
        "results/loso_confidence_interval_report_table.csv"
    )
    print("results/loso_bootstrap_checks.csv")

    print("\nNo models were retrained.")
    print("Reserved final test data was not used.")
    print(
        "Intervals estimate uncertainty within each "
        "held-out source, not across entirely new sources."
    )


if __name__ == "__main__":
    main()
