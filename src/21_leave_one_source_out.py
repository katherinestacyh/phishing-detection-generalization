from pathlib import Path
import numpy as np
import pandas as pd
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TRAIN_INPUT = DATA / "controlled_train_label_checked.csv"
VALIDATION_INPUT = DATA / "controlled_validation.csv"

SEED = 42

SOURCES = [
    "CEAS_08",
    "Enron",
    "Ling",
    "SpamAssasin"
]


def make_models():
    """Create the same models used in Script 20."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=SEED
        ),

        "Linear SVM": LinearSVC(
            C=1.0,
            dual=False,
            max_iter=5000,
            random_state=SEED
        )
    }


def evaluate(y_true, predictions, scores):
    """Calculate model performance metrics."""

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    curve_precision, curve_recall, _ = (
        precision_recall_curve(y_true, scores)
    )

    return {
        "accuracy": accuracy_score(y_true, predictions),

        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),

        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),

        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),

        "average_precision": average_precision_score(
            y_true, scores
        ),

        "pr_auc_trapezoid": auc(
            curve_recall, curve_precision
        ),

        "false_positive_rate": (
            fp / (fp + tn) if fp + tn else np.nan
        ),

        "false_negative_rate": (
            fn / (fn + tp) if fn + tp else np.nan
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def main():

    # STEP 1: Load training and validation records.
    # Do not use the reserved controlled_test.csv.

    print("Loading controlled datasets...")

    required = [
        "source_id",
        "source_row",
        "project_label",
        "model_text",
        "leakage_group_id"
    ]

    for path in (TRAIN_INPUT, VALIDATION_INPUT):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run Scripts 19 and 19b first."
            )

    df = pd.concat([
        pd.read_csv(
            TRAIN_INPUT,
            usecols=required,
            low_memory=False
        ),
        pd.read_csv(
            VALIDATION_INPUT,
            usecols=required,
            low_memory=False
        )
    ], ignore_index=True)

    # STEP 2: Validate our dataset.

    id_columns = [
        "source_id",
        "source_row",
        "project_label",
        "leakage_group_id"
    ]

    if df[id_columns].isna().any().any():
        raise ValueError("Missing labels or identifiers.")

    if not df["project_label"].isin([0, 1]).all():
        raise ValueError("Expected labels 0 and 1.")

    if df.duplicated(["source_id", "source_row"]).any():
        raise ValueError("Duplicate record identifiers found.")

    if set(df["source_id"].unique()) != set(SOURCES):
        raise ValueError("Unexpected email sources found.")

    df["model_text"] = (
        df["model_text"].fillna("").astype(str)
    )

    print(f"Loaded {len(df):,} emails.")
    print("Reserved test dataset is NOT included.")

    # STEP 3: Check for remaining conflicting-label groups.

    label_counts = (
        df.groupby("leakage_group_id")["project_label"]
        .nunique()
    )

    conflicting_groups = set(
        label_counts.index[label_counts > 1]
    )

    conflicting_mask = (
        df["leakage_group_id"].isin(conflicting_groups)
    )

    print(
        f"Conflicting-label families: "
        f"{len(conflicting_groups)}"
    )

    print(
        f"Ambiguous records omitted: "
        f"{int(conflicting_mask.sum()):,}"
    )

    # Do not train or evaluate on ambiguous groups.
    clean = df.loc[~conflicting_mask].copy()

    all_metrics = []
    all_checks = []
    all_predictions = []

    # STEP 4: Run leave-one-source-out testing.

    for heldout in SOURCES:

        print(
            f"\n===== HELD-OUT SOURCE: {heldout} ====="
        )

        # The fourth source becomes our unseen evaluation set.
        test = clean[
            clean["source_id"] == heldout
        ].copy()

        # Train using the other three sources.
        possible_train = clean[
            clean["source_id"] != heldout
        ].copy()

        # STEP 5: Prevent cross-source duplicate leakage.
        # If a detected duplicate family appears in the
        # held-out source, remove its copies from training.

        heldout_families = set(
            test["leakage_group_id"]
        )

        overlapping = (
            possible_train["leakage_group_id"]
            .isin(heldout_families)
        )

        removed_shared = int(overlapping.sum())

        train = possible_train.loc[
            ~overlapping
        ].copy()

        # Verify no detected families cross the boundary.
        overlap = (
            set(train["leakage_group_id"])
            & set(test["leakage_group_id"])
        )

        if overlap:
            raise RuntimeError(
                "Duplicate family leakage detected!"
            )

        if heldout in set(train["source_id"]):
            raise RuntimeError(
                "Held-out source was found in training!"
            )

        if set(test["source_id"]) != {heldout}:
            raise RuntimeError(
                "Unexpected source in held-out data."
            )

        if (
            train["project_label"].nunique() != 2
            or test["project_label"].nunique() != 2
        ):
            raise ValueError(
                "Both classes are required in each experiment."
            )

        print(
            "Training sources:",
            ", ".join(s for s in SOURCES if s != heldout)
        )

        print(f"Training emails: {len(train):,}")
        print(f"Unseen test emails: {len(test):,}")

        print(
            f"Cross-source duplicate records "
            f"removed from training: {removed_shared:,}"
        )

        # STEP 6: Prepare text and labels.

        y_train = train["project_label"].to_numpy()
        y_test = test["project_label"].to_numpy()

        # Create a NEW TF-IDF vectorizer for every fold.
        # It only learns vocabulary from the training sources.

        vectorizer = TfidfVectorizer(
            max_features=30000,
            min_df=3,
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        X_train = vectorizer.fit_transform(
            train["model_text"]
        )

        X_test = vectorizer.transform(
            test["model_text"]
        )

        print(
            f"TF-IDF features: {X_train.shape[1]:,}"
        )

        # Record leakage checks for this experiment.

        all_checks.append({
            "held_out_source": heldout,
            "training_emails": len(train),
            "heldout_emails": len(test),
            "removed_shared_family_records": removed_shared,
            "source_overlap": 0,
            "detected_family_overlap": 0,
            "tfidf_features": X_train.shape[1]
        })

        # STEP 7: Train both models on the same sources.

        for model_name, model in make_models().items():

            print(f"\nTraining {model_name}...")

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            if model_name == "Logistic Regression":
                scores = model.predict_proba(X_test)[:, 1]
            else:
                scores = model.decision_function(X_test)

            # STEP 8: Evaluate on the unseen source.

            result = evaluate(
                y_test,
                predictions,
                scores
            )

            result_row = {
                "model": model_name,
                "held_out_source": heldout,
                "training_emails": len(train),
                "heldout_emails": len(test),
                "heldout_spam_count": int(y_test.sum()),
                "heldout_spam_fraction": float(y_test.mean()),
                **result
            }

            all_metrics.append(result_row)

            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"Precision: {result['precision']:.4f}")
            print(f"Recall: {result['recall']:.4f}")
            print(f"F1-score: {result['f1']:.4f}")

            print(
                f"Average precision: "
                f"{result['average_precision']:.4f}"
            )

            print(
                f"PR-AUC: "
                f"{result['pr_auc_trapezoid']:.4f}"
            )

            print(
                f"Confusion matrix: "
                f"TN={result['tn']}, "
                f"FP={result['fp']}, "
                f"FN={result['fn']}, "
                f"TP={result['tp']}"
            )

            # Save predictions privately for later error analysis
            # and confidence interval calculations.

            private_rows = test[[
                "source_id",
                "source_row",
                "project_label",
                "leakage_group_id"
            ]].copy()

            private_rows["model"] = model_name
            private_rows["predicted_label"] = predictions
            private_rows["score"] = scores

            all_predictions.append(private_rows)

    # STEP 9: Save evaluation results.

    results_df = pd.DataFrame(all_metrics)

    results_df.to_csv(
        RESULTS / "loso_model_metrics.csv",
        index=False
    )

    pd.DataFrame(all_checks).to_csv(
        RESULTS / "loso_leakage_checks.csv",
        index=False
    )

    pd.concat(
        all_predictions,
        ignore_index=True
    ).to_csv(
        DATA / "loso_predictions_PRIVATE.csv",
        index=False
    )

    # STEP 10: Save experiment configuration.

    settings = {
        "random_seed": SEED,
        "tfidf_max_features": 30000,
        "tfidf_min_df": 3,
        "tfidf_ngram_range": "(1, 2)",
        "tfidf_sublinear_tf": True,
        "logistic_solver": "liblinear",
        "logistic_max_iter": 1000,
        "svm_C": 1.0,
        "svm_dual": False,
        "svm_max_iter": 5000,
        "classification_thresholds": "Default, no LOSO tuning",
        "sklearn_version": sklearn.__version__
    }

    pd.DataFrame(
        settings.items(),
        columns=["setting", "value"]
    ).to_csv(
        RESULTS / "loso_config.csv",
        index=False
    )

    # STEP 11: Display summary table.

    print("\n===== LEAVE-ONE-SOURCE-OUT SUMMARY =====")

    print(
        results_df[[
            "held_out_source",
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "average_precision",
            "fp",
            "fn"
        ]].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n===== FILES SAVED =====")

    print("results/loso_model_metrics.csv")
    print("results/loso_leakage_checks.csv")
    print("results/loso_config.csv")
    print("data/loso_predictions_PRIVATE.csv")

    print("\nIMPORTANT NOTES:")
    print("No detected duplicate families crossed source boundaries.")
    print("Undetected near-duplicates may still exist.")
    print("95% confidence intervals will be calculated later.")
    print("Reserved controlled_test.csv records were not used.")


if __name__ == "__main__":
    main()
