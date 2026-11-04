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

# Project folders
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
RESULTS.mkdir(exist_ok=True)

TRAIN_FILE = DATA / "controlled_train_label_checked.csv"
VAL_FILE = DATA / "controlled_validation.csv"

SEED = 42


def load_dataset(path):
    """Load data and check required columns."""

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, low_memory=False)

    required = [
        "model_text", "project_label", "source_id",
        "source_row", "leakage_group_id"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required[1:]].isna().any().any():
        raise ValueError("Missing labels or source identifiers.")

    if not df["project_label"].isin([0, 1]).all():
        raise ValueError("Expected labels 0 and 1 only.")

    df["model_text"] = df["model_text"].fillna("").astype(str)

    return df


def evaluate_model(y_true, predictions, scores):
    """Calculate classification metrics."""

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "emails": len(y_true),
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
        "false_positive_rate": (
            fp / (fp + tn) if fp + tn else np.nan
        ),
        "false_negative_rate": (
            fn / (fn + tp) if fn + tp else np.nan
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

    if len(np.unique(y_true)) == 2:
        precision, recall, _ = precision_recall_curve(
            y_true, scores
        )

        metrics["average_precision"] = (
            average_precision_score(y_true, scores)
        )

        metrics["pr_auc_trapezoid"] = auc(
            recall, precision
        )
    else:
        metrics["average_precision"] = np.nan
        metrics["pr_auc_trapezoid"] = np.nan

    return metrics


def main():

    # STEP 1: Load controlled training and validation data
    print("Loading controlled datasets...")

    train = load_dataset(TRAIN_FILE)
    validation = load_dataset(VAL_FILE)

    print(f"Training emails: {len(train):,}")
    print(f"Validation emails: {len(validation):,}")

    # STEP 2: Check for duplicate leakage
    train_groups = set(train["leakage_group_id"])
    val_groups = set(validation["leakage_group_id"])

    if train_groups.intersection(val_groups):
        raise RuntimeError(
            "Duplicate group overlap detected!"
        )

    train_records = set(zip(
        train["source_id"], train["source_row"]
    ))

    val_records = set(zip(
        validation["source_id"], validation["source_row"]
    ))

    if train_records.intersection(val_records):
        raise RuntimeError(
            "Training and validation contain overlapping records!"
        )

    print("Detected duplicate group overlap: 0")

    # STEP 3: Prepare text and labels
    X_train_text = train["model_text"]
    y_train = train["project_label"].to_numpy()

    X_val_text = validation["model_text"]
    y_val = validation["project_label"].to_numpy()

    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("Both datasets must contain both labels.")

    # STEP 4: Convert email text to TF-IDF features
    # Same settings used in Script 15
    vectorizer = TfidfVectorizer(
        max_features=30000,
        min_df=3,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    print("\nCreating TF-IDF features...")

    # Only learn the vocabulary from training data
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    print(f"TF-IDF features: {X_train.shape[1]:,}")

    # STEP 5: Define our two models
    models = {
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

    all_metrics = []
    source_metrics = []

    # Save predictions privately for later error analysis
    saved_predictions = validation[
        ["source_id", "source_row", "project_label"]
    ].copy()

    # STEP 6: Train and evaluate each model
    for model_name, model in models.items():

        print(f"\n===== TRAINING {model_name.upper()} =====")

        model.fit(X_train, y_train)

        print("Training complete!")

        for split_name, X, y in [
            ("train", X_train, y_train),
            ("validation", X_val, y_val)
        ]:

            predictions = model.predict(X)

            # Logistic Regression gives probabilities.
            # Linear SVM gives decision scores.
            if model_name == "Logistic Regression":
                scores = model.predict_proba(X)[:, 1]
            else:
                scores = model.decision_function(X)

            metrics = evaluate_model(
                y, predictions, scores
            )

            all_metrics.append({
                "model": model_name,
                "split": split_name,
                **metrics
            })

            if split_name == "validation":

                print("\nValidation results:")

                for key in [
                    "accuracy", "precision", "recall",
                    "f1", "average_precision",
                    "pr_auc_trapezoid",
                    "false_positive_rate",
                    "false_negative_rate"
                ]:
                    print(f"{key}: {metrics[key]:.4f}")

                print(
                    f"Confusion matrix: "
                    f"TN={metrics['tn']}, "
                    f"FP={metrics['fp']}, "
                    f"FN={metrics['fn']}, "
                    f"TP={metrics['tp']}"
                )

                saved_predictions[
                    f"{model_name}_prediction"
                ] = predictions

                saved_predictions[
                    f"{model_name}_score"
                ] = scores

                # STEP 7: Evaluate each source separately
                for source in sorted(
                    validation["source_id"].unique()
                ):

                    mask = (
                        validation["source_id"] == source
                    ).to_numpy()

                    source_result = evaluate_model(
                        y[mask],
                        predictions[mask],
                        scores[mask]
                    )

                    source_metrics.append({
                        "model": model_name,
                        "source": source,
                        **source_result
                    })

    # STEP 8: Save model comparison results
    results_df = pd.DataFrame(all_metrics)

    results_df.to_csv(
        RESULTS / "controlled_model_comparison_metrics.csv",
        index=False
    )

    pd.DataFrame(source_metrics).to_csv(
        RESULTS / "controlled_model_validation_by_source.csv",
        index=False
    )

    saved_predictions.to_csv(
        DATA / "controlled_validation_predictions_PRIVATE.csv",
        index=False
    )

    # STEP 9: Save settings for reproducibility
    settings = {
        "random_seed": SEED,
        "training_emails": len(train),
        "validation_emails": len(validation),
        "tfidf_max_features": 30000,
        "tfidf_min_df": 3,
        "tfidf_ngrams": "(1, 2)",
        "tfidf_sublinear_tf": True,
        "logistic_solver": "liblinear",
        "logistic_max_iter": 1000,
        "svm_C": 1.0,
        "svm_max_iter": 5000,
        "svm_dual": False,
        "class_weight": "None",
        "sklearn_version": sklearn.__version__,
        "model_selection": "Default thresholds; no tuning yet"
    }

    pd.DataFrame(
        settings.items(),
        columns=["setting", "value"]
    ).to_csv(
        RESULTS / "controlled_model_comparison_config.csv",
        index=False
    )

    # STEP 10: Display comparison
    print("\n===== FINAL VALIDATION COMPARISON =====")

    comparison = results_df[
        results_df["split"] == "validation"
    ]

    print(comparison[[
        "model", "accuracy", "precision",
        "recall", "f1", "average_precision",
        "pr_auc_trapezoid"
    ]].to_string(index=False))

    print("\n===== FILES SAVED =====")
    print("results/controlled_model_comparison_metrics.csv")
    print("results/controlled_model_validation_by_source.csv")
    print("results/controlled_model_comparison_config.csv")
    print("data/controlled_validation_predictions_PRIVATE.csv")

    print("\nThe reserved test dataset was NOT used.")


if __name__ == "__main__":
    main()
