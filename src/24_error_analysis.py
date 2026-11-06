

from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TRAIN = DATA / "controlled_train_label_checked_v2.csv"
VALIDATION = DATA / "controlled_validation_v2.csv"
RESERVED_TEST = DATA / "controlled_test_v2.csv"  # IDs only; never evaluated
PREDICTION_FILES = {
    "controlled_validation": DATA / "23_controlled_validation_predictions_v2_PRIVATE.csv",
    "loso": DATA / "23_loso_predictions_v2_PRIVATE.csv",
}

KEY = ["source_id", "source_row"]
GROUP = "leakage_group_id"
MODELS = ["Logistic Regression", "Linear SVM"]
SOURCES = ["CEAS_08", "Enron", "Ling", "SpamAssasin"]
BOOLEAN_FEATURES = ["has_url", "has_email_address", "has_html_tag", "has_digit", "has_exclamation"]
NUMERIC_FEATURES = ["character_count", "word_count"]
EXAMPLES_PER_ERROR_TYPE = 3
SEED = 42


def read_table(path, required):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run Script 23 first.")
    df = pd.read_csv(path, usecols=required, low_memory=False, dtype={"source_id": str})
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    df["source_row"] = pd.to_numeric(df["source_row"], errors="raise").astype("int64")
    if df[KEY].isna().any().any():
        raise ValueError(f"Missing email IDs in {path.name}")
    return df


def make_features(texts):
    """Simple descriptions of emails. These are NOT explanations of errors."""
    text = texts.fillna("").astype(str)
    low = text.str.lower()
    return pd.DataFrame({
        "has_url": low.str.contains(r"https?://|www\.", regex=True, na=False),
        "has_email_address": low.str.contains(
            r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", regex=True, na=False
        ),
        "has_html_tag": low.str.contains(
            r"<(?:html|body|a|div|span|p|table|td|br)\b", regex=True, na=False
        ),
        "has_digit": text.str.contains(r"\d", regex=True, na=False),
        "has_exclamation": text.str.contains("!", regex=False, na=False),
        "character_count": text.str.len().astype(int),
        "word_count": text.str.count(r"\b\w+\b").astype(int),
    }, index=text.index)


def load_predictions(path, expected, scope):
    """Check the predictions match their corrected source emails exactly."""
    pred = read_table(path, KEY + ["project_label", GROUP, "model", "predicted_label", "score"])
    if pred.duplicated(KEY + ["model"]).any():
        raise ValueError(f"Duplicate prediction records in {scope}")
    if set(pred["model"]) != set(MODELS):
        raise ValueError(f"Unexpected model names in {scope}")
    if not pred["project_label"].isin([0, 1]).all():
        raise ValueError(f"Invalid true labels in {scope}")
    if not pred["predicted_label"].isin([0, 1]).all():
        raise ValueError(f"Invalid predicted labels in {scope}")
    if not np.isfinite(pd.to_numeric(pred["score"], errors="raise")).all():
        raise ValueError(f"Non-finite decision scores in {scope}")
    if len(pred) != len(expected) * len(MODELS):
        raise ValueError(f"Wrong number of {scope} predictions")

    truth = expected[KEY + ["project_label", GROUP]].copy()
    joined = pred.merge(truth, on=KEY, how="left", validate="many_to_one",
                        suffixes=("", "_expected"), indicator=True)
    if not (joined["_merge"] == "both").all():
        raise ValueError(f"{scope} predictions contain IDs outside corrected data")
    if not (joined["project_label"] == joined["project_label_expected"]).all():
        raise ValueError(f"{scope} prediction labels disagree with corrected data")
    if not (joined[GROUP].astype(str) == joined[f"{GROUP}_expected"].astype(str)).all():
        raise ValueError(f"{scope} duplicate group IDs disagree with corrected data")
    for name in MODELS:
        records = joined.loc[joined["model"] == name, KEY]
        if len(records) != len(expected):
            raise ValueError(f"Incomplete {scope} predictions for {name}")
    return pred


def error_type(y, prediction):
    if y == 0 and prediction == 1:
        return "false_positive"  # legitimate email incorrectly blocked
    if y == 1 and prediction == 0:
        return "false_negative"  # spam incorrectly allowed
    return "correct"


def summarize_errors(frame, scope):
    rows = []
    for name in MODELS:
        part = frame.loc[frame["model"] == name]
        for source in ["ALL", *SOURCES]:
            current = part if source == "ALL" else part.loc[part["source_id"] == source]
            y, p = current["project_label"], current["predicted_label"]
            tn = int(((y == 0) & (p == 0)).sum())
            fp = int(((y == 0) & (p == 1)).sum())
            fn = int(((y == 1) & (p == 0)).sum())
            tp = int(((y == 1) & (p == 1)).sum())
            rows.append({
                "evaluation": scope, "source": source, "model": name,
                "emails": len(current), "legitimate": tn + fp, "spam": tp + fn,
                "true_negatives": tn, "false_positives": fp,
                "false_negatives": fn, "true_positives": tp,
                "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
                "false_negative_rate": fn / (fn + tp) if fn + tp else np.nan,
                "error_rate": (fp + fn) / len(current) if len(current) else np.nan,
            })
    return rows


def compare_error_features(frame, scope):
    """Compare errors with correct predictions of the SAME true class."""
    rows = []
    for name in MODELS:
        by_model = frame.loc[frame["model"] == name]
        for source in ["ALL", *SOURCES]:
            part = by_model if source == "ALL" else by_model.loc[by_model["source_id"] == source]
            for category, actual, wrong_prediction in [
                ("false_positive", 0, 1),
                ("false_negative", 1, 0),
            ]:
                same_class = part.loc[part["project_label"] == actual]
                errors = same_class.loc[same_class["predicted_label"] == wrong_prediction]
                correct = same_class.loc[same_class["predicted_label"] == actual]
                for feature in BOOLEAN_FEATURES:
                    err_rate = float(errors[feature].mean()) if len(errors) else np.nan
                    ok_rate = float(correct[feature].mean()) if len(correct) else np.nan
                    rows.append({
                        "evaluation": scope, "source": source, "model": name,
                        "error_type": category, "feature": feature,
                        "statistic": "fraction_true", "n_errors": len(errors),
                        "n_correct_same_true_class": len(correct),
                        "error_value": err_rate, "correct_value": ok_rate,
                        "difference": err_rate - ok_rate,
                    })
                for feature in NUMERIC_FEATURES:
                    err_median = float(errors[feature].median()) if len(errors) else np.nan
                    ok_median = float(correct[feature].median()) if len(correct) else np.nan
                    rows.append({
                        "evaluation": scope, "source": source, "model": name,
                        "error_type": category, "feature": feature,
                        "statistic": "median", "n_errors": len(errors),
                        "n_correct_same_true_class": len(correct),
                        "error_value": err_median, "correct_value": ok_median,
                        "difference": err_median - ok_median,
                    })
    return rows


def model_agreement(frame, scope):
    """Compare both models' predictions on the same emails."""
    wide = frame.pivot(index=KEY, columns="model", values="predicted_label").reset_index()
    true = frame.loc[frame["model"] == MODELS[0], KEY + ["project_label"]]
    wide = wide.merge(true, on=KEY, validate="one_to_one")
    rows = []
    for source in ["ALL", *SOURCES]:
        part = wide if source == "ALL" else wide.loc[wide["source_id"] == source]
        lr = part["Logistic Regression"] == part["project_label"]
        svm = part["Linear SVM"] == part["project_label"]
        rows.append({
            "evaluation": scope, "source": source, "emails": len(part),
            "both_correct": int((lr & svm).sum()),
            "both_wrong": int((~lr & ~svm).sum()),
            "only_logistic_correct": int((lr & ~svm).sum()),
            "only_svm_correct": int((~lr & svm).sum()),
            "models_disagree": int((part[MODELS[0]] != part[MODELS[1]]).sum()),
        })
    return rows


def error_family_concentration(frame, scope):
    rows = []
    for name in MODELS:
        by_model = frame.loc[frame["model"] == name]
        for source in SOURCES:
            part = by_model.loc[by_model["source_id"] == source]
            for category in ["false_positive", "false_negative"]:
                errors = part.loc[part["error_type"] == category]
                family_sizes = errors.groupby(GROUP).size().sort_values(ascending=False)
                rows.append({
                    "evaluation": scope, "source": source, "model": name,
                    "error_type": category, "error_emails": len(errors),
                    "distinct_error_families": len(family_sizes),
                    "largest_family_error_count": int(family_sizes.iloc[0]) if len(family_sizes) else 0,
                    "largest_family_error_share": (
                        float(family_sizes.iloc[0] / len(errors)) if len(errors) else np.nan
                    ),
                })
    return rows


def redacted_preview(text, limit=280):
    """Short local-only excerpt. Redact straightforward email addresses and URLs."""
    text = " ".join(str(text).split())
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text, flags=re.I)
    return text[:limit]


def choose_private_examples(frame, raw_emails, scope):
    rows = []
    for name in MODELS:
        for source in SOURCES:
            for category in ["false_positive", "false_negative"]:
                subset = frame.loc[
                    (frame["model"] == name)
                    & (frame["source_id"] == source)
                    & (frame["error_type"] == category)
                ]
                if subset.empty:
                    continue
                chosen = subset.sample(
                    n=min(EXAMPLES_PER_ERROR_TYPE, len(subset)), random_state=SEED
                )
                rows.append(chosen[KEY + [GROUP, "project_label", "predicted_label",
                                          "score", "model", "error_type"]].assign(evaluation=scope))
    if not rows:
        return pd.DataFrame()
    examples = pd.concat(rows, ignore_index=True)
    examples = examples.merge(raw_emails[KEY + ["model_text"]], on=KEY,
                              how="left", validate="many_to_one")
    if examples["model_text"].isna().any():
        raise ValueError("Missing email text for an example")
    examples["text_preview_REDACTED"] = examples["model_text"].map(redacted_preview)
    examples["manual_review_notes"] = ""  # fill this in after reading examples
    return examples.drop(columns="model_text")


def main():
    print("Loading corrected Script 22e data and Script 23 predictions...")
    cols = KEY + ["project_label", GROUP, "model_text"]
    train = read_table(TRAIN, cols)[cols]
    val = read_table(VALIDATION, cols)[cols]
    test = read_table(RESERVED_TEST, KEY)[KEY]  # identifiers only
    pool = pd.concat([train, val], ignore_index=True)

    if pool.duplicated(KEY).any() or test.duplicated(KEY).any():
        raise ValueError("Repeated email identifiers in corrected datasets")
    if set(map(tuple, pool[KEY].to_numpy())) & set(map(tuple, test[KEY].to_numpy())):
        raise ValueError("Reserved test IDs overlap with analysis data")

    features = make_features(pool["model_text"])
    metadata = pd.concat([pool[KEY + ["project_label", GROUP]].reset_index(drop=True),
                          features.reset_index(drop=True)], axis=1)

    all_counts, all_features, all_agreement, all_concentration, all_examples = [], [], [], [], []
    for scope, file in PREDICTION_FILES.items():
        expected = val if scope == "controlled_validation" else pool
        preds = load_predictions(file, expected, scope)
        enriched = preds.merge(metadata[KEY + BOOLEAN_FEATURES + NUMERIC_FEATURES],
                               on=KEY, how="left", validate="many_to_one")
        if enriched[BOOLEAN_FEATURES + NUMERIC_FEATURES].isna().any().any():
            raise ValueError(f"Missing text features in {scope}")
        enriched["error_type"] = [
            error_type(y, p) for y, p in zip(enriched["project_label"],
                                              enriched["predicted_label"])
        ]
        all_counts.extend(summarize_errors(enriched, scope))
        all_features.extend(compare_error_features(enriched, scope))
        all_agreement.extend(model_agreement(enriched, scope))
        all_concentration.extend(error_family_concentration(enriched, scope))
        examples = choose_private_examples(enriched, expected, scope)
        if not examples.empty:
            all_examples.append(examples)
        print(f"Checked {scope}: {len(expected):,} emails, {len(preds):,} predictions")

    pd.DataFrame(all_counts).to_csv(RESULTS / "24_error_counts_by_source.csv", index=False)
    pd.DataFrame(all_features).to_csv(RESULTS / "24_error_feature_comparisons.csv", index=False)
    pd.DataFrame(all_agreement).to_csv(RESULTS / "24_model_agreement.csv", index=False)
    pd.DataFrame(all_concentration).to_csv(
        RESULTS / "24_error_family_concentration.csv", index=False
    )
    if all_examples:
        pd.concat(all_examples, ignore_index=True).to_csv(
            DATA / "24_error_examples_PRIVATE.csv", index=False
        )

    print("\n===== SCRIPT 24: LOSO ERRORS BY SOURCE =====")
    display = pd.DataFrame(all_counts)
    display = display.loc[(display["evaluation"] == "loso") & (display["source"] != "ALL")]
    print(display[["source", "model", "emails", "false_positives", "false_negatives",
                   "false_positive_rate", "false_negative_rate"]].to_string(
                       index=False, float_format=lambda x: f"{x:.4f}"
                   ))
    print("\nSaved aggregate reports in results/: 24_error_counts_by_source.csv, "
          "24_error_feature_comparisons.csv, 24_model_agreement.csv, "
          "24_error_family_concentration.csv")
    print("Saved PRIVATE examples in data/: 24_error_examples_PRIVATE.csv")
    print("No models retrained. Reserved test emails NOT evaluated.")
    print("Text patterns are descriptive; manually review examples before explaining causes.")


if __name__ == "__main__":
    main()
