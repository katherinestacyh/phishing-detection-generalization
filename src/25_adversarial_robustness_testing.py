
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA / "controlled_train_label_checked_v2.csv"
VALIDATION_FILE = DATA / "controlled_validation_v2.csv"
TEST_FILE = DATA / "controlled_test_v2.csv"  # overlap check only
REFERENCE_PREDICTIONS = DATA / "23_controlled_validation_predictions_v2_PRIVATE.csv"

KEY = ["source_id", "source_row"]
GROUP = "leakage_group_id"
SOURCES = ["CEAS_08", "Enron", "Ling", "SpamAssasin"]
SEED = 42
MODELS = ["Logistic Regression", "Linear SVM"]

# Deterministic word substitutions; maximum of three substitutions per email.
# These are simple simulations, not automatically optimized adversarial attacks.
OBFUSCATIONS = {
    "free": "fr33", "prize": "pr1ze", "winner": "w1nner",
    "urgent": "urg3nt", "verify": "ver1fy", "account": "acc0unt",
    "password": "passw0rd", "bonus": "b0nus", "offer": "0ffer",
    "click": "cl1ck", "reward": "rew4rd", "money": "m0ney",
    "limited": "lim1ted", "congratulations": "congratulati0ns",
}
KEYWORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(map(re.escape, OBFUSCATIONS)) + r")\b",
    re.IGNORECASE,
)
LONG_WORD_PATTERN = re.compile(r"\b[A-Za-z]{7,}\b")
PROTECTED_PATTERN = re.compile(r"https?://\S+|www\.\S+|\S+@\S+", re.IGNORECASE)


def load_split(path, name, columns):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run Scripts 22e and 23 first.")
    df = pd.read_csv(path, usecols=columns, dtype={"source_id": str}, low_memory=False)
    df["source_row"] = pd.to_numeric(df["source_row"], errors="raise").astype("int64")
    if df[KEY + [GROUP]].isna().any().any() or df.duplicated(KEY).any():
        raise ValueError(f"Missing or duplicate IDs/families in {name}")
    df[GROUP] = df[GROUP].astype(str)
    if "project_label" in df.columns:
        if df["project_label"].isna().any() or not df["project_label"].isin([0, 1]).all():
            raise ValueError(f"Invalid labels in {name}")
    if "model_text" in df.columns:
        df["model_text"] = df["model_text"].fillna("").astype(str)
    return df


def record_ids(df):
    return set(df[KEY].itertuples(index=False, name=None))


def check_boundaries(train, val, test):
    for left_name, left, right_name, right in [
        ("train", train, "validation", val),
        ("train", train, "reserved test", test),
        ("validation", val, "reserved test", test),
    ]:
        if record_ids(left) & record_ids(right):
            raise ValueError(f"Record leakage between {left_name} and {right_name}")
        if set(left[GROUP]) & set(right[GROUP]):
            raise ValueError(f"Detected duplicate-family leakage: {left_name} vs {right_name}")
    if set(train["source_id"]) != set(SOURCES) or set(val["source_id"]) != set(SOURCES):
        raise ValueError("Unexpected sources in corrected splits")
    if train["project_label"].nunique() != 2 or val["project_label"].nunique() != 2:
        raise ValueError("Both spam and legitimate classes are required")


def change_case(text):
    """A negative/control-like change: TF-IDF lowercases by default."""
    return text.swapcase()


def obfuscate_keywords(text):
    """Replace up to three predefined words with visually similar spellings."""
    replaced = 0

    def replace(match):
        nonlocal replaced
        if replaced >= 3:
            return match.group(0)
        replaced += 1
        return OBFUSCATIONS[match.group(0).lower()]

    return KEYWORD_PATTERN.sub(replace, text)


def insert_zero_width_spaces(text):
    """Insert U+200B inside up to three long words, avoiding URL/email spans."""
    protected = [(m.start(), m.end()) for m in PROTECTED_PATTERN.finditer(text)]
    inserted = 0

    def replace(match):
        nonlocal inserted
        if inserted >= 3:
            return match.group(0)
        if any(start <= match.start() < end for start, end in protected):
            return match.group(0)
        word = match.group(0)
        halfway = len(word) // 2
        inserted += 1
        return word[:halfway] + "\u200b" + word[halfway:]

    return LONG_WORD_PATTERN.sub(replace, text)


PERTURBATIONS = {
    "case_change": change_case,
    "keyword_obfuscation": obfuscate_keywords,
    "zero_width_words": insert_zero_width_spaces,
}


def make_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, solver="liblinear", random_state=SEED,
        ),
        "Linear SVM": LinearSVC(
            C=1.0, dual=False, max_iter=5000, random_state=SEED,
        ),
    }


def model_scores(model, model_name, matrix):
    if model_name == "Logistic Regression":
        return model.predict_proba(matrix)[:, 1]
    return model.decision_function(matrix)


def rates(y, predictions):
    """Classification metrics; return NaN if a class denominator is absent."""
    y = np.asarray(y, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    if not len(y):
        return {k: np.nan for k in [
            "accuracy", "precision", "recall", "f1",
            "false_negative_rate", "false_positive_rate",
        ]}
    pos = y == 1
    neg = y == 0
    fn = int(np.sum(pos & (predictions == 0)))
    fp = int(np.sum(neg & (predictions == 1)))
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "false_negative_rate": fn / int(pos.sum()) if pos.any() else np.nan,
        "false_positive_rate": fp / int(neg.sum()) if neg.any() else np.nan,
    }


def check_against_script23(val, baseline_predictions):
    """Stop if retrained baseline differs from the corrected Script 23 baseline."""
    if not REFERENCE_PREDICTIONS.is_file():
        raise FileNotFoundError(
            f"Missing {REFERENCE_PREDICTIONS}. Run Script 23 first."
        )
    reference = pd.read_csv(
        REFERENCE_PREDICTIONS,
        usecols=KEY + ["model", "predicted_label"],
        dtype={"source_id": str},
    )
    reference["source_row"] = pd.to_numeric(
        reference["source_row"], errors="raise"
    ).astype("int64")
    if reference.duplicated(KEY + ["model"]).any():
        raise ValueError("Duplicate rows in Script 23 baseline predictions")
    if len(reference) != len(val) * len(MODELS):
        raise ValueError("Script 23 prediction count does not match corrected validation")
    check = baseline_predictions.merge(
        reference, on=KEY + ["model"], how="left", validate="one_to_one",
        suffixes=("_script25", "_script23"), indicator=True,
    )
    if (check["_merge"] != "both").any():
        raise ValueError("Some corrected validation records are absent from Script 23")
    different = int((
        check["predicted_label_script25"] != check["predicted_label_script23"]
    ).sum())
    if different:
        raise ValueError(
            f"Baseline predictions disagree with Script 23 on {different} records. "
            "Check library versions or input preprocessing before interpreting robustness."
        )
    print("PASS: Both fitted models reproduce Script 23's corrected validation predictions.")


def summarize(y, before, after, changed, source, model_name, variant):
    """Evaluate overall rates and flips on the SAME modified-record subset."""
    y = np.asarray(y, dtype=int)
    before = np.asarray(before, dtype=int)
    after = np.asarray(after, dtype=int)
    changed = np.asarray(changed, dtype=bool)
    before_all = rates(y, before)
    after_all = rates(y, after)
    before_changed = rates(y[changed], before[changed])
    after_changed = rates(y[changed], after[changed])

    originally_caught_spam = changed & (y == 1) & (before == 1)
    newly_missed = originally_caught_spam & (after == 0)
    originally_correct_legitimate = changed & (y == 0) & (before == 0)
    newly_flagged = originally_correct_legitimate & (after == 1)

    return {
        "source": source,
        "model": model_name,
        "perturbation": variant,
        "emails_evaluated": len(y),
        "emails_changed": int(changed.sum()),
        "spam_changed": int((changed & (y == 1)).sum()),
        "legitimate_changed": int((changed & (y == 0)).sum()),
        "prediction_flips_all": int((before != after).sum()),
        "prediction_flips_on_changed": int((changed & (before != after)).sum()),
        "baseline_accuracy_all": before_all["accuracy"],
        "modified_accuracy_all": after_all["accuracy"],
        "baseline_f1_all": before_all["f1"],
        "modified_f1_all": after_all["f1"],
        "baseline_fnr_changed_only": before_changed["false_negative_rate"],
        "modified_fnr_changed_only": after_changed["false_negative_rate"],
        "baseline_fpr_changed_only": before_changed["false_positive_rate"],
        "modified_fpr_changed_only": after_changed["false_positive_rate"],
        "originally_caught_spam_changed": int(originally_caught_spam.sum()),
        "newly_missed_spam_count": int(newly_missed.sum()),
        "newly_missed_spam_rate": (
            newly_missed.sum() / originally_caught_spam.sum()
            if originally_caught_spam.any() else np.nan
        ),
        "originally_correct_legitimate_changed": int(originally_correct_legitimate.sum()),
        "newly_flagged_legitimate_count": int(newly_flagged.sum()),
        "newly_flagged_legitimate_rate": (
            newly_flagged.sum() / originally_correct_legitimate.sum()
            if originally_correct_legitimate.any() else np.nan
        ),
    }


def main():
    print("Loading corrected Script 22e datasets...")
    required = KEY + [GROUP, "project_label", "model_text"]
    train = load_split(TRAIN_FILE, "training", required)
    val = load_split(VALIDATION_FILE, "validation", required)
    # Only identifiers and grouping, no test labels or text.
    test = load_split(TEST_FILE, "reserved test IDs", KEY + [GROUP])
    check_boundaries(train, val, test)
    print(f"Training={len(train):,} | Validation={len(val):,} | reserved test not evaluated")

    print("\nFitting the two Script 23 baseline models...")
    tfidf = TfidfVectorizer(
        max_features=30000, min_df=3, ngram_range=(1, 2), sublinear_tf=True,
    )
    x_train = tfidf.fit_transform(train["model_text"])
    x_val = tfidf.transform(val["model_text"])
    y_train = train["project_label"].to_numpy(dtype=int)
    y_val = val["project_label"].to_numpy(dtype=int)
    fitted = {}
    baseline = {}
    baseline_rows = []
    for model_name, model in make_models().items():
        model.fit(x_train, y_train)
        fitted[model_name] = model
        predictions = model.predict(x_val)
        scores = model_scores(model, model_name, x_val)
        baseline[model_name] = (predictions, scores)
        rows = val[KEY].copy()
        rows["model"] = model_name
        rows["predicted_label"] = predictions
        baseline_rows.append(rows)
        print(f"  {model_name}: baseline F1={f1_score(y_val, predictions):.4f}")

    check_against_script23(val, pd.concat(baseline_rows, ignore_index=True))

    overall = []
    per_source = []
    private_predictions = []
    base_text = val["model_text"].tolist()
    source_array = val["source_id"].to_numpy()

    for variant, modifier in PERTURBATIONS.items():
        print(f"\n===== TEST: {variant} =====")
        modified_text = [modifier(text) for text in base_text]
        changed = np.fromiter(
            (modified != original for modified, original in zip(modified_text, base_text)),
            dtype=bool, count=len(val),
        )
        print(f"Text changed for {int(changed.sum()):,} of {len(val):,} validation emails")
        if not changed.any():
            print("No text was affected by this transformation; recording a zero-coverage result.")

        # Vocabulary is fit on TRAINING ONLY; perturbed validation is transformed, never fitted.
        x_modified = tfidf.transform(modified_text)

        for model_name, model in fitted.items():
            before, before_score = baseline[model_name]
            after = model.predict(x_modified)
            after_score = model_scores(model, model_name, x_modified)

            overall.append(summarize(
                y_val, before, after, changed, "ALL", model_name, variant,
            ))
            for source in SOURCES:
                mask = source_array == source
                per_source.append(summarize(
                    y_val[mask], before[mask], after[mask], changed[mask],
                    source, model_name, variant,
                ))

            # No email text in the output; record identifiers stay PRIVATE.
            rows = val[KEY + ["project_label", GROUP]].copy()
            rows["model"] = model_name
            rows["perturbation"] = variant
            rows["text_changed"] = changed
            rows["original_prediction"] = before
            rows["modified_prediction"] = after
            rows["original_score"] = before_score
            rows["modified_score"] = after_score
            private_predictions.append(rows)

            summary = overall[-1]
            print(
                f"  {model_name}: flips={summary['prediction_flips_all']:,}, "
                f"newly missed spam={summary['newly_missed_spam_count']:,}, "
                f"newly flagged legitimate={summary['newly_flagged_legitimate_count']:,}"
            )

    pd.DataFrame(overall).to_csv(
        RESULTS / "25_adversarial_robustness_summary.csv", index=False,
    )
    pd.DataFrame(per_source).to_csv(
        RESULTS / "25_adversarial_robustness_by_source.csv", index=False,
    )
    pd.concat(private_predictions, ignore_index=True).to_csv(
        DATA / "25_adversarial_predictions_PRIVATE.csv", index=False,
    )
    pd.DataFrame([
        {"setting": "training_file", "value": TRAIN_FILE.name},
        {"setting": "validation_file", "value": VALIDATION_FILE.name},
        {"setting": "reserved_test_evaluated", "value": False},
        {"setting": "models", "value": ", ".join(MODELS)},
        {"setting": "tfidf_features", "value": x_train.shape[1]},
        {"setting": "tfidf_min_df", "value": 3},
        {"setting": "tfidf_ngram_range", "value": "(1, 2)"},
        {"setting": "tfidf_sublinear_tf", "value": True},
        {"setting": "random_seed", "value": SEED},
        {"setting": "perturbations", "value": ", ".join(PERTURBATIONS)},
        {"setting": "experimental_limitations", "value":
         "Fixed label-preserving text edits; no adaptive attacker or external test"},
    ]).to_csv(RESULTS / "25_adversarial_robustness_config.csv", index=False)

    print("\n===== SCRIPT 25 SUMMARY =====")
    print(pd.DataFrame(overall)[[
        "model", "perturbation", "emails_changed", "prediction_flips_all",
        "newly_missed_spam_count", "newly_flagged_legitimate_count",
    ]].to_string(index=False))
    print("\nSaved aggregate reports in results/: 25_adversarial_robustness_summary.csv, "
          "25_adversarial_robustness_by_source.csv, 25_adversarial_robustness_config.csv")
    print("Saved private predictions in data/: 25_adversarial_predictions_PRIVATE.csv")
    print("No original emails, labels, or splits were changed. Reserved test was not evaluated.")
    print("These are simulated fixed perturbations, not an estimate of real-world attack success.")


if __name__ == "__main__":
    main()
