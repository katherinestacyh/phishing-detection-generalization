from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve, auc,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

TRAIN_FILE = DATA / "controlled_train_label_checked_v2.csv"
VAL_FILE = DATA / "controlled_validation_v2.csv"
TEST_FILE = DATA / "controlled_test_v2.csv"  # checked for leakage ONLY
OLD_VAL_PRED = DATA / "controlled_validation_predictions_PRIVATE.csv"
OLD_LOSO_METRICS = RESULTS / "loso_model_metrics.csv"

SOURCES = ["CEAS_08", "Enron", "Ling", "SpamAssasin"]
MODELS = ["Logistic Regression", "Linear SVM"]
KEY = ["source_id", "source_row"]
GROUP = "leakage_group_id"  # Script 22e replaced this with v2 IDs
SEED = 42
N_BOOTSTRAPS = 300
METRICS = [
    "accuracy", "precision", "recall", "f1", "average_precision",
    "pr_auc_trapezoid", "false_positive_rate", "false_negative_rate",
]


def load_split(path, name):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run Script 22e first.")
    df = pd.read_csv(path, low_memory=False, dtype={"source_id": str})
    required = set(KEY + ["project_label", "model_text", GROUP])
    if not required.issubset(df.columns):
        raise ValueError(f"{name} missing columns: {sorted(required - set(df.columns))}")
    if df[KEY + ["project_label", GROUP]].isna().any().any():
        raise ValueError(f"{name} has missing IDs, labels or group IDs")
    if df.duplicated(KEY).any():
        raise ValueError(f"{name} contains repeated source/row IDs")
    if not df["project_label"].isin([0, 1]).all():
        raise ValueError(f"{name} must have only labels 0/1")
    df["source_row"] = pd.to_numeric(df["source_row"], errors="raise").astype("int64")
    df[GROUP] = df[GROUP].astype(str)
    df["model_text"] = df["model_text"].fillna("").astype(str)
    return df


def as_keys(df):
    return set(df[KEY].itertuples(index=False, name=None))


def check_splits(train, val, test):
    # Reserved test is loaded solely to verify that it stays independent.
    for a_name, a, b_name, b in [
        ("training", train, "validation", val),
        ("training", train, "reserved test", test),
        ("validation", val, "reserved test", test),
    ]:
        if as_keys(a) & as_keys(b):
            raise ValueError(f"Record IDs overlap: {a_name} vs {b_name}")
        if set(a[GROUP]) & set(b[GROUP]):
            raise ValueError(f"Detected duplicate families overlap: {a_name} vs {b_name}")
    if set(train["source_id"]) != set(SOURCES):
        raise ValueError("Unexpected training source list")
    if set(val["source_id"]) != set(SOURCES):
        raise ValueError("Unexpected validation source list")
    for df, name in [(train, "train"), (val, "validation")]:
        if df["project_label"].nunique() != 2:
            raise ValueError(f"{name} must contain both classes")
    print("PASS: no record or detected family overlap among corrected splits.")


def new_vectorizer():
    return TfidfVectorizer(
        max_features=30000, min_df=3, ngram_range=(1, 2),
        sublinear_tf=True,
    )


def new_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, solver="liblinear", random_state=SEED,
        ),
        "Linear SVM": LinearSVC(
            C=1.0, dual=False, max_iter=5000, random_state=SEED,
        ),
    }


def model_scores(model, name, x):
    if name == "Logistic Regression":
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def evaluate(y, pred, scores, weights=None):
    """Calculate metrics; optional weights support family bootstrapping."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    scores = np.asarray(scores, dtype=float)
    w = np.ones(len(y), dtype=int) if weights is None else np.asarray(weights)
    if len(y) != len(pred) or len(y) != len(scores) or len(y) != len(w):
        raise ValueError("Prediction/label lengths differ")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite prediction score")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Invalid true labels")
    if not set(np.unique(pred)).issubset({0, 1}):
        raise ValueError("Invalid predicted labels")
    if w[y == 0].sum() == 0 or w[y == 1].sum() == 0:
        return None  # a resample lacking one class
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1], sample_weight=w,
    ).ravel()
    p, r, _ = precision_recall_curve(y, scores, sample_weight=w)
    return {
        "accuracy": accuracy_score(y, pred, sample_weight=w),
        "precision": precision_score(y, pred, sample_weight=w, zero_division=0),
        "recall": recall_score(y, pred, sample_weight=w, zero_division=0),
        "f1": f1_score(y, pred, sample_weight=w, zero_division=0),
        "average_precision": average_precision_score(y, scores, sample_weight=w),
        "pr_auc_trapezoid": auc(r, p),
        "false_positive_rate": fp / (fp + tn),
        "false_negative_rate": fn / (fn + tp),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def make_prediction_rows(df, model_name, predictions, scores):
    out = df[KEY + ["project_label", GROUP]].copy()
    out["model"] = model_name
    out["predicted_label"] = np.asarray(predictions, dtype=int)
    out["score"] = np.asarray(scores, dtype=float)
    return out


def reuse_validation_predictions(train, val):
    """Reuse only a verified long-format Script 20 prediction file."""
    if not OLD_VAL_PRED.exists():
        print("Script 20 predictions file not found; retraining validation models.")
        return None
    old = pd.read_csv(OLD_VAL_PRED, low_memory=False, dtype={"source_id": str})
    needed = set(KEY + ["model", "predicted_label", "score"])
    if not needed.issubset(old.columns):
        print("Script 20 predictions have a different column layout; retraining validation models.")
        return None
    if old.duplicated(KEY + ["model"]).any():
        raise ValueError("Old validation predictions contain duplicate IDs for a model")
    if set(old["model"]) != set(MODELS):
        print("Script 20 model names differ; retraining validation models.")
        return None
    old["source_row"] = pd.to_numeric(old["source_row"], errors="raise").astype("int64")
    expected = as_keys(val)
    for name in MODELS:
        available = as_keys(old.loc[old["model"] == name])
        if not expected.issubset(available):
            print("Script 20 predictions are missing corrected validation records; retraining models.")
            return None
    # Ensure the original labels agree, if present.
    if "project_label" in old.columns:
        truth = val[KEY + ["project_label"]].rename(columns={"project_label": "true_v2"})
        merged = old.merge(truth, on=KEY, validate="many_to_one")
        if not (merged["project_label"] == merged["true_v2"]).all():
            raise ValueError("Old prediction labels disagree with v2 validation")
    # Ensure the old train and v2 train contain exactly the same emails and text.
    old_train = DATA / "controlled_train_label_checked.csv"
    if not old_train.exists():
        print("Cannot verify old training set; retraining validation models.")
        return None
    prior = pd.read_csv(old_train, usecols=KEY + ["project_label", "model_text"],
                        dtype={"source_id": str}, low_memory=False)
    previous = prior[KEY + ["project_label", "model_text"]].fillna("")
    current = train[KEY + ["project_label", "model_text"]]
    if not previous.sort_values(KEY).reset_index(drop=True).equals(
        current.sort_values(KEY).reset_index(drop=True)
    ):
        print("Training text or labels changed; retraining validation models.")
        return None
    merged = old[KEY + ["model", "predicted_label", "score"]].merge(
        val[KEY + ["project_label", GROUP]], on=KEY, how="inner",
        validate="many_to_one",
    )
    if len(merged) != len(val) * len(MODELS):
        raise ValueError("Verified validation predictions have an unexpected row count")
    print("Reusing Script 20 predictions for all corrected validation records.")
    return merged


def controlled_validation(train, val):
    """Update Script 20. Reuse predictions if possible, otherwise refit."""
    print("\n===== STEP 1: UPDATED CONTROLLED VALIDATION =====")
    predictions = reuse_validation_predictions(train, val)
    if predictions is None:
        tfidf = new_vectorizer()
        xtrain = tfidf.fit_transform(train["model_text"])
        xval = tfidf.transform(val["model_text"])
        rows = []
        for name, model in new_models().items():
            print(f"Fitting {name} on unchanged label-checked training emails...")
            model.fit(xtrain, train["project_label"].to_numpy())
            pred = model.predict(xval)
            score = model_scores(model, name, xval)
            rows.append(make_prediction_rows(val, name, pred, score))
        predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(DATA / "23_controlled_validation_predictions_v2_PRIVATE.csv", index=False)

    overall, by_source = [], []
    for name in MODELS:
        subset = predictions.loc[predictions["model"] == name]
        results = evaluate(subset["project_label"], subset["predicted_label"], subset["score"])
        overall.append({"model": name, "dataset": "corrected_validation_v2", "emails": len(subset), **results})
        for source in SOURCES:
            item = subset.loc[subset["source_id"] == source]
            metrics = evaluate(item["project_label"], item["predicted_label"], item["score"])
            if metrics is None:
                raise ValueError(f"Validation source {source} lacks one class")
            by_source.append({"model": name, "source": source, "emails": len(item), **metrics})
    pd.DataFrame(overall).to_csv(RESULTS / "23_controlled_validation_metrics_v2.csv", index=False)
    pd.DataFrame(by_source).to_csv(RESULTS / "23_controlled_validation_by_source_v2.csv", index=False)
    print(pd.DataFrame(overall)[["model", "emails", "accuracy", "f1", "average_precision"]].to_string(index=False))


def loso(train, val):
    """Rerun Script 21 using the corrected pool and v2 family IDs."""
    print("\n===== STEP 2: UPDATED LOSO TESTING =====")
    pool = pd.concat([train, val], ignore_index=True)
    if pool.duplicated(KEY).any():
        raise ValueError("Repeated record in LOSO pool")
    conflicts = pool.groupby(GROUP)["project_label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Conflicting-label family in LOSO pool; inspect it before fitting")
    metrics_rows, checks, predictions = [], [], []
    for heldout in SOURCES:
        unseen = pool.loc[pool["source_id"] == heldout].copy()
        other = pool.loc[pool["source_id"] != heldout].copy()
        shared = set(unseen[GROUP])
        removed = int(other[GROUP].isin(shared).sum())
        fitting = other.loc[~other[GROUP].isin(shared)].copy()
        if not len(unseen) or not len(fitting):
            raise ValueError(f"Empty LOSO split for {heldout}")
        if set(unseen[GROUP]) & set(fitting[GROUP]):
            raise RuntimeError(f"Detected family leakage in LOSO fold {heldout}")
        if fitting["project_label"].nunique() != 2 or unseen["project_label"].nunique() != 2:
            raise ValueError(f"Both labels required in LOSO fold {heldout}")
        print(f"\nHeld out: {heldout} | train={len(fitting):,}, test={len(unseen):,}, "
              f"shared-family training records removed={removed:,}")
        vec = new_vectorizer()
        xtrain = vec.fit_transform(fitting["model_text"])
        xunseen = vec.transform(unseen["model_text"])
        checks.append({
            "held_out_source": heldout, "training_emails": len(fitting),
            "heldout_emails": len(unseen), "removed_shared_family_records": removed,
            "source_overlap": 0, "detected_family_overlap": 0,
            "tfidf_features": xtrain.shape[1],
        })
        for name, model in new_models().items():
            print(f"  Fitting {name}...")
            model.fit(xtrain, fitting["project_label"].to_numpy())
            pred = model.predict(xunseen)
            score = model_scores(model, name, xunseen)
            result = evaluate(unseen["project_label"], pred, score)
            metrics_rows.append({
                "model": name, "held_out_source": heldout,
                "training_emails": len(fitting), "heldout_emails": len(unseen),
                "heldout_spam_count": int(unseen["project_label"].sum()),
                "heldout_spam_fraction": float(unseen["project_label"].mean()),
                **result,
            })
            predictions.append(make_prediction_rows(unseen, name, pred, score))
            print(f"  F1={result['f1']:.4f}, PR-AUC={result['pr_auc_trapezoid']:.4f}")
    met = pd.DataFrame(metrics_rows)
    preds = pd.concat(predictions, ignore_index=True)
    met.to_csv(RESULTS / "23_loso_model_metrics_v2.csv", index=False)
    pd.DataFrame(checks).to_csv(RESULTS / "23_loso_leakage_checks_v2.csv", index=False)
    preds.to_csv(DATA / "23_loso_predictions_v2_PRIVATE.csv", index=False)
    return met, preds


def loso_intervals(preds):
    """Rerun Script 22's 300-replicate family-cluster bootstrap."""
    print("\n===== STEP 3: UPDATED 95% CONFIDENCE INTERVALS =====")
    rows, report_rows, check_rows = [], [], []
    for source_idx, source in enumerate(SOURCES):
        for name in MODELS:
            part = preds.loc[
                (preds["source_id"] == source) & (preds["model"] == name)
            ]
            y = part["project_label"].to_numpy(dtype=int)
            pred = part["predicted_label"].to_numpy(dtype=int)
            scores = part["score"].to_numpy(dtype=float)
            codes, groups = pd.factorize(part[GROUP], sort=True)
            n_groups = len(groups)
            if n_groups == 0:
                raise ValueError(f"No families for {source}/{name}")
            observed = evaluate(y, pred, scores)
            rng = np.random.default_rng(SEED + source_idx)
            samples = {metric: [] for metric in METRICS}
            skipped = 0
            for _ in range(N_BOOTSTRAPS):
                drawn = rng.integers(0, n_groups, size=n_groups)
                counts = np.bincount(drawn, minlength=n_groups)
                result = evaluate(y, pred, scores, weights=counts[codes])
                if result is None:
                    skipped += 1
                    continue
                for metric in METRICS:
                    samples[metric].append(result[metric])
            valid = N_BOOTSTRAPS - skipped
            if valid < 0.8 * N_BOOTSTRAPS:
                raise ValueError(f"Too many invalid bootstraps for {source}/{name}")
            report = {"held_out_source": source, "model": name,
                      "emails": len(part), "duplicate_families": n_groups}
            for metric in METRICS:
                low, high = np.percentile(samples[metric], [2.5, 97.5])
                rows.append({
                    "held_out_source": source, "model": name, "metric": metric,
                    "estimate": observed[metric], "ci_lower": low, "ci_upper": high,
                    "confidence_level": 95, "valid_bootstrap_samples": valid,
                    "emails": len(part), "duplicate_families": n_groups,
                })
                report[metric] = f"{observed[metric]:.4f} [{low:.4f}, {high:.4f}]"
            report_rows.append(report)
            check_rows.append({
                "held_out_source": source, "model": name, "emails": len(part),
                "duplicate_families": n_groups, "requested_bootstraps": N_BOOTSTRAPS,
                "valid_bootstraps": valid, "skipped_bootstraps": skipped,
                "resampling_unit": GROUP,
            })
            f1_l, f1_u = np.percentile(samples["f1"], [2.5, 97.5])
            print(f"{source:13s} | {name:20s} F1={observed['f1']:.4f} "
                  f"[{f1_l:.4f}, {f1_u:.4f}]")
    pd.DataFrame(rows).to_csv(RESULTS / "23_loso_confidence_intervals_v2.csv", index=False)
    pd.DataFrame(report_rows).to_csv(
        RESULTS / "23_loso_confidence_interval_report_table_v2.csv", index=False
    )
    pd.DataFrame(check_rows).to_csv(RESULTS / "23_loso_bootstrap_checks_v2.csv", index=False)


def compare_old_new(new_metrics):
    print("\n===== STEP 4: OLD VS CORRECTED LOSO =====")
    if not OLD_LOSO_METRICS.exists():
        print("Old LOSO metrics not found; skipping comparison (new results saved).")
        return
    old = pd.read_csv(OLD_LOSO_METRICS)
    required = {"held_out_source", "model", "accuracy", "f1", "average_precision"}
    if not required.issubset(old.columns):
        print("Old LOSO metrics use different columns; skipping old/new comparison.")
        return
    joined = old.merge(new_metrics, on=["held_out_source", "model"],
                       suffixes=("_old", "_v2"), validate="one_to_one")
    if len(joined) != len(new_metrics):
        raise ValueError("Old and corrected LOSO model/source combinations differ")
    for metric in ["accuracy", "f1", "average_precision"]:
        joined[f"{metric}_change_pp"] = 100 * (
            joined[f"{metric}_v2"] - joined[f"{metric}_old"]
        )
    joined.to_csv(RESULTS / "23_old_vs_new_loso_comparison.csv", index=False)
    print(joined[["held_out_source", "model", "f1_old", "f1_v2", "f1_change_pp"]]
          .to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main():
    print("Loading corrected Script 22e datasets...")
    train = load_split(TRAIN_FILE, "checked training v2")
    val = load_split(VAL_FILE, "validation v2")
    test = load_split(TEST_FILE, "reserved test v2")
    print(f"Train={len(train):,} | Validation={len(val):,} | Reserved test={len(test):,}")
    check_splits(train, val, test)
    controlled_validation(train, val)
    updated, preds = loso(train, val)
    loso_intervals(preds)
    compare_old_new(updated)
    pd.DataFrame([
        {"item": "training_emails", "value": len(train)},
        {"item": "validation_emails", "value": len(val)},
        {"item": "reserved_test_emails_not_evaluated", "value": len(test)},
        {"item": "loso_pool_emails", "value": len(train) + len(val)},
        {"item": "bootstrap_replicates_per_fold_and_model", "value": N_BOOTSTRAPS},
        {"item": "source_count", "value": len(SOURCES)},
    ]).to_csv(RESULTS / "23_evaluation_update_summary.csv", index=False)
    print("\nDONE. New files begin with '23_' in data/ and results/.")
    print("Previous results were not overwritten.")
    print("Reserved test emails were used ONLY for overlap checks, NOT model evaluation.")
    print("Near-duplicate detection remains approximate; zero detected overlap is not proof of zero leakage.")


if __name__ == "__main__":
    main()
