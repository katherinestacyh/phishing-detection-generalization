
from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix
)

# Project folders
project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)

# Load training and validation data.
# Do NOT use the test dataset yet.
train = pd.read_csv(data_dir / "train_random.csv")
validation = pd.read_csv(data_dir / "validation_random.csv")

X_train = train["model_text"].fillna("").astype(str)
y_train = train["project_label"].astype(int)

X_val = validation["model_text"].fillna("").astype(str)
y_val = validation["project_label"].astype(int)

print("\n===== DATASET =====")
print(f"Training emails: {len(train):,}")
print(f"Validation emails: {len(validation):,}")

# Create a pipeline.
# TF-IDF will learn its vocabulary from training data ONLY.
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=30000,
        min_df=3,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=42
    ))
])

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Training complete!")

# Evaluate only on the validation set.
predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_val, predictions),
    "Precision": precision_score(y_val, predictions, zero_division=0),
    "Recall": recall_score(y_val, predictions, zero_division=0),
    "F1": f1_score(y_val, predictions, zero_division=0),
    "PR-AUC": average_precision_score(y_val, probabilities)
}

print("\n===== VALIDATION RESULTS =====")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# Confusion matrix:
# TN = legitimate correctly identified
# FP = legitimate incorrectly flagged as spam
# FN = spam incorrectly classified as legitimate
# TP = spam correctly identified
tn, fp, fn, tp = confusion_matrix(
    y_val, predictions, labels=[0, 1]
).ravel()

print("\n===== CONFUSION MATRIX =====")
print(f"True Negatives: {tn:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")
print(f"True Positives: {tp:,}")

# Save aggregate results.
pd.DataFrame([
    {"metric": name, "value": value}
    for name, value in metrics.items()
]).to_csv(
    results_dir / "logistic_regression_baseline_metrics.csv",
    index=False
)

pd.DataFrame([{
    "true_negatives": tn,
    "false_positives": fp,
    "false_negatives": fn,
    "true_positives": tp
}]).to_csv(
    results_dir / "logistic_regression_confusion_matrix.csv",
    index=False
)

print("\n===== FILES SAVED =====")
print("results/logistic_regression_baseline_metrics.csv")
print("results/logistic_regression_confusion_matrix.csv")
