
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
results_dir = project_dir / "results"

df = pd.read_csv(
    data_dir / "spam_cleaned.csv",
    low_memory=False
)

# First, split 70% for training and 30% for validation/testing.
train, remaining = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["project_label"]
)

# Split the remaining 30% equally.
validation, test = train_test_split(
    remaining,
    test_size=0.50,
    random_state=42,
    stratify=remaining["project_label"]
)

# Save all three splits locally.
train.to_csv(data_dir / "train_random.csv", index=False)
validation.to_csv(data_dir / "validation_random.csv", index=False)
test.to_csv(data_dir / "test_random.csv", index=False)

# Create a summary without email content.
summary = []

for name, subset in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:
    summary.append({
        "split": name,
        "total": len(subset),
        "legitimate": int((subset["project_label"] == 0).sum()),
        "spam": int((subset["project_label"] == 1).sum())
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    results_dir / "random_split_summary.csv",
    index=False
)

print("\n===== RANDOM DATA SPLIT =====")
print(summary_df.to_string(index=False))

print("\n===== TOTAL RECORDS =====")
print(f"Original: {len(df):,}")
print(f"Across splits: {len(train) + len(validation) + len(test):,}")

print("\n===== FILES SAVED =====")
print("data/train_random.csv")
print("data/validation_random.csv")
print("data/test_random.csv")
print("results/random_split_summary.csv")
