
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

INPUT = DATA / "spam_near_duplicate_master_v2.csv"
OUTPUT = DATA / "manual_review_private"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Emails selected from Script 24's error analysis
CASES = [
    ("CEAS_08", 7084, "SVM false negative"),
    ("CEAS_08", 23006, "LR false negative"),
    ("Enron", 25062, "LR false positive"),
    ("Ling", 12, "Both models false positive"),
    ("Enron", 10299, "SVM false negative")
]


def clean(value):
    if pd.isna(value):
        return ""
    return str(value)


def main():

    print("Loading master dataset...")

    df = pd.read_csv(INPUT, low_memory=False)

    df["source_id"] = df["source_id"].astype(str).str.strip()
    df["source_row"] = pd.to_numeric(
        df["source_row"], errors="coerce"
    )

    review_rows = []

    for source, row_number, error in CASES:

        match = df[
            (df["source_id"] == source)
            & (df["source_row"] == row_number)
        ]

        if len(match) != 1:
            raise ValueError(
                f"Expected one record for {source} "
                f"row {row_number}, found {len(match)}"
            )

        email = match.iloc[0]
        label = int(email["project_label"])

        subject = clean(email.get("subject", ""))
        body = clean(email.get("body", ""))
        model_text = clean(email.get("model_text", ""))

        # Prefer subject and body if they were retained.
        # Otherwise use the text given to the model.
        if subject.strip() or body.strip():
            full_text = (
                f"SUBJECT:\n{subject}\n\n"
                f"BODY:\n{body}"
            )
        else:
            full_text = f"MODEL TEXT:\n{model_text}"

        filename = f"{source}_{row_number}_PRIVATE.txt"
        output_path = OUTPUT / filename

        content = (
            f"Source: {source}\n"
            f"Source row: {row_number}\n"
            f"Original label: "
            f"{'Spam' if label == 1 else 'Legitimate'}\n"
            f"Model error: {error}\n"
            f"\n{'=' * 55}\n\n"
            f"{full_text}\n"
        )

        output_path.write_text(
            content,
            encoding="utf-8"
        )

        review_rows.append({
            "source_id": source,
            "source_row": row_number,
            "original_label": (
                "Spam" if label == 1 else "Legitimate"
            ),
            "model_error": error,
            "label_review": "",
            "reason_for_decision": "",
            "possible_model_issue": ""
        })

        print(f"Saved: {filename}")

    # Create a blank table for our manual decisions.
    pd.DataFrame(review_rows).to_csv(
        OUTPUT / "manual_label_review_PRIVATE.csv",
        index=False
    )

    print("\nDone!")
    print("All five emails were extracted.")
    print("Original dataset was not modified.")
    print(
        "Review files are in: "
        "data/manual_review_private/"
    )


if __name__ == "__main__":
    main()
