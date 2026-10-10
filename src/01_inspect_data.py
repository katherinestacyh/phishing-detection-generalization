
from pathlib import Path
import pandas as pd

# Find our project and data folders
project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"

# Find all CSV files
csv_files = sorted(data_folder.glob("*.csv"))

print("Total CSV files:", len(csv_files))

for file in csv_files:
    print("\n" + "=" * 50)
    print("File:", file.name)

    # Load the CSV file
    df = pd.read_csv(file)

    # Basic information
    print("Number of emails:", len(df))
    print("Number of columns:", len(df.columns))
    print("Column names:", df.columns.tolist())

    # Check column types
    print("\nData types:")
    print(df.dtypes.to_string())

    # Check missing values
    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    # Check original labels
    for column in df.columns:
        if column.lower() in ["label", "labels", "class"]:
            print("\nOriginal label distribution:")
            print(df[column].value_counts(dropna=False).to_string())


# Create a summary of the six original sources
summary = []

for file in csv_files:
    if file.name == "phishing_email.csv":
        continue

    df = pd.read_csv(file)

    summary.append({
        "Source": file.stem,
        "Emails": len(df),
        "Columns": ", ".join(df.columns),
        "Missing values": int(df.isnull().sum().sum()),
        "Label 0": int((df["label"] == 0).sum()),
        "Label 1": int((df["label"] == 1).sum())
    })

summary_df = pd.DataFrame(summary)

print("\nSOURCE INVENTORY")
print(summary_df.to_string(index=False))

# Save a summary without exposing email content
output_folder = project_folder / "results"
output_folder.mkdir(exist_ok=True)

summary_df.to_csv(
    output_folder / "source_inventory.csv",
    index=False
)
