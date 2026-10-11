
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

sources = [
    "CEAS_08",
    "Enron",
    "Ling",
    "Nazario",
    "Nigerian_Fraud",
    "SpamAssasin"
]

for source in sources:
    df = pd.read_csv(DATA_DIR / f"{source}.csv", low_memory=False)

    print(f"\n{'=' * 55}")
    print(f"SOURCE: {source}")
    print(f"Records: {len(df):,}")

    print("\nFIELDS AND DATA TYPES:")
    print(df.dtypes.to_string())

    print("\nMISSING VALUES BY FIELD:")
    print(df.isna().sum().to_string())

    print("\nORIGINAL LABEL DISTRIBUTION:")
    print(df["label"].value_counts(dropna=False).to_string())

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
        valid_dates = dates.dropna()

        print("\nDATE COVERAGE:")
        print(f"Valid dates: {len(valid_dates):,}")
        print(f"Earliest: {valid_dates.min() if len(valid_dates) else 'N/A'}")
        print(f"Latest: {valid_dates.max() if len(valid_dates) else 'N/A'}")
    else:
        print("\nDATE COVERAGE: No date column")
