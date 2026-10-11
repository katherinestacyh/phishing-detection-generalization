
from pathlib import Path
import pandas as pd

sources = [
    "CEAS_08",
    "Nazario",
    "Nigerian_Fraud",
    "SpamAssasin"
]

for source in sources:
    df = pd.read_csv(Path("data") / f"{source}.csv")

    print(f"\n{'=' * 50}")
    print(f"SOURCE: {source}")

    print("\nURL FIELD VALUES:")
    print(df["urls"].value_counts(dropna=False).to_string())

    print("\nSAMPLE ORIGINAL DATE FORMATS:")
    print(df["date"].dropna().head(5).to_string(index=False))

    print("\nSAMPLE DATES NOT PARSED BY INITIAL METHOD:")
    dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
    failed = df.loc[df["date"].notna() & dates.isna(), "date"]
    print(f"Non-missing dates that failed parsing: {len(failed):,}")
    print(failed.head(5).to_string(index=False))
