
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

    # Mixed tells pandas that dates may have different formats.
    dates = pd.to_datetime(
        df["date"],
        format="mixed",
        errors="coerce",
        utc=True
    )

    years = dates.dt.year

    missing_original = df["date"].isna().sum()
    failed_parsing = (
        df["date"].notna() & dates.isna()
    ).sum()

    before_1995 = (years < 1995).sum()
    after_2026 = (years > 2026).sum()

    plausible = dates[
        (years >= 1995) & (years <= 2026)
    ]

    print(f"\n{'=' * 50}")
    print(f"SOURCE: {source}")
    print(f"Total records: {len(df):,}")
    print(f"Originally missing dates: {missing_original:,}")
    print(f"Successfully parsed dates: {dates.notna().sum():,}")
    print(f"Non-missing dates that failed parsing: {failed_parsing:,}")
    print(f"Dates before 1995: {before_1995:,}")
    print(f"Dates after 2026: {after_2026:,}")

    if not plausible.empty:
        print(f"Earliest plausible date: {plausible.min()}")
        print(f"Latest plausible date: {plausible.max()}")
