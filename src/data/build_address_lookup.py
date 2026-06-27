from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/raw/property-tax-report.csv")
OUTPUT_PATH = Path("data/deploy/address_lookup.parquet")

KEEP_COLS = [
    "PID",
    "REPORT_YEAR",
    "PROPERTY_POSTAL_CODE",
    "FROM_CIVIC_NUMBER",
    "TO_CIVIC_NUMBER",
    "STREET_NAME",
    "LEGAL_TYPE",
    "ZONING_DISTRICT",
    "ZONING_CLASSIFICATION",
    "NEIGHBOURHOOD_CODE",
    "YEAR_BUILT",
    "BIG_IMPROVEMENT_YEAR",
]


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(
        INPUT_PATH,
        sep=";",
        low_memory=False,
        usecols=lambda c: c in KEEP_COLS,
    ).copy()

    df["REPORT_YEAR"] = pd.to_numeric(df["REPORT_YEAR"], errors="coerce")
    df = df[(df["REPORT_YEAR"] >= 2024) & (df["REPORT_YEAR"] <= 2026)].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved address lookup parquet: {OUTPUT_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Cols: {len(df.columns)}")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()