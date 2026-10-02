from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
BRONZE_DIR = Path("data/bronze")


FILES = [
    "train.csv",
    "test.csv",
    "stores.csv",
    "oil.csv",
    "holidays_events.csv",
    "transactions.csv",
]


def ingest_file(filename: str) -> None:

    input_path = RAW_DIR / filename

    output_name = filename.replace(".csv", ".parquet")
    output_path = BRONZE_DIR / output_name

    print(f"Reading {input_path}...")

    df = pd.read_csv(input_path)

    print(
        f"{filename}: "
        f"{len(df):,} rows × {len(df.columns)} columns"
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved → {output_path}")
    print()


def main():

    BRONZE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename in FILES:

        input_path = RAW_DIR / filename

        if not input_path.exists():

            print(
                f"WARNING: {input_path} not found."
            )

            continue

        ingest_file(filename)

    print("Bronze ingestion complete.")


if __name__ == "__main__":
    main()