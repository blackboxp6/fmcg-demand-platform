from pathlib import Path

import pandas as pd


BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")


def load_data():

    print("Loading Bronze datasets...")

    sales = pd.read_parquet(
        BRONZE_DIR / "train.parquet"
    )

    stores = pd.read_parquet(
        BRONZE_DIR / "stores.parquet"
    )

    oil = pd.read_parquet(
        BRONZE_DIR / "oil.parquet"
    )

    transactions = pd.read_parquet(
        BRONZE_DIR / "transactions.parquet"
    )

    return sales, stores, oil, transactions


def clean_sales(df):

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["family"] = (
        df["family"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["sales"] = pd.to_numeric(
        df["sales"],
        errors="coerce",
    )

    df["onpromotion"] = pd.to_numeric(
        df["onpromotion"],
        errors="coerce",
    ).fillna(0)

    return df


def merge_datasets(
    sales,
    stores,
    oil,
    transactions,
):

    stores = stores.copy()
    oil = oil.copy()
    transactions = transactions.copy()

    oil["date"] = pd.to_datetime(
        oil["date"]
    )

    transactions["date"] = pd.to_datetime(
        transactions["date"]
    )

    # Add store metadata
    df = sales.merge(
        stores,
        on="store_nbr",
        how="left",
    )

    # Add oil price
    df = df.merge(
        oil,
        on="date",
        how="left",
    )

    # Add store transaction volume
    df = df.merge(
        transactions,
        on=["date", "store_nbr"],
        how="left",
    )

    return df


def clean_merged_data(df):

    df = df.copy()

    # Rename oil price
    df = df.rename(
        columns={
            "dcoilwtico": "oil_price"
        }
    )

    # Oil data contains gaps.
    df["oil_price"] = (
        df["oil_price"]
        .ffill()
        .bfill()
    )

    # Transactions can be unavailable
    df["transactions"] = (
        df["transactions"]
        .fillna(0)
    )

    return df


def validate(df):

    if df.empty:
        raise ValueError(
            "Silver dataset is empty."
        )

    if df["date"].isna().any():
        raise ValueError(
            "Invalid dates found."
        )

    if df["sales"].isna().any():
        raise ValueError(
            "Missing sales values found."
        )

    if (df["sales"] < 0).any():
        raise ValueError(
            "Negative sales found."
        )


def main():

    SILVER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sales, stores, oil, transactions = load_data()

    print("Cleaning sales...")

    sales = clean_sales(sales)

    print("Joining datasets...")

    df = merge_datasets(
        sales,
        stores,
        oil,
        transactions,
    )

    print("Cleaning merged dataset...")

    df = clean_merged_data(df)

    print("Validating...")

    validate(df)

    output = (
        SILVER_DIR /
        "sales_clean.parquet"
    )

    df.to_parquet(
        output,
        index=False,
    )

    print()
    print("Silver dataset created.")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print()
    print(df.head())


if __name__ == "__main__":
    main()