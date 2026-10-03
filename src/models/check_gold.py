from pathlib import Path

import pandas as pd


GOLD_PATH = Path(
    "data/gold/forecast_features.parquet"
)


def main():

    print("Loading Gold dataset...")

    df = pd.read_parquet(GOLD_PATH)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDate range:")
    print(
        df["date"].min(),
        "→",
        df["date"].max()
    )

    print("\nNumber of stores:")
    print(df["store_nbr"].nunique())

    print("\nNumber of product families:")
    print(df["family"].nunique())

    print("\nSample:")
    print(df.head())

    print("\nMissing values:")
    print(
        df.isnull()
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )


if __name__ == "__main__":
    main()