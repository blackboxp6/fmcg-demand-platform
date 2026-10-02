from pathlib import Path

import pandas as pd


SILVER_PATH = Path(
    "data/silver/sales_clean.parquet"
)

GOLD_PATH = Path(
    "data/gold/forecast_features.parquet"
)


def load_data():

    print("Loading Silver dataset...")

    return pd.read_parquet(
        SILVER_PATH
    )


def add_calendar_features(df):

    df = df.copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek

    df["week_of_year"] = (
        df["date"]
        .dt
        .isocalendar()
        .week
        .astype(int)
    )

    df["quarter"] = (
        df["date"].dt.quarter
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df


def add_lag_features(df):

    print("Creating lag features...")

    df = df.sort_values(
        [
            "store_nbr",
            "family",
            "date",
        ]
    ).copy()

    group = df.groupby(
        [
            "store_nbr",
            "family",
        ]
    )["sales"]

    df["sales_lag_1"] = (
        group.shift(1)
    )

    df["sales_lag_7"] = (
        group.shift(7)
    )

    df["sales_lag_14"] = (
        group.shift(14)
    )

    df["sales_lag_28"] = (
        group.shift(28)
    )

    return df


def add_rolling_features(df):

    print("Creating rolling features...")

    grouped = df.groupby(
        [
            "store_nbr",
            "family",
        ]
    )["sales"]

    # Shift BEFORE rolling.
    #
    # This prevents today's sales from
    # leaking into today's features.

    df["rolling_mean_7"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
            .rolling(7)
            .mean()
        )
    )

    df["rolling_mean_14"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
            .rolling(14)
            .mean()
        )
    )

    df["rolling_mean_28"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
            .rolling(28)
            .mean()
        )
    )

    df["rolling_std_7"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
            .rolling(7)
            .std()
        )
    )

    return df


def main():

    GOLD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    print(
        f"Silver rows: {len(df):,}"
    )

    df = add_calendar_features(df)

    df = add_lag_features(df)

    df = add_rolling_features(df)

    # Remove rows where insufficient
    # historical information exists.
    df = df.dropna(
        subset=[
            "sales_lag_28",
            "rolling_mean_28",
        ]
    )

    df.to_parquet(
        GOLD_PATH,
        index=False,
    )

    print()
    print("Gold dataset created.")

    print(
        f"Rows: {len(df):,}"
    )

    print()
    print(
        df[
            [
                "date",
                "store_nbr",
                "family",
                "sales",
                "sales_lag_1",
                "sales_lag_7",
                "rolling_mean_7",
                "rolling_mean_28",
            ]
        ].head(20)
    )


if __name__ == "__main__":
    main()