from pathlib import Path

import pandas as pd


GOLD_PATH = Path(
    "data/gold/forecast_features.parquet"
)

OUTPUT_DIR = Path(
    "data/gold/model_data"
)


def load_data():

    print("Loading Gold dataset...")

    df = pd.read_parquet(GOLD_PATH)

    df["date"] = pd.to_datetime(df["date"])

    return df


def create_time_split(
    df,
    validation_days=15
):

    max_date = df["date"].max()

    validation_start = (
        max_date
        - pd.Timedelta(days=validation_days - 1)
    )

    print()
    print("Last available date:")
    print(max_date)

    print()
    print("Validation starts:")
    print(validation_start)

    train = df[
        df["date"] < validation_start
    ].copy()

    validation = df[
        df["date"] >= validation_start
    ].copy()

    return train, validation


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    train, validation = create_time_split(df)

    print()
    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print()
    print("Training period:")

    print(
        train["date"].min(),
        "→",
        train["date"].max()
    )

    print()
    print("Validation period:")

    print(
        validation["date"].min(),
        "→",
        validation["date"].max()
    )

    train.to_parquet(
        OUTPUT_DIR / "train.parquet",
        index=False
    )

    validation.to_parquet(
        OUTPUT_DIR / "validation.parquet",
        index=False
    )

    print()
    print("Time split saved.")


if __name__ == "__main__":
    main()