from pathlib import Path

import pandas as pd

from metrics import evaluate_forecast


VALIDATION_PATH = Path(
    "data/gold/model_data/"
    "validation.parquet"
)


def main():

    print(
        "Loading validation data..."
    )

    df = pd.read_parquet(
        VALIDATION_PATH
    )

    # Seasonal naive:
    #
    # predicted sales today =
    # sales 7 days ago

    df["prediction"] = (
        df["sales_lag_7"]
    )

    # Ensure non-negative
    df["prediction"] = (
        df["prediction"]
        .clip(lower=0)
    )

    metrics = evaluate_forecast(
        df["sales"],
        df["prediction"]
    )

    print()
    print(
        "SEASONAL NAIVE BASELINE"
    )

    print(
        "-----------------------"
    )

    for name, value in metrics.items():

        print(
            f"{name}: "
            f"{value:.4f}"
        )


if __name__ == "__main__":
    main()