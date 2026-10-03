from pathlib import Path

import pandas as pd

from metrics import evaluate_forecast


VALIDATION_PATH = Path(
    "data/gold/model_data/"
    "validation.parquet"
)

LIGHTGBM_PATH = Path(
    "models/"
    "validation_predictions.parquet"
)


def main():

    validation = pd.read_parquet(
        VALIDATION_PATH
    )

    lightgbm = pd.read_parquet(
        LIGHTGBM_PATH
    )

    baseline_metrics = (
        evaluate_forecast(

            validation["sales"],

            validation["sales_lag_7"]
        )
    )

    lightgbm_metrics = (
        evaluate_forecast(

            lightgbm["sales"],

            lightgbm["prediction"]
        )
    )

    comparison = pd.DataFrame(
        [
            {
                "Model":
                    "Seasonal Naive",

                **baseline_metrics
            },

            {
                "Model":
                    "LightGBM",

                **lightgbm_metrics
            },
        ]
    )

    print()
    print(
        comparison.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()