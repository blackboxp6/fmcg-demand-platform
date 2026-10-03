from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from metrics import evaluate_forecast


GOLD_PATH = Path(
    "data/gold/forecast_features.parquet"
)


FEATURES = [
    "store_nbr",
    "family",

    "onpromotion",
    "oil_price",

    "city",
    "state",
    "type",
    "cluster",

    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "quarter",
    "is_weekend",

    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",

    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
]


CATEGORICAL_FEATURES = [
    "family",
    "city",
    "state",
    "type",
]


def prepare_data(df):

    df = df.copy()

    for column in CATEGORICAL_FEATURES:

        df[column] = (
            df[column]
            .astype("category")
        )

    return df


def create_model():

    return lgb.LGBMRegressor(
        objective="regression",

        n_estimators=500,

        learning_rate=0.05,

        num_leaves=31,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        n_jobs=-1,
    )


def main():

    print("Loading Gold dataset...")

    df = pd.read_parquet(
        GOLD_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values("date")

    max_date = df["date"].max()

    print()
    print(
        "Dataset ends:",
        max_date
    )

    # Three historical validation windows.
    #
    # Each validation period is 15 days.

    folds = [

        (
            max_date
            - pd.Timedelta(days=44),

            max_date
            - pd.Timedelta(days=30)
        ),

        (
            max_date
            - pd.Timedelta(days=29),

            max_date
            - pd.Timedelta(days=15)
        ),

        (
            max_date
            - pd.Timedelta(days=14),

            max_date
        ),
    ]

    results = []

    for fold_number, (
        valid_start,
        valid_end
    ) in enumerate(
        folds,
        start=1
    ):

        print()
        print(
            "=" * 60
        )

        print(
            f"FOLD {fold_number}"
        )

        print(
            "=" * 60
        )

        print(
            "Validation:",
            valid_start.date(),
            "→",
            valid_end.date()
        )

        train = df[
            df["date"] < valid_start
        ].copy()

        valid = df[
            (
                df["date"] >= valid_start
            )
            &
            (
                df["date"] <= valid_end
            )
        ].copy()

        print(
            f"Train rows: "
            f"{len(train):,}"
        )

        print(
            f"Validation rows: "
            f"{len(valid):,}"
        )

        train = prepare_data(train)
        valid = prepare_data(valid)

        X_train = train[FEATURES]
        y_train = train["sales"]

        X_valid = valid[FEATURES]
        y_valid = valid["sales"]

        # -------------------------
        # Seasonal naive baseline
        # -------------------------

        baseline_prediction = (
            valid["sales_lag_7"]
            .clip(lower=0)
        )

        baseline_metrics = (
            evaluate_forecast(
                y_valid,
                baseline_prediction
            )
        )

        # -------------------------
        # LightGBM
        # -------------------------

        model = create_model()

        model.fit(
            X_train,
            y_train,

            categorical_feature=
            CATEGORICAL_FEATURES
        )

        predictions = model.predict(
            X_valid
        )

        predictions = np.clip(
            predictions,
            0,
            None
        )

        model_metrics = (
            evaluate_forecast(
                y_valid,
                predictions
            )
        )

        print()
        print("Seasonal Naive:")

        for metric, value in (
            baseline_metrics.items()
        ):

            print(
                f"  {metric}: "
                f"{value:.4f}"
            )

        print()
        print("LightGBM:")

        for metric, value in (
            model_metrics.items()
        ):

            print(
                f"  {metric}: "
                f"{value:.4f}"
            )

        results.append(
            {
                "fold": fold_number,

                "validation_start":
                    valid_start,

                "validation_end":
                    valid_end,

                "baseline_rmsle":
                    baseline_metrics[
                        "RMSLE"
                    ],

                "baseline_mae":
                    baseline_metrics[
                        "MAE"
                    ],

                "baseline_wmape":
                    baseline_metrics[
                        "WMAPE"
                    ],

                "lightgbm_rmsle":
                    model_metrics[
                        "RMSLE"
                    ],

                "lightgbm_mae":
                    model_metrics[
                        "MAE"
                    ],

                "lightgbm_wmape":
                    model_metrics[
                        "WMAPE"
                    ],
            }
        )

    results_df = pd.DataFrame(
        results
    )

    print()
    print(
        "=" * 60
    )

    print(
        "BACKTEST SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        "AVERAGE PERFORMANCE"
    )

    print()

    numeric_columns = [
        "baseline_rmsle",
        "baseline_mae",
        "baseline_wmape",

        "lightgbm_rmsle",
        "lightgbm_mae",
        "lightgbm_wmape",
    ]

    print(
        results_df[
            numeric_columns
        ]
        .mean()
    )

    output_path = Path(
        "models/backtest_results.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        f"Results saved → "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()