from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb

from metrics import evaluate_forecast


TRAIN_PATH = Path(
    "data/gold/model_data/train.parquet"
)

VALID_PATH = Path(
    "data/gold/model_data/validation.parquet"
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


def encode_categories(
    train,
    valid
):

    train = train.copy()
    valid = valid.copy()

    for column in (
        CATEGORICAL_FEATURES
    ):

        categories = (
            train[column]
            .astype("category")
            .cat.categories
        )

        train[column] = (
            pd.Categorical(
                train[column],
                categories=categories
            )
            .codes
        )

        valid[column] = (
            pd.Categorical(
                valid[column],
                categories=categories
            )
            .codes
        )

    return train, valid


def main():

    print(
        "Loading datasets..."
    )

    train = pd.read_parquet(
        TRAIN_PATH
    )

    valid = pd.read_parquet(
        VALID_PATH
    )

    train, valid = (
        encode_categories(
            train,
            valid
        )
    )

    X_train = train[FEATURES]
    y_train = train["sales"]

    X_valid = valid[FEATURES]
    y_valid = valid["sales"]

    params = {

        "n_estimators":
            800,

        "learning_rate":
            0.05,

        "max_depth":
            8,

        "subsample":
            0.8,

        "colsample_bytree":
            0.8,

        "objective":
            "reg:squarederror",

        "tree_method":
            "hist",

        "random_state":
            42,

        "n_jobs":
            -1,
    }

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    mlflow.set_experiment(
        "FMCG Demand Forecasting"
    )

    with mlflow.start_run(
        run_name="xgboost_v1"
    ):

        print(
            "Training XGBoost..."
        )

        model = (
            xgb.XGBRegressor(
                **params
            )
        )

        model.fit(
            X_train,
            y_train,

            eval_set=[
                (
                    X_valid,
                    y_valid
                )
            ],

            verbose=100
        )

        predictions = (
            model.predict(
                X_valid
            )
        )

        predictions = np.clip(
            predictions,
            0,
            None
        )

        metrics = (
            evaluate_forecast(
                y_valid,
                predictions
            )
        )

        mlflow.log_params(
            params
        )

        for name, value in (
            metrics.items()
        ):

            mlflow.log_metric(
                name.lower(),
                value
            )

        mlflow.xgboost.log_model(
            model,
            name="model"
        )

        print()

        for name, value in (
            metrics.items()
        ):

            print(
                f"{name}: "
                f"{value:.4f}"
            )


if __name__ == "__main__":
    main()