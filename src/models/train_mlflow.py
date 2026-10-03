from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

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


PARAMS = {

    "objective":
        "regression",

    "n_estimators":
        1000,

    "learning_rate":
        0.05,

    "num_leaves":
        31,

    "subsample":
        0.8,

    "colsample_bytree":
        0.8,

    "random_state":
        42,

    "n_jobs":
        -1,
}


def prepare_data(df):

    df = df.copy()

    for column in (
        CATEGORICAL_FEATURES
    ):

        df[column] = (
            df[column]
            .astype("category")
        )

    return df


def main():

    print(
        "Loading model datasets..."
    )

    train = pd.read_parquet(
        TRAIN_PATH
    )

    valid = pd.read_parquet(
        VALID_PATH
    )

    train = prepare_data(train)
    valid = prepare_data(valid)

    X_train = train[FEATURES]
    y_train = train["sales"]

    X_valid = valid[FEATURES]
    y_valid = valid["sales"]

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    mlflow.set_experiment(
        "FMCG Demand Forecasting"
    )

    with mlflow.start_run(
        run_name="lightgbm_v1"
    ):

        print()
        print(
            "Starting MLflow run..."
        )

        model = (
            lgb.LGBMRegressor(
                **PARAMS
            )
        )

        model.fit(

            X_train,
            y_train,

            categorical_feature=
            CATEGORICAL_FEATURES,

            eval_set=[
                (
                    X_valid,
                    y_valid
                )
            ],

            callbacks=[

                lgb.early_stopping(
                    stopping_rounds=50
                ),

                lgb.log_evaluation(
                    period=50
                ),
            ],
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

        # ------------------
        # Log parameters
        # ------------------

        mlflow.log_params(
            PARAMS
        )

        # ------------------
        # Log metrics
        # ------------------

        for name, value in (
            metrics.items()
        ):

            mlflow.log_metric(
                name.lower(),
                value
            )

        # ------------------
        # Additional info
        # ------------------

        mlflow.log_param(
            "feature_count",
            len(FEATURES)
        )

        mlflow.log_param(
            "validation_days",
            15
        )

        mlflow.log_param(
            "training_rows",
            len(train)
        )

        # ------------------
        # Log model
        # ------------------

        mlflow.lightgbm.log_model(
            model,
            name="model"
        )

        # Also save local model

        Path("models").mkdir(
            exist_ok=True
        )

        joblib.dump(
            model,
            "models/"
            "lightgbm_mlflow.joblib"
        )

        print()
        print(
            "MODEL RESULTS"
        )

        print(
            "-------------"
        )

        for name, value in (
            metrics.items()
        ):

            print(
                f"{name}: "
                f"{value:.4f}"
            )

        print()
        print(
            "Experiment logged "
            "to MLflow."
        )


if __name__ == "__main__":
    main()