from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from metrics import evaluate_forecast


TRAIN_PATH = Path(
    "data/gold/model_data/train.parquet"
)

VALIDATION_PATH = Path(
    "data/gold/model_data/validation.parquet"
)

MODEL_DIR = Path(
    "models"
)


FEATURES = [

    # Identity
    "store_nbr",
    "family",

    # Known before/during planning
    "onpromotion",

    # External
    "oil_price",

    # Store metadata
    "city",
    "state",
    "type",
    "cluster",

    # Calendar
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "quarter",
    "is_weekend",

    # Historical demand
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",

    # Rolling demand
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


def main():

    print(
        "Loading training data..."
    )

    train = pd.read_parquet(
        TRAIN_PATH
    )

    validation = pd.read_parquet(
        VALIDATION_PATH
    )

    train = prepare_data(train)
    validation = prepare_data(validation)

    X_train = train[FEATURES]
    y_train = train["sales"]

    X_valid = validation[FEATURES]
    y_valid = validation["sales"]

    print()
    print(
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_valid):,}"
    )

    print()
    print("Training LightGBM...")

    model = lgb.LGBMRegressor(

        objective="regression",

        n_estimators=1000,

        learning_rate=0.05,

        num_leaves=31,

        max_depth=-1,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        n_jobs=-1,
    )

    model.fit(

        X_train,
        y_train,

        categorical_feature=
        CATEGORICAL_FEATURES,

        eval_set=[
            (X_valid, y_valid)
        ],

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50
            ),

            lgb.log_evaluation(
                period=50
            )
        ]
    )

    print()
    print(
        "Generating predictions..."
    )

    predictions = model.predict(
        X_valid
    )

    # Sales cannot be negative
    predictions = np.clip(
        predictions,
        0,
        None
    )

    metrics = evaluate_forecast(
        y_valid,
        predictions
    )

    print()
    print(
        "LIGHTGBM RESULTS"
    )

    print(
        "----------------"
    )

    for name, value in metrics.items():

        print(
            f"{name}: "
            f"{value:.4f}"
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        MODEL_DIR /
        "lightgbm_forecaster.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print()
    print(
        f"Model saved → "
        f"{model_path}"
    )

    results = validation[
        [
            "date",
            "store_nbr",
            "family",
            "sales",
        ]
    ].copy()

    results[
        "prediction"
    ] = predictions

    results.to_parquet(

        MODEL_DIR /
        "validation_predictions.parquet",

        index=False
    )


if __name__ == "__main__":
    main()