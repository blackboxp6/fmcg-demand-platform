from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(
    "models/lightgbm_forecaster.joblib"
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


model = joblib.load(
    MODEL_PATH
)


importance = pd.DataFrame({

    "feature": FEATURES,

    "importance":
        model.feature_importances_
})


importance = importance.sort_values(

    "importance",

    ascending=False
)


print(
    importance.to_string(
        index=False
    )
)