from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path(
    "models/lightgbm_forecaster.joblib"
)

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


class ForecastService:

    def __init__(self):

        print("Loading forecasting model...")

        self.model = joblib.load(
            MODEL_PATH
        )

        print("Loading historical data...")

        self.history = pd.read_parquet(
            GOLD_PATH
        )

        self.history["date"] = (
            pd.to_datetime(
                self.history["date"]
            )
        )

        print("Forecast service ready.")

    def get_available_families(self):

        return sorted(
            self.history["family"]
            .dropna()
            .unique()
            .tolist()
        )

    def get_available_stores(self):

        return sorted(
            self.history["store_nbr"]
            .dropna()
            .unique()
            .tolist()
        )

    def _get_series(
        self,
        store_nbr,
        family,
    ):

        df = self.history[
            (
                self.history["store_nbr"]
                == store_nbr
            )
            &
            (
                self.history["family"]
                == family
            )
        ].copy()

        df = df.sort_values("date")

        if df.empty:

            raise ValueError(
                "No historical data found for "
                "this store/family combination."
            )

        return df

    def forecast_next_day(
        self,
        store_nbr,
        family,
        onpromotion=0,
    ):

        result = self.forecast_multiple_days(
            store_nbr=store_nbr,
            family=family,
            forecast_days=1,
            promotions=[onpromotion],
        )

        first_forecast = result["forecast"][0]

        return {
            "store_nbr": int(store_nbr),
            "family": family,
            "forecast_date":
                first_forecast["date"],
            "predicted_sales":
                first_forecast[
                    "predicted_sales"
                ],
            "onpromotion":
                first_forecast[
                    "onpromotion"
                ],
        }

    def forecast_multiple_days(
        self,
        store_nbr,
        family,
        forecast_days=7,
        promotions=None,
    ):

        if forecast_days < 1:

            raise ValueError(
                "forecast_days must be at least 1."
            )

        if forecast_days > 30:

            raise ValueError(
                "Maximum forecast horizon is 30 days."
            )

        history = self._get_series(
            store_nbr,
            family,
        )

        if len(history) < 28:

            raise ValueError(
                "At least 28 historical "
                "observations are required."
            )

        # --------------------------------
        # Promotion schedule
        # --------------------------------

        if promotions is None:

            promotions = (
                [0] * forecast_days
            )

        if len(promotions) != forecast_days:

            raise ValueError(
                "Number of promotion values "
                "must equal forecast_days."
            )

        # --------------------------------
        # Historical sales
        # --------------------------------

        sales_history = (
            history["sales"]
            .astype(float)
            .tolist()
        )

        last_row = history.iloc[-1]

        last_date = last_row["date"]

        forecasts = []

        # --------------------------------
        # Recursive forecasting
        # --------------------------------

        for step in range(
            1,
            forecast_days + 1,
        ):

            forecast_date = (
                last_date
                + pd.Timedelta(
                    days=step
                )
            )

            # Lag features

            lag_1 = sales_history[-1]
            lag_7 = sales_history[-7]
            lag_14 = sales_history[-14]
            lag_28 = sales_history[-28]

            # Rolling features

            rolling_mean_7 = np.mean(
                sales_history[-7:]
            )

            rolling_mean_14 = np.mean(
                sales_history[-14:]
            )

            rolling_mean_28 = np.mean(
                sales_history[-28:]
            )

            rolling_std_7 = np.std(
                sales_history[-7:],
                ddof=1,
            )

            # Build model input

            feature_row = {

                "store_nbr":
                    store_nbr,

                "family":
                    family,

                "onpromotion":
                    promotions[
                        step - 1
                    ],

                "oil_price":
                    float(
                        last_row[
                            "oil_price"
                        ]
                    ),

                "city":
                    last_row["city"],

                "state":
                    last_row["state"],

                "type":
                    last_row["type"],

                "cluster":
                    last_row["cluster"],

                "year":
                    forecast_date.year,

                "month":
                    forecast_date.month,

                "day":
                    forecast_date.day,

                "day_of_week":
                    forecast_date.dayofweek,

                "week_of_year":
                    int(
                        forecast_date
                        .isocalendar()
                        .week
                    ),

                "quarter":
                    forecast_date.quarter,

                "is_weekend":
                    int(
                        forecast_date
                        .dayofweek >= 5
                    ),

                "sales_lag_1":
                    lag_1,

                "sales_lag_7":
                    lag_7,

                "sales_lag_14":
                    lag_14,

                "sales_lag_28":
                    lag_28,

                "rolling_mean_7":
                    rolling_mean_7,

                "rolling_mean_14":
                    rolling_mean_14,

                "rolling_mean_28":
                    rolling_mean_28,

                "rolling_std_7":
                    rolling_std_7,
            }

            X = pd.DataFrame(
                [feature_row]
            )

            for column in (
                CATEGORICAL_FEATURES
            ):

                X[column] = (
                    X[column]
                    .astype("category")
                )

            X = X[FEATURES]

            # Predict

            prediction = (
                self.model.predict(
                    X
                )[0]
            )

            prediction = float(
                np.clip(
                    prediction,
                    0,
                    None,
                )
            )

            forecasts.append(
                {
                    "date":
                        forecast_date
                        .strftime(
                            "%Y-%m-%d"
                        ),

                    "predicted_sales":
                        round(
                            prediction,
                            2,
                        ),

                    "onpromotion":
                        int(
                            promotions[
                                step - 1
                            ]
                        ),
                }
            )

            # Critical recursive step:
            # prediction becomes history
            # for the next day.

            sales_history.append(
                prediction
            )

        return {
            "store_nbr":
                int(store_nbr),

            "family":
                family,

            "forecast_days":
                forecast_days,

            "forecast":
                forecasts,
        }