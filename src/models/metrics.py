import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def rmsle(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Sales predictions cannot be negative
    y_pred = np.clip(
        y_pred,
        0,
        None
    )

    return np.sqrt(
        np.mean(
            (
                np.log1p(y_pred)
                - np.log1p(y_true)
            ) ** 2
        )
    )


def rmse(y_true, y_pred):

    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


def mae(y_true, y_pred):

    return mean_absolute_error(
        y_true,
        y_pred
    )


def wmape(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.sum(
        np.abs(y_true)
    )

    if denominator == 0:
        return np.nan

    return (
        np.sum(
            np.abs(
                y_true - y_pred
            )
        )
        / denominator
    )


def evaluate_forecast(
    y_true,
    y_pred
):

    return {
        "RMSLE": rmsle(
            y_true,
            y_pred
        ),

        "RMSE": rmse(
            y_true,
            y_pred
        ),

        "MAE": mae(
            y_true,
            y_pred
        ),

        "WMAPE": wmape(
            y_true,
            y_pred
        ),
    }