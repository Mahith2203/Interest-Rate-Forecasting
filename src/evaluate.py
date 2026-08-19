"""Forecast accuracy metrics: MAE, RMSE, MAPE."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute MAE, RMSE and MAPE (%) between true and predicted rate series.
    Zero-valued true rates are excluded from the MAPE calculation to avoid
    division by zero.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    non_zero = y_true != 0
    mape = np.mean(
        np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])
    ) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
