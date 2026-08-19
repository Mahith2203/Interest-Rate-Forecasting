"""
LSTM-based one-step-ahead forecasting for simulated short-rate series.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input


def prepare_lstm_data(series: np.ndarray, n_lags: int = 10):
    """
    Convert a 1-D (scaled) series into supervised-learning (X, y) pairs
    using a sliding lag window, reshaped for an LSTM: [samples, timesteps, features].
    """
    X, y = [], []
    for i in range(len(series) - n_lags):
        X.append(series[i:i + n_lags])
        y.append(series[i + n_lags])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def build_lstm_model(n_lags: int, units: int = 50) -> Sequential:
    model = Sequential([
        Input(shape=(n_lags, 1)),
        LSTM(units, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_forecast_lstm(train_series: np.ndarray, test_series: np.ndarray,
                         n_lags: int = 10, epochs: int = 20, batch_size: int = 16,
                         verbose: int = 0):
    """
    Scale, train an LSTM on `train_series`, and forecast one-step-ahead
    values over `test_series`. Returns (y_true, y_pred) in the original
    (unscaled) rate units.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
    test_scaled = scaler.transform(test_series.reshape(-1, 1))

    X_train, y_train = prepare_lstm_data(train_scaled, n_lags)
    X_test, y_test = prepare_lstm_data(test_scaled, n_lags)

    model = build_lstm_model(n_lags)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    y_pred_scaled = model.predict(X_test, verbose=verbose)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    return y_true, y_pred, model
