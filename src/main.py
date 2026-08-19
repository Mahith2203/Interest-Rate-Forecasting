"""
End-to-end pipeline:
  1. Simulate short-rate paths with Vasicek and CIR models.
  2. Train an LSTM forecaster on each simulated series.
  3. Evaluate forecasts (MAE, RMSE, MAPE) and plot true vs predicted rates.

Run from the repo root:
    python src/main.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from rate_models import vasicek_sim, cir_sim
from lstm_forecast import train_forecast_lstm
from evaluate import evaluate_forecast

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Model / simulation parameters -----------------------------------
R0, A, B, SIGMA = 0.03, 0.10, 0.05, 0.02
T, DT = 5.0, 0.01
N_LAGS, EPOCHS, BATCH_SIZE = 10, 20, 16
SEED = 42


def run_pipeline(name: str, series: np.ndarray):
    split_idx = int(len(series) * 0.8)
    train_series = series[:split_idx]
    test_series = series[split_idx - N_LAGS:]  # include lag window for test prep

    y_true, y_pred, _ = train_forecast_lstm(
        train_series, test_series, n_lags=N_LAGS, epochs=EPOCHS, batch_size=BATCH_SIZE
    )
    metrics = evaluate_forecast(y_true, y_pred)

    print(f"\n[{name}] MAE={metrics['MAE']:.5f}  RMSE={metrics['RMSE']:.5f}  MAPE={metrics['MAPE']:.2f}%")

    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="True rate")
    plt.plot(y_pred, label="LSTM forecast")
    plt.title(f"{name}: True vs Forecasted Short Rate")
    plt.xlabel("Time step")
    plt.ylabel("Short rate")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{name.lower()}_forecast.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    return metrics


def main():
    np.random.seed(SEED)

    rates_vasicek = vasicek_sim(R0, A, B, SIGMA, T, DT, seed=SEED)
    rates_cir = cir_sim(R0, A, B, SIGMA, T, DT, seed=SEED)

    all_metrics = {
        "Vasicek": run_pipeline("Vasicek", rates_vasicek),
        "CIR": run_pipeline("CIR", rates_cir),
    }

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved plots and metrics.json to {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()
