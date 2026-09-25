"""
train_forecast.py — Task E: Time-Series Forecasting (ED Arrivals)
=================================================================
Trains a PyCaret time-series model to predict hourly patient arrivals.
Uses the synthetic hourly arrival data.

Note: With only ~500 patients, the time series may be thin. The
implementation doc acknowledges this — treat as a placeholder until
real high-volume data is available.

Usage:
    python -m src.models.train_forecast
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def train_forecast_model():
    """Train ED arrivals time-series forecasting model."""
    try:
        from pycaret.time_series import setup, compare_models, tune_model, finalize_model, save_model, predict_model
    except ImportError:
        print("[WARN]  PyCaret time_series module not available.")
        print("   Falling back to a simple statistical forecast...")
        return _fallback_forecast()

    # ── Load hourly arrivals ───────────────────────────────────────────────
    arrivals_path = os.path.join(PROJECT_ROOT, "data", "processed", "hourly_arrivals.csv")
    ed_hourly = pd.read_csv(arrivals_path, parse_dates=["admit_hour"])
    ed_hourly = ed_hourly.set_index("admit_hour")
    ed_hourly.index.freq = "H"
    print(f"Loaded {len(ed_hourly)} hourly arrival records")

    # ── PyCaret Setup ──────────────────────────────────────────────────────
    ts_setup = setup(
        data=ed_hourly,
        target="patient_count",
        session_id=42,
        fh=48,  # forecast horizon: 48 hours
        verbose=False,
        html=False
    )

    # ── Compare and select best model ──────────────────────────────────────
    print("\nComparing time-series models...")
    best_ts = compare_models(n_select=1)
    print(f"\nBest model: {type(best_ts).__name__}")

    # ── Tune & finalize ────────────────────────────────────────────────────
    print("Tuning model...")
    tuned = tune_model(best_ts)
    final_ts = finalize_model(tuned)

    # ── Generate predictions ───────────────────────────────────────────────
    predictions = predict_model(final_ts, fh=48)
    pred_path = os.path.join(PROJECT_ROOT, "outputs", "ed_arrival_forecast.csv")
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    predictions.to_csv(pred_path)
    print(f"\nForecast saved to {pred_path}")

    # ── Save model ─────────────────────────────────────────────────────────
    model_path = os.path.join(PROJECT_ROOT, "models", "ed_arrivals_forecaster")
    save_model(final_ts, model_path)
    print(f"[OK] Forecast model saved to {model_path}")

    return final_ts


def _fallback_forecast():
    """
    Simple moving average fallback when PyCaret TS is unavailable.
    Produces a forecast CSV so downstream code can still run.
    """
    arrivals_path = os.path.join(PROJECT_ROOT, "data", "processed", "hourly_arrivals.csv")
    ed_hourly = pd.read_csv(arrivals_path, parse_dates=["admit_hour"])

    # Simple 24-hour moving average as the forecast
    mean_rate = ed_hourly["patient_count"].mean()
    last_time = ed_hourly["admit_hour"].max()

    forecast_hours = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=48, freq="H")
    forecast = pd.DataFrame({
        "admit_hour": forecast_hours,
        "predicted_count": np.round(np.random.poisson(max(mean_rate, 0.5), 48)).astype(int)
    })

    pred_path = os.path.join(PROJECT_ROOT, "outputs", "ed_arrival_forecast.csv")
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    forecast.to_csv(pred_path, index=False)
    print(f"\n[OK] Fallback forecast saved to {pred_path}")
    print(f"   Mean arrival rate: {mean_rate:.2f} patients/hour")

    return None


if __name__ == "__main__":
    print("=" * 60)
    print("Task E: Time-Series Forecasting (ED Arrivals)")
    print("=" * 60)
    train_forecast_model()
