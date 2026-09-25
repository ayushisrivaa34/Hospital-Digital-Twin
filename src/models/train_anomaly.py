"""
train_anomaly.py — Task D: Anomaly Detection (Deterioration)
=============================================================
Trains a PyCaret anomaly detection model on ICU vitals time-series
to identify deteriorating patients.

This is the most "fiddly" task (per the implementation doc) because
it requires vitals time-series rather than tabular admission data.

Usage:
    python -m src.models.train_anomaly
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def train_anomaly_model():
    """Train anomaly detection model on ICU vitals."""
    from pycaret.anomaly import setup, create_model, assign_model, save_model

    # ── Load vitals data ───────────────────────────────────────────────────
    vitals_path = os.path.join(PROJECT_ROOT, "data", "processed", "vitals_features.csv")
    df_vitals = pd.read_csv(vitals_path)
    print(f"Loaded {len(df_vitals)} vitals records")

    # ── Prepare features (numeric vitals only) ─────────────────────────────
    vitals_features = df_vitals[[
        "heart_rate", "systolic_bp", "spo2", "temperature", "respiratory_rate"
    ]].copy()

    # Drop rows with any NaN
    vitals_features = vitals_features.dropna()
    print(f"Using {len(vitals_features)} complete vitals records")

    # ── PyCaret Setup ──────────────────────────────────────────────────────
    ano_setup = setup(
        data=vitals_features,
        normalize=True,
        session_id=42,
        verbose=False,
        html=False
    )

    # ── Create Isolation Forest model ──────────────────────────────────────
    print("\nTraining Isolation Forest anomaly detector...")
    iforest = create_model("iforest", fraction=0.05)

    # ── Assign anomaly labels ──────────────────────────────────────────────
    results = assign_model(iforest)

    # ── Extract alerts ─────────────────────────────────────────────────────
    anomaly_col = [c for c in results.columns if "Anomaly" in c][0]
    alerts = results[results[anomaly_col] == 1]

    alerts_path = os.path.join(PROJECT_ROOT, "outputs", "deterioration_alerts.csv")
    os.makedirs(os.path.dirname(alerts_path), exist_ok=True)
    alerts.to_csv(alerts_path, index=False)

    # ── Save model ─────────────────────────────────────────────────────────
    model_path = os.path.join(PROJECT_ROOT, "models", "anomaly_detector")
    save_model(iforest, model_path)

    print(f"\n[OK] Anomaly detector saved to {model_path}")
    print(f"   Alerts exported to {alerts_path}")
    print(f"   Total anomalies detected: {len(alerts)} / {len(results)} "
          f"({len(alerts)/len(results):.1%})")

    # ── Compare with ground truth if available ─────────────────────────────
    if "is_anomalous_ground_truth" in df_vitals.columns:
        gt = df_vitals.loc[vitals_features.index, "is_anomalous_ground_truth"]
        detected = results[anomaly_col].values
        from sklearn.metrics import classification_report
        print("\nComparison with synthetic ground truth:")
        print(classification_report(gt.values, detected, target_names=["Normal", "Anomalous"]))

    return iforest


if __name__ == "__main__":
    print("=" * 60)
    print("Task D: Anomaly Detection (Deterioration)")
    print("=" * 60)
    train_anomaly_model()
