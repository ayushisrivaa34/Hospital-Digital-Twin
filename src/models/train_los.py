"""
train_los.py — Task A: Length-of-Stay Regression
=================================================
Trains a PyCaret regression model to predict length_of_stay_days.
This model's output feeds directly into the SimPy simulation.

Usage:
    python -m src.models.train_los
"""

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def train_los_model():
    """Train, tune, and save the LOS regression model."""
    from pycaret.regression import setup, compare_models, tune_model, finalize_model, save_model

    # ── Load features ──────────────────────────────────────────────────────
    features_path = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} records for LOS training")

    # ── Remove target leakage ──────────────────────────────────────────────
    # Do NOT include in_hospital_death as a feature for LOS prediction
    ignore_features = ["in_hospital_death"]
    ignore_cols = [c for c in ignore_features if c in df.columns]

    # ── PyCaret Setup ──────────────────────────────────────────────────────
    reg_setup = setup(
        data=df,
        target="length_of_stay_days",
        session_id=42,
        normalize=True,
        fold=5,
        ignore_features=ignore_cols,
        verbose=False,
        html=False
    )

    # ── Compare models & select best ───────────────────────────────────────
    print("\nComparing regression models...")
    best_model = compare_models(n_select=1)
    print(f"\nBest model: {type(best_model).__name__}")

    # ── Tune the best model ────────────────────────────────────────────────
    print("Tuning best model...")
    tuned_model = tune_model(best_model, n_iter=25, optimize="MAE")

    # ── Finalize (retrain on full dataset) ─────────────────────────────────
    final_model = finalize_model(tuned_model)

    # ── Save ───────────────────────────────────────────────────────────────
    model_path = os.path.join(PROJECT_ROOT, "models", "los_predictor")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    save_model(final_model, model_path)
    print(f"\n[OK] LOS model saved to {model_path}.pkl")

    return final_model


if __name__ == "__main__":
    print("=" * 60)
    print("Task A: Length-of-Stay Regression")
    print("=" * 60)
    train_los_model()
