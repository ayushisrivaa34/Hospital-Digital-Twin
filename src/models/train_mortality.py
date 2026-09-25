"""
train_mortality.py — Task B: Mortality / Readmission Risk Classification
=========================================================================
Trains a PyCaret classification model to predict in_hospital_death.
Includes calibration (critical for the simulation layer).

Usage:
    python -m src.models.train_mortality
"""

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def train_mortality_model():
    """Train, tune, calibrate, and save the mortality risk model."""
    from pycaret.classification import (
        setup, compare_models, tune_model, calibrate_model,
        finalize_model, save_model
    )

    # ── Load features ──────────────────────────────────────────────────────
    features_path = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} records for mortality training")
    print(f"Mortality rate: {df['in_hospital_death'].mean():.1%}")

    # ── Remove target leakage ──────────────────────────────────────────────
    # Do NOT include length_of_stay_days as a feature for mortality prediction
    ignore_features = ["length_of_stay_days"]
    ignore_cols = [c for c in ignore_features if c in df.columns]

    # ── PyCaret Setup ──────────────────────────────────────────────────────
    clf_setup = setup(
        data=df,
        target="in_hospital_death",
        session_id=42,
        normalize=True,
        fix_imbalance=True,   # critical — death events are rare
        fold=5,
        ignore_features=ignore_cols,
        verbose=False,
        html=False
    )

    # ── Compare models & select best ───────────────────────────────────────
    print("\nComparing classification models...")
    best = compare_models(n_select=1)
    print(f"\nBest model: {type(best).__name__}")

    # ── Tune for AUC ───────────────────────────────────────────────────────
    print("Tuning for AUC...")
    tuned = tune_model(best, optimize="AUC", n_iter=25)

    # ── Calibrate (DO NOT SKIP — see implementation doc) ───────────────────
    # An uncalibrated model produces unreliable probability estimates.
    # The simulation layer treats probabilities as literal event odds,
    # so uncalibrated probabilities = wrong simulation mortality numbers.
    print("Calibrating model...")
    calibrated = calibrate_model(tuned)

    # ── Finalize & save ────────────────────────────────────────────────────
    final_model = finalize_model(calibrated)
    model_path = os.path.join(PROJECT_ROOT, "models", "mortality_risk_model")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    save_model(final_model, model_path)
    print(f"\n[OK] Mortality model saved to {model_path}.pkl")

    return final_model


if __name__ == "__main__":
    print("=" * 60)
    print("Task B: Mortality / Readmission Risk Classification")
    print("=" * 60)
    train_mortality_model()
