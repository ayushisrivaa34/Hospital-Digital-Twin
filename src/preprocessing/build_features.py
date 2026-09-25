"""
build_features.py — Feature Engineering
========================================
Takes cleaned admissions data and produces a PyCaret-ready features table.

Features computed:
    - LOS (discharge_time − admit_time)  [already present in synthetic data]
    - Simplified comorbidity count
    - One-hot encoding for categorical fields
    - Binary encoding for sex
    - Ordinal encoding for admission type

Usage:
    python -m src.preprocessing.build_features
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from the cleaned admissions table.
    Returns a DataFrame ready for PyCaret setup().
    """
    features = df.copy()

    # ── Compute LOS if not already present ─────────────────────────────────
    if "length_of_stay_days" not in features.columns:
        if "admit_time" in features.columns and "discharge_time" in features.columns:
            features["admit_time"] = pd.to_datetime(features["admit_time"])
            features["discharge_time"] = pd.to_datetime(features["discharge_time"])
            features["length_of_stay_days"] = (
                (features["discharge_time"] - features["admit_time"]).dt.total_seconds()
                / 86400
            ).round(2)

    # ── Binary sex encoding ────────────────────────────────────────────────
    if "sex" in features.columns:
        features["sex_numeric"] = (features["sex"] == "M").astype(int)

    # ── Admission type ordinal ─────────────────────────────────────────────
    if "admission_type" in features.columns:
        admission_map = {"elective": 0, "urgent": 1, "emergency": 2}
        features["admission_type_ordinal"] = features["admission_type"].map(admission_map)

    # ── Diagnosis one-hot encoding ─────────────────────────────────────────
    if "primary_diagnosis" in features.columns:
        diag_dummies = pd.get_dummies(
            features["primary_diagnosis"], prefix="diag", drop_first=False
        )
        features = pd.concat([features, diag_dummies], axis=1)

    # ── Age bins (clinically meaningful groups) ────────────────────────────
    if "age" in features.columns:
        features["age_group"] = pd.cut(
            features["age"],
            bins=[0, 30, 45, 60, 75, 120],
            labels=["18-30", "31-45", "46-60", "61-75", "76+"]
        )
        age_dummies = pd.get_dummies(features["age_group"], prefix="age_grp", drop_first=False)
        features = pd.concat([features, age_dummies], axis=1)

    # ── Interaction features ───────────────────────────────────────────────
    if "severity_score" in features.columns and "age" in features.columns:
        features["severity_x_age"] = features["severity_score"] * features["age"]

    if "severity_score" in features.columns and "num_comorbidities" in features.columns:
        features["severity_x_comorbidities"] = (
            features["severity_score"] * features["num_comorbidities"]
        )

    # ── Drop raw categorical / temporal columns ────────────────────────────
    # PyCaret will handle these, but we provide clean numeric features
    drop_cols = [
        "patient_id", "admit_time", "discharge_time",
        "sex", "admission_type", "primary_diagnosis", "age_group"
    ]
    features = features.drop(columns=[c for c in drop_cols if c in features.columns])

    return features


if __name__ == "__main__":
    print("=== Feature Engineering ===")

    # Load cleaned data
    clean_path = os.path.join(PROJECT_ROOT, "data", "processed", "clean_admissions.csv")
    if not os.path.exists(clean_path):
        print("Clean admissions not found — trying synthetic features directly...")
        clean_path = os.path.join(PROJECT_ROOT, "data", "synthetic", "synthetic_admissions.csv")

    df = pd.read_csv(clean_path)
    print(f"  Loaded {len(df)} records from {os.path.basename(clean_path)}")

    features = build_features(df)

    out_path = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
    features.to_csv(out_path, index=False)
    print(f"\n[OK] Features saved to {out_path}")
    print(f"   Shape: {features.shape}")
    print(f"   Columns: {list(features.columns)}")
    print(f"\n   Target distributions:")
    print(f"     LOS mean:      {features['length_of_stay_days'].mean():.2f} days")
    print(f"     Mortality rate: {features['in_hospital_death'].mean():.1%}")
