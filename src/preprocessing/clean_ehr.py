"""
clean_ehr.py — EHR Data Cleaning
=================================
Loads raw admissions data, performs deduplication, handles missing values,
and outputs a clean table ready for feature engineering.

Usage:
    python -m src.preprocessing.clean_ehr
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_raw_data(raw_dir: str = None) -> pd.DataFrame:
    """
    Load raw admissions data. Tries raw/ first, falls back to synthetic/.
    """
    if raw_dir is None:
        raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    synthetic_dir = os.path.join(PROJECT_ROOT, "data", "synthetic")

    # Try real data first
    raw_file = os.path.join(raw_dir, "admissions.csv")
    if os.path.exists(raw_file):
        print(f"Loading real data from {raw_file}")
        return pd.read_csv(raw_file)

    # Fall back to synthetic
    synth_file = os.path.join(synthetic_dir, "synthetic_admissions.csv")
    if os.path.exists(synth_file):
        print(f"Loading synthetic data from {synth_file}")
        return pd.read_csv(synth_file, parse_dates=["admit_time", "discharge_time"])

    raise FileNotFoundError(
        f"No data found. Run generate_synthetic.py first, or place admissions.csv in {raw_dir}"
    )


def clean_admissions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the admissions table:
    1. Drop exact duplicates
    2. Handle missing values (median for numeric, mode for categorical)
    3. Validate ranges (age, LOS, severity)
    4. Remove impossible records
    """
    print(f"  Raw records: {len(df)}")

    # ── 1. Drop duplicates ─────────────────────────────────────────────────
    if "patient_id" in df.columns:
        # Keep last admission per patient (most recent)
        df = df.drop_duplicates(subset=["patient_id"], keep="last")
    else:
        df = df.drop_duplicates()
    print(f"  After dedup: {len(df)}")

    # ── 2. Handle missing values ───────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Imputed {n_missing} missing in '{col}' with median={median_val:.2f}")

    for col in categorical_cols:
        if col in ["patient_id", "admit_time", "discharge_time"]:
            continue
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"  Imputed {n_missing} missing in '{col}' with mode='{mode_val}'")

    # ── 3. Validate ranges ─────────────────────────────────────────────────
    if "age" in df.columns:
        df["age"] = df["age"].clip(0, 120)

    if "length_of_stay_days" in df.columns:
        df = df[df["length_of_stay_days"] > 0]
        df["length_of_stay_days"] = df["length_of_stay_days"].clip(upper=365)

    if "severity_score" in df.columns:
        df["severity_score"] = df["severity_score"].clip(0, 10)

    print(f"  After validation: {len(df)}")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    print("=== EHR Data Cleaning ===")
    raw_df = load_raw_data()
    clean_df = clean_admissions(raw_df)

    out_path = os.path.join(PROJECT_ROOT, "data", "processed", "clean_admissions.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    clean_df.to_csv(out_path, index=False)
    print(f"\n[OK] Clean data saved to {out_path}")
    print(f"   Shape: {clean_df.shape}")
