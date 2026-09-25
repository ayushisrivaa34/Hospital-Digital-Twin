"""
Synthetic Hospital Data Generator
=================================
Generates a realistic synthetic dataset matching the MIMIC-IV Demo schema
so all downstream pipeline code can be built and tested before real data
access is available.

Output:
    data/synthetic/synthetic_admissions.csv
    data/synthetic/synthetic_vitals.csv
    data/processed/features.csv  (ready for PyCaret)
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Configuration ──────────────────────────────────────────────────────────
NUM_PATIENTS = 500
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Resolve project root (this script can be run from anywhere)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(SYNTHETIC_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── ICD-10 simplified diagnosis categories ─────────────────────────────────
DIAGNOSIS_CATEGORIES = [
    "Sepsis", "Pneumonia", "Heart Failure", "COPD Exacerbation",
    "Acute MI", "Stroke", "Renal Failure", "GI Bleeding",
    "Diabetic Emergency", "Trauma", "Surgical (Elective)",
    "Surgical (Emergency)", "Other Medical"
]

ADMISSION_TYPES = ["emergency", "urgent", "elective"]
ADMISSION_TYPE_WEIGHTS = [0.55, 0.25, 0.20]  # emergency-heavy, realistic


def _age_from_admission_type(admission_type: str) -> int:
    """Generate an age that's correlated with admission type."""
    if admission_type == "elective":
        return int(np.clip(np.random.normal(55, 15), 18, 95))
    elif admission_type == "emergency":
        return int(np.clip(np.random.normal(62, 18), 18, 99))
    else:  # urgent
        return int(np.clip(np.random.normal(58, 16), 18, 97))


def _severity_score(age: int, diagnosis: str, admission_type: str) -> float:
    """
    Simplified Charlson-like comorbidity/severity score (0-10 scale).
    Higher = sicker patient. Correlated with age and diagnosis acuity.
    """
    base = np.random.exponential(1.5)
    age_component = (age - 40) * 0.04 if age > 40 else 0
    diag_component = {
        "Sepsis": 3.0, "Acute MI": 2.5, "Stroke": 2.5,
        "Heart Failure": 2.0, "Renal Failure": 2.0,
        "GI Bleeding": 1.5, "COPD Exacerbation": 1.5,
        "Diabetic Emergency": 1.8, "Pneumonia": 1.2,
        "Trauma": 2.0, "Surgical (Emergency)": 1.5,
        "Surgical (Elective)": 0.5, "Other Medical": 0.8
    }.get(diagnosis, 1.0)
    admission_component = {"emergency": 0.8, "urgent": 0.4, "elective": 0.0}[admission_type]

    return round(np.clip(base + age_component + diag_component + admission_component, 0, 10), 2)


def _length_of_stay(severity: float, diagnosis: str, admission_type: str) -> float:
    """Generate realistic LOS (in days) conditioned on severity."""
    base_los = {
        "Sepsis": 8, "Pneumonia": 5, "Heart Failure": 6,
        "COPD Exacerbation": 4, "Acute MI": 5, "Stroke": 7,
        "Renal Failure": 6, "GI Bleeding": 4, "Diabetic Emergency": 3,
        "Trauma": 6, "Surgical (Elective)": 3, "Surgical (Emergency)": 5,
        "Other Medical": 3
    }.get(diagnosis, 4)

    los = base_los + severity * 0.8 + np.random.exponential(2)
    if admission_type == "elective":
        los *= 0.7  # elective patients tend to have shorter stays
    return round(max(los, 0.5), 2)


def _mortality_flag(severity: float, age: int, los: float) -> int:
    """
    In-hospital death probability correlated with severity and age.
    Overall target rate: ~5-8% to be realistic.
    """
    logit = -4.5 + 0.35 * severity + 0.02 * (age - 50) + 0.01 * los
    prob = 1 / (1 + np.exp(-logit))
    return int(np.random.random() < prob)


def generate_admissions() -> pd.DataFrame:
    """Generate the main admissions/features table."""
    records = []
    base_date = datetime(2023, 1, 1)

    for i in range(NUM_PATIENTS):
        admit_time = base_date + timedelta(
            hours=np.random.exponential(2.0) + i * 1.2
        )
        admission_type = np.random.choice(ADMISSION_TYPES, p=ADMISSION_TYPE_WEIGHTS)
        sex = np.random.choice(["M", "F"])
        age = _age_from_admission_type(admission_type)
        diagnosis = np.random.choice(DIAGNOSIS_CATEGORIES)
        severity = _severity_score(age, diagnosis, admission_type)
        los = _length_of_stay(severity, diagnosis, admission_type)
        death = _mortality_flag(severity, age, los)

        # If died, LOS might be shorter (some patients die quickly)
        if death and np.random.random() < 0.4:
            los = round(max(los * 0.5, 0.5), 2)

        discharge_time = admit_time + timedelta(days=los)

        records.append({
            "patient_id": f"P{i:04d}",
            "admit_time": admit_time,
            "discharge_time": discharge_time,
            "age": age,
            "sex": sex,
            "admission_type": admission_type,
            "primary_diagnosis": diagnosis,
            "severity_score": severity,
            "length_of_stay_days": los,
            "in_hospital_death": death,
            "num_comorbidities": int(np.clip(np.random.poisson(severity / 2), 0, 8)),
        })

    df = pd.DataFrame(records)
    return df


def generate_vitals(admissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic ICU vitals time-series for anomaly detection (Task D).
    Produces hourly readings for a subset of patients (ICU admissions).
    """
    vitals_records = []
    # ~30% of patients are ICU patients
    icu_patients = admissions_df.sample(frac=0.30, random_state=RANDOM_SEED)

    for _, row in icu_patients.iterrows():
        hours_in_icu = int(row["length_of_stay_days"] * 24)
        hours_in_icu = min(hours_in_icu, 168)  # cap at 7 days of hourly data

        # Base vitals conditioned on severity
        base_hr = 80 + row["severity_score"] * 3
        base_sbp = 120 - row["severity_score"] * 2
        base_spo2 = 97 - row["severity_score"] * 0.5
        base_temp = 37.0 + row["severity_score"] * 0.1
        base_rr = 16 + row["severity_score"] * 0.8

        for h in range(max(hours_in_icu, 1)):
            # Introduce occasional anomalous readings (~5% of timestamps)
            is_anomalous = np.random.random() < 0.05
            anomaly_factor = 1.0 + (np.random.uniform(0.2, 0.5) if is_anomalous else 0)

            vitals_records.append({
                "patient_id": row["patient_id"],
                "timestamp_hour": h,
                "heart_rate": round(np.random.normal(base_hr * anomaly_factor, 8), 1),
                "systolic_bp": round(np.random.normal(base_sbp / anomaly_factor, 12), 1),
                "spo2": round(np.clip(np.random.normal(base_spo2 / anomaly_factor, 2), 70, 100), 1),
                "temperature": round(np.random.normal(base_temp * anomaly_factor, 0.4), 1),
                "respiratory_rate": round(np.random.normal(base_rr * anomaly_factor, 3), 1),
                "is_anomalous_ground_truth": int(is_anomalous),
            })

    return pd.DataFrame(vitals_records)


def generate_hourly_arrivals(admissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate admissions into hourly arrival counts for time-series
    forecasting (Task E).
    """
    df = admissions_df.copy()
    df["admit_hour"] = df["admit_time"].dt.floor("h")
    hourly = df.groupby("admit_hour").size().reset_index(name="patient_count")
    # Fill gaps in the time series
    full_range = pd.date_range(
        start=hourly["admit_hour"].min(),
        end=hourly["admit_hour"].max(),
        freq="h"
    )
    hourly = hourly.set_index("admit_hour").reindex(full_range, fill_value=0)
    hourly.index.name = "admit_hour"
    hourly = hourly.reset_index()
    return hourly


if __name__ == "__main__":
    print("Generating synthetic hospital data...")
    print(f"  Patients: {NUM_PATIENTS}")
    print(f"  Random seed: {RANDOM_SEED}")

    # ── Generate ───────────────────────────────────────────────────────────
    admissions = generate_admissions()
    vitals = generate_vitals(admissions)
    hourly_arrivals = generate_hourly_arrivals(admissions)

    # ── Save synthetic data ────────────────────────────────────────────────
    admissions.to_csv(os.path.join(SYNTHETIC_DIR, "synthetic_admissions.csv"), index=False)
    vitals.to_csv(os.path.join(SYNTHETIC_DIR, "synthetic_vitals.csv"), index=False)
    hourly_arrivals.to_csv(os.path.join(SYNTHETIC_DIR, "synthetic_hourly_arrivals.csv"), index=False)

    # ── Also save processed features.csv (ready for PyCaret) ───────────────
    features = admissions.drop(columns=["admit_time", "discharge_time", "patient_id"])
    features.to_csv(os.path.join(PROCESSED_DIR, "features.csv"), index=False)

    # Save vitals for anomaly detection
    vitals.to_csv(os.path.join(PROCESSED_DIR, "vitals_features.csv"), index=False)

    # Save hourly arrivals for forecasting
    hourly_arrivals.to_csv(os.path.join(PROCESSED_DIR, "hourly_arrivals.csv"), index=False)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n[OK] Data generated successfully!")
    print(f"  Admissions:      {len(admissions)} rows")
    print(f"  Vitals:          {len(vitals)} rows")
    print(f"  Hourly arrivals: {len(hourly_arrivals)} rows")
    print(f"  Mortality rate:  {admissions['in_hospital_death'].mean():.1%}")
    print(f"  Mean LOS:        {admissions['length_of_stay_days'].mean():.1f} days")
    print(f"\nFiles saved to:")
    print(f"  {SYNTHETIC_DIR}")
    print(f"  {PROCESSED_DIR}")
