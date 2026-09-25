"""
hospital_sim.py — SimPy Digital Twin Core (Enhanced)
=====================================================
Implements the SimPy simulation layer with:
  - PriorityResource for triage queues (high-risk patients admitted first)
  - Staff resource pool (nurses/doctors as a separate constraint)
  - Configurable bed capacity, staff capacity, arrival rates
  - Detailed event logging for downstream analysis

This is the enhanced version from §5.7 of the implementation doc, which
adds staff allocation and triage priority beyond the base paper.

Usage:
    python -m src.simulation.hospital_sim
"""

import os
import sys
import simpy
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# ── Try to load PyCaret models, fall back to simple estimators ─────────────
_los_model = None
_mortality_model = None


def _load_models():
    """Load trained PyCaret models. Falls back to statistical estimates."""
    global _los_model, _mortality_model

    los_path = os.path.join(PROJECT_ROOT, "models", "los_predictor.pkl")
    mort_path = os.path.join(PROJECT_ROOT, "models", "mortality_risk_model.pkl")

    if os.path.exists(los_path):
        try:
            from pycaret.regression import load_model as load_reg_model
            _los_model = load_reg_model(os.path.join(PROJECT_ROOT, "models", "los_predictor"))
            print("  Loaded LOS prediction model")
        except Exception as e:
            print(f"  [WARN] Could not load LOS model: {e}")

    if os.path.exists(mort_path):
        try:
            from pycaret.classification import load_model as load_clf_model
            _mortality_model = load_clf_model(os.path.join(PROJECT_ROOT, "models", "mortality_risk_model"))
            print("  Loaded mortality risk model")
        except Exception as e:
            print(f"  [WARN] Could not load mortality model: {e}")


def predict_los(features_row: pd.DataFrame) -> float:
    """Predict length of stay for a patient."""
    if _los_model is not None:
        try:
            pred = _los_model.predict(features_row)
            return float(pred.iloc[0]) if hasattr(pred, 'iloc') else float(pred[0])
        except Exception:
            pass
    # Fallback: use the LOS value from the data if available
    if "length_of_stay_days" in features_row.columns:
        return float(features_row["length_of_stay_days"].values[0])
    return float(np.random.exponential(5))


def predict_mortality_risk(features_row: pd.DataFrame) -> float:
    """Predict mortality probability for a patient."""
    if _mortality_model is not None:
        try:
            prob = _mortality_model.predict_proba(features_row)
            return float(prob[0][1]) if hasattr(prob, '__getitem__') else 0.05
        except Exception:
            pass
    # Fallback: use severity score to estimate mortality risk
    if "severity_score" in features_row.columns:
        severity = float(features_row["severity_score"].values[0])
        return min(0.02 + severity * 0.05, 0.6)
    return 0.05


def patient(env, name, ward, staff_pool, features_row, log, queue_log):
    """
    SimPy process for a single patient journey.

    Enhanced with:
      - Triage priority (high-risk patients get priority=1, others priority=2)
      - Staff resource allocation for initial care
      - Detailed event logging
    """
    arrival_time = env.now
    los_pred = predict_los(features_row)
    mort_risk = predict_mortality_risk(features_row)

    # ── Triage Priority Assignment ─────────────────────────────────────────
    # High-risk patients (mortality > 40%) get higher priority (lower number)
    priority_level = 1 if mort_risk > 0.4 else 2

    # ── Wait for bed ───────────────────────────────────────────────────────
    with ward.request(priority=priority_level) as bed:
        yield bed
        admit_time = env.now
        wait_time = admit_time - arrival_time

        # Log queue event
        queue_log.append({
            "time": arrival_time,
            "event": "queued",
            "patient": name,
            "priority": priority_level,
            "wait_hours": wait_time,
        })

        # ── Staff allocation for initial assessment ────────────────────────
        with staff_pool.request() as staff:
            yield staff
            yield env.timeout(0.5)  # 30 min initial care allocation

        # ── Patient stay ───────────────────────────────────────────────────
        stay_hours = max(los_pred, 0.1) * 24  # convert days to hours
        yield env.timeout(stay_hours)

        # ── Determine outcome ──────────────────────────────────────────────
        outcome = "expired" if np.random.random() < mort_risk else "discharged"

        log.append({
            "patient": name,
            "arrival_time": arrival_time,
            "admit_time": admit_time,
            "discharge_time": env.now,
            "wait_hours": wait_time,
            "los_days": los_pred,
            "mortality_risk": mort_risk,
            "outcome": outcome,
            "priority": priority_level,
        })


def patient_generator(env, ward, staff_pool, df_historical, log, queue_log,
                       mean_interarrival_hours=2.0):
    """
    Generates patients arriving at exponential inter-arrival times.
    Each patient's features are sampled from the historical data.
    """
    i = 0
    while True:
        yield env.timeout(np.random.exponential(mean_interarrival_hours))
        features_row = df_historical.sample(1)
        env.process(patient(env, f"Patient_{i}", ward, staff_pool,
                           features_row, log, queue_log))
        i += 1


def run_simulation(df_historical, bed_capacity=20, staff_capacity=5,
                   sim_days=30, mean_interarrival_hours=2.0, seed=42):
    """
    Run the hospital digital twin simulation.

    Args:
        df_historical: Feature DataFrame to sample patient profiles from
        bed_capacity: Number of hospital beds
        staff_capacity: Number of staff (nurses/doctors) available
        sim_days: Duration of simulation in days
        mean_interarrival_hours: Mean time between patient arrivals
        seed: Random seed for reproducibility

    Returns:
        Tuple of (patient_log_df, queue_log_df)
    """
    np.random.seed(seed)
    env = simpy.Environment()

    # PriorityResource enables triage queues (from docx §5.7)
    ward = simpy.PriorityResource(env, capacity=bed_capacity)
    staff_pool = simpy.Resource(env, capacity=staff_capacity)

    log = []
    queue_log = []

    env.process(patient_generator(
        env, ward, staff_pool, df_historical, log, queue_log,
        mean_interarrival_hours
    ))
    env.run(until=24 * sim_days)

    return pd.DataFrame(log), pd.DataFrame(queue_log)


def compute_kpis(patient_log: pd.DataFrame, bed_capacity: int,
                 sim_days: int) -> dict:
    """Compute key performance indicators from simulation output."""
    if patient_log.empty:
        return {}

    total_patients = len(patient_log)
    avg_los = patient_log["los_days"].mean()
    avg_wait = patient_log["wait_hours"].mean()
    mortality_rate = (patient_log["outcome"] == "expired").mean()
    throughput = total_patients / sim_days

    # Bed occupancy: total bed-hours used / total bed-hours available
    total_bed_hours = patient_log["los_days"].sum() * 24
    available_bed_hours = bed_capacity * sim_days * 24
    occupancy = min(total_bed_hours / available_bed_hours, 1.0)

    return {
        "total_patients": total_patients,
        "avg_los_days": round(avg_los, 2),
        "avg_wait_hours": round(avg_wait, 2),
        "mortality_rate": round(mortality_rate, 4),
        "throughput_per_day": round(throughput, 1),
        "bed_occupancy": round(occupancy, 4),
        "discharged": (patient_log["outcome"] == "discharged").sum(),
        "expired": (patient_log["outcome"] == "expired").sum(),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Hospital Digital Twin — SimPy Simulation")
    print("=" * 60)

    # Load models if available
    _load_models()

    # Load features
    features_path = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
    df_features = pd.read_csv(features_path)
    print(f"\nLoaded {len(df_features)} patient feature records")

    # ── Baseline scenario ──────────────────────────────────────────────────
    print("\n--- Baseline Scenario ---")
    print("  Beds: 20, Staff: 5, Arrival interval: 2h, Duration: 30 days")
    patient_log, queue_log = run_simulation(
        df_features, bed_capacity=20, staff_capacity=5,
        sim_days=30, mean_interarrival_hours=2.0
    )

    os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
    patient_log.to_csv(os.path.join(PROJECT_ROOT, "outputs", "simulation_log.csv"), index=False)
    queue_log.to_csv(os.path.join(PROJECT_ROOT, "outputs", "queue_log.csv"), index=False)

    kpis = compute_kpis(patient_log, bed_capacity=20, sim_days=30)
    print("\n  KPIs:")
    for k, v in kpis.items():
        print(f"    {k}: {v}")

    # ── What-if: Surge scenario (halve interarrival time) ──────────────────
    print("\n--- What-If: Surge Scenario ---")
    print("  Beds: 20, Staff: 5, Arrival interval: 1h (doubled arrivals)")
    surge_log, _ = run_simulation(
        df_features, bed_capacity=20, staff_capacity=5,
        sim_days=30, mean_interarrival_hours=1.0
    )
    surge_kpis = compute_kpis(surge_log, bed_capacity=20, sim_days=30)
    print("\n  KPIs:")
    for k, v in surge_kpis.items():
        print(f"    {k}: {v}")

    # ── What-if: Bed capacity variations ───────────────────────────────────
    print("\n--- What-If: Bed Capacity Variations ---")
    for beds in [10, 20, 30]:
        cap_log, _ = run_simulation(
            df_features, bed_capacity=beds, staff_capacity=5,
            sim_days=30, mean_interarrival_hours=2.0
        )
        cap_kpis = compute_kpis(cap_log, bed_capacity=beds, sim_days=30)
        print(f"  {beds} beds -> occupancy={cap_kpis.get('bed_occupancy', 'N/A')}, "
              f"avg_wait={cap_kpis.get('avg_wait_hours', 'N/A')}h, "
              f"throughput={cap_kpis.get('throughput_per_day', 'N/A')}/day")

    print("\n[OK] Simulation complete! Results saved to outputs/")
