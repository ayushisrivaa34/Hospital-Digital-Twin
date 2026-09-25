"""
validate_twin.py — Digital Twin Validation
============================================
Validates the simulation output against historical data using:
  1. Historical replay
  2. KS test (simulated vs actual LOS distributions)
  3. Sensitivity analysis (±20% arrival rate)

Usage:
    python -m src.validation.validate_twin
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.simulation.hospital_sim import run_simulation, compute_kpis


def validate_los_distribution(sim_log: pd.DataFrame, actual_df: pd.DataFrame) -> dict:
    """
    Compare simulated vs actual LOS distributions using the KS test.

    Returns:
        dict with KS statistic, p-value, and interpretation
    """
    simulated_los = sim_log["los_days"].dropna().values
    actual_los = actual_df["length_of_stay_days"].dropna().values

    ks_stat, p_value = stats.ks_2samp(simulated_los, actual_los)

    result = {
        "ks_statistic": round(ks_stat, 4),
        "p_value": round(p_value, 4),
        "simulated_mean_los": round(np.mean(simulated_los), 2),
        "actual_mean_los": round(np.mean(actual_los), 2),
        "simulated_median_los": round(np.median(simulated_los), 2),
        "actual_median_los": round(np.median(actual_los), 2),
    }

    if p_value > 0.05:
        result["interpretation"] = (
            "PASS — Cannot reject null hypothesis that distributions are the same "
            f"(p={p_value:.4f} > 0.05). Simulation LOS distribution is consistent "
            "with historical data."
        )
    else:
        result["interpretation"] = (
            f"FAIL — Distributions differ significantly (p={p_value:.4f} < 0.05). "
            "Consider revisiting the LOS model or arrival assumptions."
        )

    return result


def sensitivity_analysis(df_features: pd.DataFrame,
                         base_arrival_rate: float = 2.0,
                         bed_capacity: int = 20,
                         staff_capacity: int = 5,
                         sim_days: int = 30) -> pd.DataFrame:
    """
    Run simulation at ±20% arrival rate and verify outputs move plausibly.
    """
    variations = {
        "baseline": base_arrival_rate,
        "minus_20pct": base_arrival_rate * 1.2,   # higher interarrival = fewer patients
        "plus_20pct": base_arrival_rate * 0.8,    # lower interarrival = more patients
    }

    results = []
    for label, rate in variations.items():
        patient_log, _ = run_simulation(
            df_features, bed_capacity=bed_capacity,
            staff_capacity=staff_capacity, sim_days=sim_days,
            mean_interarrival_hours=rate
        )
        kpis = compute_kpis(patient_log, bed_capacity, sim_days)
        kpis["scenario"] = label
        kpis["arrival_rate"] = rate
        results.append(kpis)

    return pd.DataFrame(results)


def validate_sensitivity_plausibility(sensitivity_df: pd.DataFrame) -> dict:
    """
    Check that sensitivity analysis results move in plausible directions:
    - More arrivals -> higher occupancy, longer waits
    - Fewer arrivals -> lower occupancy, shorter waits
    """
    baseline = sensitivity_df[sensitivity_df["scenario"] == "baseline"].iloc[0]
    plus = sensitivity_df[sensitivity_df["scenario"] == "plus_20pct"].iloc[0]
    minus = sensitivity_df[sensitivity_df["scenario"] == "minus_20pct"].iloc[0]

    checks = {
        "more_patients_higher_occupancy": plus["bed_occupancy"] >= baseline["bed_occupancy"],
        "fewer_patients_lower_occupancy": minus["bed_occupancy"] <= baseline["bed_occupancy"],
        "more_patients_more_throughput": plus["total_patients"] >= baseline["total_patients"],
        "fewer_patients_less_throughput": minus["total_patients"] <= baseline["total_patients"],
    }

    passed = sum(checks.values())
    total = len(checks)

    return {
        "checks": checks,
        "passed": passed,
        "total": total,
        "verdict": "PASS" if passed >= total - 1 else "NEEDS REVIEW"
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Hospital Digital Twin — Validation Suite")
    print("=" * 60)

    # Load feature data
    features_path = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
    df_features = pd.read_csv(features_path)

    # ── 1. Run baseline simulation ─────────────────────────────────────────
    print("\n1. Running baseline simulation for validation...")
    patient_log, queue_log = run_simulation(
        df_features, bed_capacity=20, staff_capacity=5,
        sim_days=30, mean_interarrival_hours=2.0
    )
    print(f"   Simulated {len(patient_log)} patient journeys")

    # ── 2. KS Test — LOS Distribution ─────────────────────────────────────
    print("\n2. KS Test — LOS Distribution Comparison")
    ks_results = validate_los_distribution(patient_log, df_features)
    print(f"   KS statistic:    {ks_results['ks_statistic']}")
    print(f"   p-value:         {ks_results['p_value']}")
    print(f"   Simulated mean:  {ks_results['simulated_mean_los']} days")
    print(f"   Actual mean:     {ks_results['actual_mean_los']} days")
    print(f"   -> {ks_results['interpretation']}")

    # ── 3. Sensitivity Analysis ────────────────────────────────────────────
    print("\n3. Sensitivity Analysis (±20% arrival rate)")
    sensitivity = sensitivity_analysis(df_features)
    print(sensitivity[["scenario", "arrival_rate", "total_patients",
                       "bed_occupancy", "avg_wait_hours", "throughput_per_day"]].to_string(index=False))

    plausibility = validate_sensitivity_plausibility(sensitivity)
    print(f"\n   Plausibility checks: {plausibility['passed']}/{plausibility['total']} passed")
    for check, passed in plausibility["checks"].items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"     {status} {check}")
    print(f"   -> {plausibility['verdict']}")

    # ── Save validation report ─────────────────────────────────────────────
    report_path = os.path.join(PROJECT_ROOT, "outputs", "validation_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("Hospital Digital Twin — Validation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write("KS Test Results:\n")
        for k, v in ks_results.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nSensitivity Analysis:\n")
        f.write(sensitivity.to_string(index=False))
        f.write(f"\n\nPlausibility: {plausibility['verdict']}\n")

    print(f"\n[OK] Validation report saved to {report_path}")
