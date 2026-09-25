# Hospital Digital Twin — Implementation Plan
*A practical build guide derived from the "Hospital Digital Twin: A PyCaret-Based Research Workflow Using Public Datasets" paper.*

This plan turns the paper's six-stage architecture into a concrete, buildable project. It's written so a human (or an AI coding agent like Antigravity) can follow it top to bottom without needing to re-derive anything from the paper.

---

## 0. What You're Building

A pipeline of 5 PyCaret models + 1 SimPy simulation, wired together like this:

```
Public Dataset → Preprocessing → PyCaret Models → SimPy Simulation → Dashboard/Report
```

The models predict things about patients (how long they'll stay, will they die, what "type" of patient they are, are they deteriorating, how many patients are coming). The simulation uses those predictions to run a virtual hospital forward in time so you can test "what if we had fewer beds / more nurses / a surge."

---

## 1. Pick Your Dataset First (Do This Before Anything Else)

The paper lists 7 datasets. **Don't try to use all of them.** Pick one to get a working system, then expand later.

### Recommended starting dataset: **MIMIC-IV Clinical Database Demo**

Why this one, specifically, and not full MIMIC-IV:
- The full MIMIC-IV requires PhysioNet credentialing + a course on human-subjects research (takes days to weeks for approval).
- The **MIMIC-IV Demo** (100 de-identified patients) is publicly downloadable **immediately, no credentialing**, and has the same table structure as the full dataset — so your code doesn't need to change later if you upgrade.
- It's small enough to run entirely on a laptop.

Fallback options if you want something even simpler to start coding against today:
- **NHANES** — CSV/XPT files, no approval needed at all, good for the clustering (phenotyping) task.
- A **synthetic dataset you generate yourself** (see §2.4) — useful for testing your pipeline code before real data access comes through.

**Action:** Download the MIMIC-IV Clinical Database Demo from PhysioNet. Also grab NHANES CSVs for the clustering task since it's simpler to work with.

---

## 2. Project Structure

Set this up first — every stage below writes into it.

```
hospital-digital-twin/
├── data/
│   ├── raw/                  # untouched downloaded files
│   ├── processed/            # cleaned, feature-engineered tables
│   └── synthetic/            # fallback synthetic data (see §2.4)
├── notebooks/                # exploration only — nothing production lives here
├── src/
│   ├── preprocessing/
│   │   ├── clean_ehr.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_los.py            # Task A: length-of-stay
│   │   ├── train_mortality.py      # Task B: mortality/readmission
│   │   ├── train_phenotypes.py     # Task C: clustering
│   │   ├── train_anomaly.py        # Task D: deterioration detection
│   │   └── train_forecast.py       # Task E: ED arrivals / bed occupancy
│   ├── simulation/
│   │   └── hospital_sim.py         # SimPy digital twin core
│   └── validation/
│       └── validate_twin.py
├── models/                    # saved PyCaret .pkl models land here
├── outputs/                   # simulation logs, forecasts, plots
├── requirements.txt
└── README.md
```

**Action:** Create this folder structure now, empty files are fine — you'll fill them in as you go.

---

## 3. Environment Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install pycaret[full]
pip install simpy
pip install pandas numpy matplotlib
pip install mlflow                 # PyCaret's experiment tracking, optional but recommended
```

**requirements.txt:**
```
pycaret[full]>=3.0
simpy>=4.1
pandas
numpy
matplotlib
mlflow
```

Note on `use_gpu=True` in the paper's code samples: only set this if you actually have a CUDA GPU set up. On a normal laptop, leave it as `use_gpu=False` (or omit it) or PyCaret setup will fail.

---

## 4. Build Order (Do Models in This Sequence, Not the Paper's Order)

The paper presents 5 modeling tasks side by side, but for **implementation** it's easier to build them in order of dependency and difficulty:

| Order | Task | Why this order |
|---|---|---|
| 1 | **Preprocessing pipeline** | Everything downstream needs clean feature tables |
| 2 | **Task A — LOS Regression** | Simplest task (regression), and its output feeds directly into the simulation |
| 3 | **Task B — Mortality Classification** | Second-simplest, also feeds the simulation directly |
| 4 | **Task C — Clustering (Phenotyping)** | Doesn't depend on A/B, good to parallelize if you have a second contributor |
| 5 | **Task E — Time-Series Forecasting** | Needed to parameterize patient arrivals in the simulation |
| 6 | **Task D — Anomaly Detection** | Most fiddly (needs vitals time-series, not just tabular admission data) — do last |
| 7 | **SimPy Simulation Layer** | Only buildable once Tasks A and B produce saved models |
| 8 | **Validation** | Only makes sense once simulation exists |

---

## 5. Stage-by-Stage Implementation

### 5.1 Preprocessing (`src/preprocessing/`)

Minimum viable feature table needs these columns per patient/admission:
- Demographics: age, sex
- Admission type (elective / emergency / urgent)
- Primary diagnosis (ICD code, or a simplified category)
- A severity/acuity score — compute a simplified version of SOFA or Charlson Comorbidity Index if the full clinical score is too complex to derive from the demo data
- Length of stay (target for Task A)
- In-hospital death flag (target for Task B)

**Action checklist:**
- [ ] Write `clean_ehr.py`: load raw admissions table, drop duplicates, handle missing values (median/mode imputation is fine for a first pass)
- [ ] Write `build_features.py`: compute LOS (discharge time − admit time), compute a simplified comorbidity count, one-hot or target-encode categorical fields
- [ ] Save output to `data/processed/features.csv`

### 5.2 Task A — Length-of-Stay Prediction

```python
from pycaret.regression import *
import pandas as pd

df = pd.read_csv("data/processed/features.csv")

reg_setup = setup(
    data=df,
    target="length_of_stay_days",
    session_id=42,
    normalize=True,
    fold=5
    # use_gpu=True   # only if you have a GPU set up
)

best_model = compare_models()
tuned_model = tune_model(best_model, n_iter=25)   # 50 iters is slow on CPU — start with 25
final_model = finalize_model(tuned_model)
save_model(final_model, "models/los_predictor")
```

**Action:** Run this, confirm `models/los_predictor.pkl` is created, note the MAE/RMSE from `compare_models()` output.

### 5.3 Task B — Mortality / Readmission Risk

```python
from pycaret.classification import *

clf_setup = setup(
    data=df,
    target="in_hospital_death",
    session_id=42,
    normalize=True,
    fix_imbalance=True,   # important — death events are rare
    fold=5
)

best = compare_models()
tuned = tune_model(best, optimize="AUC")
calibrated = calibrate_model(tuned)   # do NOT skip this step
save_model(finalize_model(calibrated), "models/mortality_risk_model")
```

**Why calibration matters (don't skip it):** an uncalibrated model might say "80% risk" when the real-world rate for similar patients is 40%. The simulation layer treats these probabilities as literal odds of an event happening — if they're uncalibrated, your whole simulation's mortality numbers will be wrong even if the model's *ranking* of risky patients is fine.

### 5.4 Task C — Patient Phenotyping (Clustering)

```python
from pycaret.clustering import *

cluster_setup = setup(
    data=df_patients,
    normalize=True,
    session_id=42,
    pca=True,
    pca_components=0.95
)

kmeans = create_model("kmeans", num_clusters=6)
plot_model(kmeans, plot="cluster")
plot_model(kmeans, plot="silhouette")

results = assign_model(kmeans)
results.to_csv("outputs/patient_phenotypes.csv", index=False)
```

**Action:** Try `num_clusters=4`, `6`, and `8` — pick whichever gives the best silhouette score *and* produces clusters a non-technical reader could name (e.g., "young low-acuity elective," "elderly high-comorbidity emergency").

### 5.5 Task E — Time-Series Forecasting (ED Arrivals)

```python
from pycaret.time_series import *

ts_setup = setup(
    data=ed_hourly_arrivals,   # needs a datetime index + a count column
    target="patient_count",
    session_id=42,
    fh=48
)

best_ts = compare_models()
final_ts = finalize_model(tune_model(best_ts))
predictions = predict_model(final_ts, fh=48)
save_model(final_ts, "models/ed_arrivals_forecaster")
```

Note: MIMIC-IV demo data may not have enough admissions to build a good hourly time series (100 patients is thin). If forecasts look noisy, this is expected — treat this model as a placeholder until you have access to eICU or HCUP NIS scale data.

### 5.6 Task D — Anomaly Detection (do this last)

```python
from pycaret.anomaly import *

ano_setup = setup(data=df_vitals, normalize=True, session_id=42)
iforest = create_model("iforest", fraction=0.05)
results = assign_model(iforest)
alerts = results[results["Anomaly"] == 1]
alerts.to_csv("outputs/deterioration_alerts.csv", index=False)
```

This needs ICU vitals time-series (heart rate, BP, SpO2, etc. over time), which the MIMIC-IV Demo has for its ICU subset — filter to `icustays` + `chartevents` tables specifically.

### 5.7 Simulation Layer (`src/simulation/hospital_sim.py`)

This is the paper's SimPy skeleton, made runnable:

```python
import simpy
import numpy as np
import pandas as pd
from pycaret.regression import load_model as load_reg_model
from pycaret.classification import load_model as load_clf_model

los_model = load_reg_model("models/los_predictor")
mortality_model = load_clf_model("models/mortality_risk_model")

def patient(env, name, ward, features_row, log):
    los_pred = los_model.predict(features_row)[0]
    mort_risk = mortality_model.predict_proba(features_row)[0][1] \
        if hasattr(mortality_model, "predict_proba") else 0.05

    with ward.beds.request() as bed:
        yield bed
        admit_time = env.now
        yield env.timeout(max(los_pred, 0.1) * 24)  # hours
        outcome = "expired" if np.random.random() < mort_risk else "discharged"
        log.append({"patient": name, "admit_time": admit_time,
                    "discharge_time": env.now, "los_days": los_pred,
                    "outcome": outcome})

def patient_generator(env, ward, df_historical, log, mean_interarrival_hours=2.0):
    i = 0
    while True:
        yield env.timeout(np.random.exponential(mean_interarrival_hours))
        features_row = df_historical.sample(1)
        env.process(patient(env, f"Patient_{i}", ward, features_row, log))
        i += 1

def run_simulation(df_historical, bed_capacity=20, sim_days=30, mean_interarrival_hours=2.0):
    env = simpy.Environment()
    ward = simpy.Resource(env, capacity=bed_capacity)
    log = []
    env.process(patient_generator(env, ward, df_historical, log, mean_interarrival_hours))
    env.run(until=24 * sim_days)
    return pd.DataFrame(log)

if __name__ == "__main__":
    df_features = pd.read_csv("data/processed/features.csv")
    result_log = run_simulation(df_features, bed_capacity=20, sim_days=30)
    result_log.to_csv("outputs/simulation_log.csv", index=False)
    print(result_log.describe())
```

**What-if scenarios to run once this works** (straight from the paper's §6.2):
- Double `mean_interarrival_hours` sensitivity → simulate surge (halve the interarrival time instead)
- Vary `bed_capacity` (10, 20, 30) → find where queue length blows up
- Vary nurse-to-patient ratio (you'll need to add a `staff` Resource alongside `beds`)

### 5.8 Validation (`src/validation/validate_twin.py`)

Minimum bar before you call this "validated":
1. **Historical replay** — run the sim using real historical arrival timestamps instead of the exponential generator, compare simulated vs. actual LOS distributions.
2. **KS test** — `scipy.stats.ks_2samp(simulated_los, actual_los)` — if p-value is very low, your simulation doesn't match reality and you need to revisit the LOS model or arrival assumptions.
3. **Sensitivity check** — rerun with arrival rate ±20% and confirm outputs move in a plausible direction, not randomly.

---

## 6. What to Skip on a First Pass

To keep this achievable, deliberately skip these from the paper until the core loop works end-to-end:
- Multi-dataset fusion (SEER, UK Biobank, HCUP) — single dataset only, for now
- `stack_models()` ensembling — use `tune_model()` output directly first
- OMOP CDM alignment — nice for multi-site replication later, irrelevant for a single-dataset prototype
- MLflow experiment tracking — useful, but optional scaffolding, not core functionality
- Simio/AnyLogic — SimPy is sufficient and free; only consider commercial tools if you need polished visualization for a stakeholder demo

---

## 7. Milestone Checklist (Use This to Track Progress)

- [ ] Project structure created, environment installed
- [ ] MIMIC-IV Demo (or NHANES) downloaded into `data/raw/`
- [ ] `features.csv` produced by preprocessing scripts
- [ ] LOS model trained, saved, MAE recorded
- [ ] Mortality model trained, calibrated, saved, AUROC recorded
- [ ] Clustering run, cluster count chosen, phenotypes exported
- [ ] ED arrivals forecaster trained (or explicitly deferred if data is too thin)
- [ ] Anomaly detector trained on ICU vitals subset
- [ ] `hospital_sim.py` runs end-to-end and produces `simulation_log.csv`
- [ ] At least one what-if scenario run and compared to baseline
- [ ] Historical replay + KS test performed
- [ ] README written summarizing findings

---

## 8. Notes for an AI Coding Agent Picking This Up

If you're an agent implementing this from this document:
- Work stage by stage in the order given in §4 — later stages import saved models from earlier ones (`models/*.pkl`), so out-of-order execution will fail on missing files.
- All file paths above are relative to the project root `hospital-digital-twin/`.
- Every PyCaret `setup()` call needs a pandas DataFrame already free of leakage — e.g., never include `length_of_stay_days` as a feature when training the mortality model, and vice versa.
- If MIMIC-IV Demo access isn't available yet, generate a synthetic dataset with the same column names/types described in §5.1 so all downstream code paths can be built and tested against placeholder data, then swapped for real data later.
- Do not attempt to re-identify any patient-level data, and do not commit raw clinical data files to version control — add `data/raw/` and `data/processed/` to `.gitignore`.
