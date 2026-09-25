# 🏥 Hospital Digital Twin

A **SimPy-powered simulation** and **PyCaret ML pipeline** that creates a digital twin of a hospital ward, enabling data-driven capacity planning, resource optimization, and what-if scenario analysis.

Built following: *"Hospital Digital Twin: A PyCaret-Based Research Workflow Using Public Datasets"* (`ImplementationGemini.docx`).

---

## 🚀 Quick Start

### 1. Set up the environment

```bash
cd hospital-digital-twin
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### 2. Generate synthetic data & run preprocessing

```bash
python -m src.preprocessing.generate_synthetic
python -m src.preprocessing.clean_ehr
python -m src.preprocessing.build_features
```

This creates `data/processed/features.csv` with 500 patient records across 27 clinical features.

### 3. Train all 5 PyCaret ML models (in order)

```bash
python -m src.models.train_los           # Task A: Length-of-Stay prediction (Regression)
python -m src.models.train_mortality     # Task B: Mortality risk (Classification + Calibration)
python -m src.models.train_phenotypes    # Task C: Patient phenotyping (Clustering)
python -m src.models.train_forecast      # Task E: ED arrival forecasting (Time-Series)
python -m src.models.train_anomaly       # Task D: Anomaly/deterioration detection (Anomaly Detection)
```

### 4. Run the SimPy simulation

```bash
python -m src.simulation.hospital_sim
```

### 5. Validate the digital twin

```bash
python -m src.validation.validate_twin
```

### 6. Launch the interactive dashboard

```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 📊 Trained Models & Empirical Results

All 5 PyCaret machine learning models are trained and saved in `models/`:

| Task | Category | Target / Objective | Best Model | Key Metrics | Saved Model File |
|------|----------|-------------------|------------|-------------|-----------------|
| **Task A** | Regression | Length of Stay (days) | Bayesian Ridge / Tuned | **MAE = 1.65 days**, R² = 0.51 | `models/los_predictor.pkl` |
| **Task B** | Classification | In-hospital mortality | Logistic Regression (Calibrated) | **AUC = 0.69**, Recall = 0.44 | `models/mortality_risk_model.pkl` |
| **Task C** | Clustering | Patient Phenotypes | K-Means (PCA 95%) | **k = 4 clusters**, Silhouette = 0.08 | `models/phenotype_model.pkl` |
| **Task E** | Time-Series | Hourly ED arrivals | AutoETS / Forecaster | **MASE = 0.66**, RMSSE = 0.67 | `models/ed_arrivals_forecaster.pkl` |
| **Task D** | Anomaly | Patient deterioration | Isolation Forest (5% fraction) | **Precision = 97%**, **Recall = 100%**, F1 = 0.98 | `models/anomaly_detector.pkl` |

---

## 🧪 Digital Twin Validation Results

Validation confirmed statistical fidelity via Kolmogorov-Smirnov test and 4-way sensitivity monotonicity checks:

```
Hospital Digital Twin - Validation Report
==================================================

KS Test (Simulated vs Actual LOS):
  ks_statistic:         0.1323
  p_value:              0.3789  (PASS: p > 0.05, cannot reject null hypothesis)
  simulated_mean_los:   9.27 days
  actual_mean_los:      9.81 days
  simulated_median_los: 8.69 days
  actual_median_los:    9.44 days
  Result:               PASS - Simulation LOS distribution matches historical data.

Sensitivity Analysis (Monotonicity Checks):
  - +20% arrival rate  -> higher bed occupancy (79.7% vs 75.7%) & higher throughput (PASS)
  - -20% arrival rate  -> lower bed occupancy (75.4% vs 75.7%) & lower throughput (PASS)
  Plausibility Check:   PASS (4/4 checks passed)
```

---

## 📁 Project Architecture & Directory Structure

```
hospital-digital-twin/
├── data/
│   ├── raw/                  # Real clinical data (e.g. MIMIC-IV demo)
│   ├── processed/            # Cleaned, feature-engineered tables (features.csv)
│   └── synthetic/            # Synthetic admissions, vitals, hourly arrivals
├── src/
│   ├── preprocessing/
│   │   ├── generate_synthetic.py   # Synthetic data generator (500 pts, 24k vitals)
│   │   ├── clean_ehr.py            # Missing value imputation & validation
│   │   └── build_features.py       # Clinical feature engineering (27 columns)
│   ├── models/
│   │   ├── train_los.py            # Task A: LOS regression
│   │   ├── train_mortality.py      # Task B: Mortality classification + calibration
│   │   ├── train_phenotypes.py     # Task C: Patient clustering
│   │   ├── train_anomaly.py        # Task D: Anomaly detection on ICU vitals
│   │   └── train_forecast.py       # Task E: ED arrival forecasting
│   ├── simulation/
│   │   └── hospital_sim.py         # SimPy digital twin (PriorityResource triage + staff pool)
│   └── validation/
│       └── validate_twin.py        # KS test & sensitivity analysis
├── models/                         # Saved PyCaret .pkl model artifacts
│   ├── los_predictor.pkl
│   ├── mortality_risk_model.pkl
│   ├── phenotype_model.pkl
│   ├── anomaly_detector.pkl
│   └── ed_arrivals_forecaster.pkl
├── outputs/                        # Simulation logs, forecasts, reports
│   ├── simulation_log.csv
│   ├── queue_log.csv
│   ├── patient_phenotypes.csv
│   ├── deterioration_alerts.csv
│   ├── ed_arrival_forecast.csv
│   └── validation_report.txt
├── app.py                          # Streamlit web dashboard with 2D floor map
├── requirements.txt
└── README.md
```

---

## 🕹️ Simulation & Web Dashboard Features

- **Triage Queuing (`simpy.PriorityResource`)**: High-risk patients (`mortality_risk > 0.4`) automatically receive Priority 1 admission.
- **Dedicated Staff Pool (`simpy.Resource`)**: Models doctor/nurse availability as an independent operational bottleneck alongside physical bed capacity.
- **Interactive Control Panel**: Real-time parameter sliders in the sidebar for bed capacity, staff size, simulation duration, and arrival rates.
- **2D Hospital Ward Floor Map**: Visual bed status grid dynamically indicating open beds (green), occupied beds (amber), and critical triage beds (red).
- **Cost & Resource Optimization**: Calculates operational cost per patient day, staffing cost vs revenue throughput, and identifies bottleneck resources.
- **Scenario Comparison**: Toggle what-if scenarios (e.g., patient surge, capacity reduction) side-by-side with baseline metrics.

---

## 🛡️ Data & Governance Note

This project is pre-configured with synthetic data for local experimentation and zero-credential deployment. To run against PhysioNet data:
1. Obtain PhysioNet credentialing for **MIMIC-IV Clinical Database Demo**.
2. Place demo tables in `data/raw/`.
3. Re-run `clean_ehr.py` and `build_features.py`.

*Never commit real clinical data or patient identifiers to version control.*
