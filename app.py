"""
app.py — Interactive Streamlit Dashboard & 2D Floor Map
========================================================
A web application presenting the Hospital Digital Twin findings:
  - Control Panel Sidebar: sliders for bed capacity, staff size, arrival rates
  - KPI Metrics: occupancy, LOS, mortality, throughput
  - Visualizations: Plotly charts for bed utilization and queue lengths
  - 2D Floor Map: live grid showing open (green) vs occupied/critical (red) beds

From §5.9 of the implementation document.

Usage:
    streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import simpy

# ── Project root resolution ───────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital Digital Twin",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for premium look ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .stMetric {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .stMetric label { color: #a0c4e8 !important; font-size: 0.85rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 600 !important; }
    .stMetric [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1b2d 0%, #1a2744 100%);
    }

    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 {
        color: #60a5fa !important;
    }

    h1, h2, h3 { font-family: 'Inter', sans-serif; }

    .floor-map-title {
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        color: #60a5fa;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE (inline, lightweight version for real-time dashboard)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_features():
    """Load patient feature data."""
    path = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    # Generate inline if not found
    return _generate_inline_features()


def _generate_inline_features():
    """Quick inline synthetic data if features.csv doesn't exist yet."""
    np.random.seed(42)
    n = 500
    data = {
        "age": np.clip(np.random.normal(60, 16, n).astype(int), 18, 99),
        "sex_numeric": np.random.binomial(1, 0.5, n),
        "admission_type_ordinal": np.random.choice([0, 1, 2], n, p=[0.2, 0.25, 0.55]),
        "severity_score": np.clip(np.random.exponential(2.5, n), 0, 10).round(2),
        "num_comorbidities": np.random.poisson(1.5, n).clip(0, 8),
        "length_of_stay_days": np.clip(np.random.exponential(5, n) + 1, 0.5, 60).round(2),
        "in_hospital_death": np.random.binomial(1, 0.06, n),
    }
    return pd.DataFrame(data)


def run_quick_simulation(df, bed_capacity, staff_capacity, sim_days,
                         mean_interarrival_hours, seed=42):
    """
    Lightweight simulation for the dashboard (no PyCaret dependency).
    Uses data-driven LOS and severity-based mortality estimates.
    """
    np.random.seed(seed)
    env = simpy.Environment()
    ward = simpy.PriorityResource(env, capacity=bed_capacity)
    staff_pool = simpy.Resource(env, capacity=staff_capacity)
    log = []
    hourly_occupancy = []
    hourly_queue = []

    def _monitor(env, ward, interval=1.0):
        """Record occupancy and queue length every hour."""
        while True:
            hourly_occupancy.append({
                "time": env.now,
                "occupied_beds": ward.count,
                "available_beds": bed_capacity - ward.count,
                "queue_length": len(ward.queue),
            })
            yield env.timeout(interval)

    def _patient(env, name, ward, staff_pool, features_row):
        arrival = env.now
        severity = features_row.get("severity_score", 3.0)
        los = features_row.get("length_of_stay_days", 5.0)
        mort_risk = min(0.02 + severity * 0.05, 0.6)
        priority = 1 if mort_risk > 0.4 else 2

        with ward.request(priority=priority) as bed:
            yield bed
            admit = env.now

            with staff_pool.request() as staff:
                yield staff
                yield env.timeout(0.5)

            yield env.timeout(max(los, 0.1) * 24)
            outcome = "expired" if np.random.random() < mort_risk else "discharged"
            log.append({
                "patient": name, "arrival": arrival, "admit": admit,
                "discharge": env.now, "wait_hours": admit - arrival,
                "los_days": los, "mortality_risk": mort_risk,
                "outcome": outcome, "priority": priority,
                "severity": severity,
            })

    def _generator(env, ward, staff_pool, df):
        i = 0
        while True:
            yield env.timeout(np.random.exponential(mean_interarrival_hours))
            row = df.sample(1).iloc[0].to_dict()
            env.process(_patient(env, f"P{i}", ward, staff_pool, row))
            i += 1

    env.process(_generator(env, ward, staff_pool, df))
    env.process(_monitor(env, ward))
    env.run(until=24 * sim_days)

    return pd.DataFrame(log), pd.DataFrame(hourly_occupancy)


def build_floor_map(occupied: int, critical: int, total_beds: int):
    """
    Build a 2D floor map grid showing bed status.
    Green = open, Orange = occupied (stable), Red = critical
    """
    cols = min(10, total_beds)
    rows = max(1, (total_beds + cols - 1) // cols)
    total_cells = rows * cols

    # Assign statuses
    statuses = []
    for i in range(total_cells):
        if i >= total_beds:
            statuses.append("empty")
        elif i < critical:
            statuses.append("critical")
        elif i < occupied:
            statuses.append("occupied")
        else:
            statuses.append("open")

    color_map = {
        "open": "#22c55e",       # green
        "occupied": "#f59e0b",   # amber
        "critical": "#ef4444",   # red
        "empty": "#1e293b",      # dark bg
    }

    z = []
    text = []
    for r in range(rows):
        z_row = []
        text_row = []
        for c in range(cols):
            idx = r * cols + c
            status = statuses[idx]
            z_row.append(
                0 if status == "empty" else
                1 if status == "open" else
                2 if status == "occupied" else 3
            )
            if status == "empty":
                text_row.append("")
            else:
                bed_num = idx + 1
                text_row.append(f"Bed {bed_num}<br>{status.title()}")
        z.append(z_row)
        text.append(text_row)

    colorscale = [
        [0, "#1e293b"],     # empty
        [0.33, "#22c55e"],  # open
        [0.66, "#f59e0b"],  # occupied
        [1.0, "#ef4444"],   # critical
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"},
        colorscale=colorscale,
        zmin=0, zmax=3,
        showscale=False,
        xgap=3, ygap=3,
        hovertemplate="%{text}<extra></extra>"
    ))

    fig.update_layout(
        height=max(200, rows * 60),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed"),
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <h1 style='background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 2.5rem; font-weight: 700; margin-bottom: 0.2rem;'>
            🏥 Hospital Digital Twin
        </h1>
        <p style='color: #94a3b8; font-size: 1rem;'>
            SimPy-Powered Simulation · PyCaret ML Models · Real-Time Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar Controls ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        st.markdown("---")

        st.markdown("### 🛏️ Capacity")
        bed_capacity = st.slider("Bed Capacity", 5, 100, 20, 5)
        staff_capacity = st.slider("Staff (Nurses/Doctors)", 1, 30, 5, 1)

        st.markdown("### 🚑 Arrivals")
        arrival_rate = st.slider("Mean Arrival Interval (hours)",
                                 0.5, 8.0, 2.0, 0.5)

        st.markdown("### ⏱️ Simulation")
        sim_days = st.slider("Simulation Duration (days)", 7, 90, 30, 7)

        st.markdown("---")
        run_btn = st.button("▶️ Run Simulation", use_container_width=True,
                           type="primary")

        st.markdown("---")
        st.markdown("### 📊 Scenario Comparison")
        compare_enabled = st.checkbox("Compare with what-if scenario")
        if compare_enabled:
            whatif_beds = st.slider("What-If Beds", 5, 100, 30, 5,
                                   key="whatif_beds")
            whatif_staff = st.slider("What-If Staff", 1, 30, 8, 1,
                                    key="whatif_staff")
            whatif_arrival = st.slider("What-If Arrival Interval (h)",
                                      0.5, 8.0, 1.0, 0.5,
                                      key="whatif_arrival")

    # ── Load data ──────────────────────────────────────────────────────────
    df_features = load_features()

    # ── Run simulation ─────────────────────────────────────────────────────
    if run_btn or "sim_results" not in st.session_state:
        with st.spinner("🔄 Running simulation..."):
            patient_log, occupancy_log = run_quick_simulation(
                df_features, bed_capacity, staff_capacity,
                sim_days, arrival_rate
            )
            st.session_state.sim_results = (patient_log, occupancy_log)
            st.session_state.sim_params = {
                "beds": bed_capacity, "staff": staff_capacity,
                "arrival": arrival_rate, "days": sim_days
            }

            if compare_enabled:
                whatif_log, whatif_occ = run_quick_simulation(
                    df_features, whatif_beds, whatif_staff,
                    sim_days, whatif_arrival, seed=99
                )
                st.session_state.whatif_results = (whatif_log, whatif_occ)
            else:
                st.session_state.whatif_results = None

    patient_log, occupancy_log = st.session_state.sim_results
    params = st.session_state.sim_params

    if patient_log.empty:
        st.warning("No patients processed in this simulation. Try adjusting parameters.")
        return

    # ── KPI Metrics Row ────────────────────────────────────────────────────
    total_patients = len(patient_log)
    avg_los = patient_log["los_days"].mean()
    avg_wait = patient_log["wait_hours"].mean()
    mortality_rate = (patient_log["outcome"] == "expired").mean()
    total_bed_hours = patient_log["los_days"].sum() * 24
    available_hours = params["beds"] * params["days"] * 24
    occupancy_rate = min(total_bed_hours / available_hours, 1.0)
    throughput = total_patients / params["days"]

    # Compute deltas vs what-if if available
    delta_suffix = None
    if st.session_state.get("whatif_results"):
        wf_log, _ = st.session_state.whatif_results
        if not wf_log.empty:
            delta_suffix = {
                "patients": len(wf_log) - total_patients,
                "occupancy": (min(wf_log["los_days"].sum() * 24 /
                             (whatif_beds * params["days"] * 24), 1.0) - occupancy_rate),
                "wait": wf_log["wait_hours"].mean() - avg_wait,
                "mortality": (wf_log["outcome"] == "expired").mean() - mortality_rate,
            }

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Patients", f"{total_patients:,}",
                  delta=f"{delta_suffix['patients']:+,}" if delta_suffix else None)
    with col2:
        st.metric("Bed Occupancy", f"{occupancy_rate:.0%}",
                  delta=f"{delta_suffix['occupancy']:+.0%}" if delta_suffix else None,
                  delta_color="inverse")
    with col3:
        st.metric("Avg LOS", f"{avg_los:.1f} days")
    with col4:
        st.metric("Avg Wait", f"{avg_wait:.1f} hrs",
                  delta=f"{delta_suffix['wait']:+.1f}h" if delta_suffix else None,
                  delta_color="inverse")
    with col5:
        st.metric("Mortality Rate", f"{mortality_rate:.1%}",
                  delta=f"{delta_suffix['mortality']:+.1%}" if delta_suffix else None,
                  delta_color="inverse")
    with col6:
        st.metric("Throughput", f"{throughput:.0f}/day")

    st.markdown("---")

    # ── Main visualizations ────────────────────────────────────────────────
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("### 🛏️ Bed Utilization Over Time")
        if not occupancy_log.empty:
            fig_occ = go.Figure()
            fig_occ.add_trace(go.Scatter(
                x=occupancy_log["time"] / 24,
                y=occupancy_log["occupied_beds"],
                mode="lines",
                name="Occupied Beds",
                line=dict(color="#f59e0b", width=2),
                fill="tozeroy",
                fillcolor="rgba(245, 158, 11, 0.15)"
            ))
            fig_occ.add_trace(go.Scatter(
                x=occupancy_log["time"] / 24,
                y=occupancy_log["available_beds"],
                mode="lines",
                name="Available Beds",
                line=dict(color="#22c55e", width=2, dash="dot"),
            ))
            fig_occ.add_hline(y=params["beds"], line_dash="dash",
                             line_color="#ef4444", annotation_text="Max Capacity")
            fig_occ.update_layout(
                xaxis_title="Day",
                yaxis_title="Beds",
                template="plotly_dark",
                height=350,
                margin=dict(l=40, r=20, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig_occ, use_container_width=True)

    with viz_col2:
        st.markdown("### 📈 Queue Length Over Time")
        if not occupancy_log.empty:
            fig_q = go.Figure()
            fig_q.add_trace(go.Scatter(
                x=occupancy_log["time"] / 24,
                y=occupancy_log["queue_length"],
                mode="lines",
                name="Queue Length",
                line=dict(color="#a78bfa", width=2),
                fill="tozeroy",
                fillcolor="rgba(167, 139, 250, 0.15)"
            ))
            fig_q.update_layout(
                xaxis_title="Day",
                yaxis_title="Patients Waiting",
                template="plotly_dark",
                height=350,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig_q, use_container_width=True)

    # ── Second row: Distributions ──────────────────────────────────────────
    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        st.markdown("### 📊 LOS Distribution")
        fig_los = px.histogram(
            patient_log, x="los_days", nbins=30,
            color_discrete_sequence=["#60a5fa"],
            template="plotly_dark",
            labels={"los_days": "Length of Stay (days)"}
        )
        fig_los.update_layout(
            height=300,
            margin=dict(l=40, r=20, t=30, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_los, use_container_width=True)

    with dist_col2:
        st.markdown("### ⏳ Wait Time Distribution")
        fig_wait = px.histogram(
            patient_log, x="wait_hours", nbins=30,
            color_discrete_sequence=["#f472b6"],
            template="plotly_dark",
            labels={"wait_hours": "Wait Time (hours)"}
        )
        fig_wait.update_layout(
            height=300,
            margin=dict(l=40, r=20, t=30, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_wait, use_container_width=True)

    st.markdown("---")

    # ── 2D Floor Map ───────────────────────────────────────────────────────
    st.markdown('<div class="floor-map-title">🗺️ 2D Hospital Floor Map — '
                'Live Bed Status</div>', unsafe_allow_html=True)

    # Use the last snapshot from occupancy log
    if not occupancy_log.empty:
        last = occupancy_log.iloc[-1]
        occupied = int(last["occupied_beds"])
        in_queue = int(last["queue_length"])
        # Estimate critical beds as ~20% of occupied
        critical = min(int(occupied * 0.2), occupied)
    else:
        occupied = int(params["beds"] * 0.7)
        critical = int(occupied * 0.2)

    floor_fig = build_floor_map(occupied, critical, params["beds"])
    st.plotly_chart(floor_fig, use_container_width=True)

    # Legend
    leg1, leg2, leg3, leg4 = st.columns(4)
    with leg1:
        st.markdown("🟢 **Open** — Available bed")
    with leg2:
        st.markdown("🟡 **Occupied** — Stable patient")
    with leg3:
        st.markdown("🔴 **Critical** — High-risk patient")
    with leg4:
        st.markdown(f"📋 **Queue**: {in_queue if not occupancy_log.empty else 0} waiting")

    st.markdown("---")

    # ── Outcome breakdown ──────────────────────────────────────────────────
    out_col1, out_col2, out_col3 = st.columns([1, 1, 1])

    with out_col1:
        st.markdown("### 🎯 Patient Outcomes")
        outcome_counts = patient_log["outcome"].value_counts()
        fig_out = px.pie(
            values=outcome_counts.values,
            names=outcome_counts.index,
            color_discrete_sequence=["#22c55e", "#ef4444"],
            template="plotly_dark",
            hole=0.4
        )
        fig_out.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_out, use_container_width=True)

    with out_col2:
        st.markdown("### ⚡ Triage Priority Split")
        priority_counts = patient_log["priority"].value_counts().sort_index()
        labels = {1: "High Priority", 2: "Standard"}
        fig_pri = px.bar(
            x=[labels.get(k, str(k)) for k in priority_counts.index],
            y=priority_counts.values,
            color=[labels.get(k, str(k)) for k in priority_counts.index],
            color_discrete_map={"High Priority": "#ef4444", "Standard": "#60a5fa"},
            template="plotly_dark",
            labels={"x": "Priority Level", "y": "Count"}
        )
        fig_pri.update_layout(height=300, margin=dict(l=40, r=20, t=20, b=40),
                             showlegend=False)
        st.plotly_chart(fig_pri, use_container_width=True)

    with out_col3:
        st.markdown("### 📉 Severity Distribution")
        if "severity" in patient_log.columns:
            fig_sev = px.histogram(
                patient_log, x="severity", nbins=20,
                color_discrete_sequence=["#a78bfa"],
                template="plotly_dark",
                labels={"severity": "Severity Score"}
            )
            fig_sev.update_layout(height=300, margin=dict(l=40, r=20, t=20, b=40),
                                 showlegend=False)
            st.plotly_chart(fig_sev, use_container_width=True)

    # ── Cost Analysis (§5.8) ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💰 Cost & Resource Optimization")

    cost_col1, cost_col2 = st.columns(2)

    with cost_col1:
        bed_cost_per_day = 500  # simplified per-bed daily cost
        staff_cost_per_day = 300  # simplified per-staff daily cost
        total_bed_cost = params["beds"] * params["days"] * bed_cost_per_day
        total_staff_cost = params["staff"] * params["days"] * staff_cost_per_day
        total_cost = total_bed_cost + total_staff_cost
        cost_per_patient = total_cost / max(total_patients, 1)

        st.metric("Total Operating Cost", f"${total_cost:,.0f}")
        st.metric("Cost per Patient", f"${cost_per_patient:,.0f}")

        # Efficiency ratio
        efficiency = total_patients / (total_cost / 1000)
        st.metric("Efficiency (patients per $1K)", f"{efficiency:.2f}")

    with cost_col2:
        # Optimal staffing ratio analysis
        st.markdown("**Staffing Ratio Analysis**")
        ratios = []
        for staff in range(max(1, params["staff"] - 3), params["staff"] + 4):
            ratio_log, _ = run_quick_simulation(
                df_features, params["beds"], staff,
                min(params["days"], 14), arrival_rate, seed=42
            )
            if not ratio_log.empty:
                wait = ratio_log["wait_hours"].mean()
                cost = (params["beds"] * bed_cost_per_day + staff * staff_cost_per_day) * min(params["days"], 14)
                ratios.append({"staff": staff, "avg_wait_hours": wait,
                              "total_cost": cost, "ratio": f"1:{params['beds']//staff}"})

        if ratios:
            ratio_df = pd.DataFrame(ratios)
            fig_ratio = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ratio.add_trace(
                go.Bar(x=ratio_df["staff"], y=ratio_df["avg_wait_hours"],
                       name="Avg Wait (hrs)", marker_color="#f59e0b"),
                secondary_y=False
            )
            fig_ratio.add_trace(
                go.Scatter(x=ratio_df["staff"], y=ratio_df["total_cost"],
                          name="Total Cost ($)", line=dict(color="#ef4444", width=2)),
                secondary_y=True
            )
            fig_ratio.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=40, r=40, t=20, b=40),
                xaxis_title="Staff Count",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            fig_ratio.update_yaxes(title_text="Avg Wait (hrs)", secondary_y=False)
            fig_ratio.update_yaxes(title_text="Cost ($)", secondary_y=True)
            st.plotly_chart(fig_ratio, use_container_width=True)

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color: #64748b; padding: 1rem;'>
        <p>Hospital Digital Twin · Powered by SimPy + PyCaret · 
        Built from research paper methodology</p>
        <p style='font-size: 0.8rem;'>
            Data is synthetic — for demonstration and research purposes only
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
