import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from pathlib import Path
import sys

# Add src/ to Python path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.data_generation import save_synthetic_data
from src.modeling import load_data_and_train
from src.fda_summary import generate_fda_style_summary

DATA_PATH = BASE_DIR / "data" / "synthetic_pupillometry.csv"

st.set_page_config(
    page_title="NeurOptics Pupillometry LLM System",
    layout="wide"
)

# -----------------------------
# Load or generate dataset
# -----------------------------
@st.cache_data
def load_or_create_data():
    if not DATA_PATH.exists():
        save_synthetic_data()
    return pd.read_csv(DATA_PATH)

# -----------------------------
# Train models
# -----------------------------
@st.cache_resource
def train_all_models(df):
    from src.modeling import train_models
    return df, train_models(df)

df = load_or_create_data()
df_all, model_results = train_all_models(df)
models_dict = model_results["models"]

st.title("GenAI‑Enhanced Pupillometry ML Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset Overview",
    "Model Performance",
    "Patient Explorer",
    "Narrative Summary (FDA)"
])

# ============================================================
# 1. DATASET OVERVIEW
# ============================================================
with tab1:
    st.subheader("Synthetic pupillometry dataset modeled after NeurOptics clinical device outputs. First 50 rows of data")
    st.dataframe(df_all.head(50))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### NPi Distribution")
        fig_npi = px.histogram(df_all, x="npi", nbins=30)
        st.plotly_chart(fig_npi, use_container_width=True)

        st.markdown("### Age Distribution")
        fig_age = px.histogram(df_all, x="age", nbins=30)
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        st.markdown("### GCS vs NPi")
        fig_gcs_npi = px.scatter(
            df_all,
            x="npi",
            y="gcs",
            color="severity",
            opacity=0.7,
            trendline=None,
        )
        st.plotly_chart(fig_gcs_npi, use_container_width=True)

        st.markdown("### NPi by Severity")
        fig_npi_sev = px.box(
            df_all,
            x="severity",
            y="npi",
            color="severity"
        )
        st.plotly_chart(fig_npi_sev, use_container_width=True)

# ============================================================
# 2. MODEL PERFORMANCE
# ============================================================
with tab2:
    st.subheader("Model Performance Metrics")

    # -----------------------------------------
    # Apply filters from Patient Explorer
    # -----------------------------------------
    df_model = df_all.copy()

    sf = st.session_state.get("site_filter", "All")
    dfi = st.session_state.get("diag_filter", "All")
    sev = st.session_state.get("severity_filter", "All")
    pid = st.session_state.get("patient_id_filter", "").strip()

    if sf != "All":
        df_model = df_model[df_model["site_id"] == sf]
    if dfi != "All":
        df_model = df_model[df_model["diagnosis"] == dfi]
    if sev != "All":
        df_model = df_model[df_model["severity"] == sev]
    if pid:
        try:
            df_model = df_model[df_model["patient_id"] == int(pid)]
        except:
            pass

    # -----------------------------------------
    # Train models on filtered data
    # -----------------------------------------
    _, model_results = train_all_models(df_model)
    models_dict = model_results["models"]

    # -----------------------------------------
    # Display metrics
    # -----------------------------------------
    metrics_rows = []
    for name, info in models_dict.items():
        m = info["metrics"]
        metrics_rows.append({
            "model": name,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
        })

    metrics_df = pd.DataFrame(metrics_rows)

    numeric_cols = metrics_df.select_dtypes(include=["float", "int"]).columns
    st.dataframe(
        metrics_df.style.format({col: "{:.3f}" for col in numeric_cols}),
        use_container_width=True
    )

    st.subheader("ROC Curves")
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, info in models_dict.items():
        fpr, tpr = info["roc"]
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    st.pyplot(fig)
# ============================================================
# 3. PATIENT EXPLORER
# ============================================================
with tab3:
    st.subheader("Patient-Level Exploration")

    col_filters = st.columns(4)

    with col_filters[0]:
        site_filter = st.selectbox(
            "Site",
            options=["All"] + sorted(df_all["site_id"].unique().tolist())
        )
    with col_filters[1]:
        diag_filter = st.selectbox(
            "Diagnosis",
            options=["All"] + sorted(df_all["diagnosis"].unique().tolist())
        )
    with col_filters[2]:
        severity_filter = st.selectbox(
            "Severity",
            options=["All"] + sorted(df_all["severity"].unique().tolist())
        )
    with col_filters[3]:
        patient_id_filter = st.text_input("Patient ID (optional)")

    # 🔥 NEW: Save filters globally
    st.session_state["site_filter"] = site_filter
    st.session_state["diag_filter"] = diag_filter
    st.session_state["severity_filter"] = severity_filter
    st.session_state["patient_id_filter"] = patient_id_filter

    df_filtered = df_all.copy()

    if site_filter != "All":
        df_filtered = df_filtered[df_filtered["site_id"] == site_filter]
    if diag_filter != "All":
        df_filtered = df_filtered[df_filtered["diagnosis"] == diag_filter]
    if severity_filter != "All":
        df_filtered = df_filtered[df_filtered["severity"] == severity_filter]
    if patient_id_filter.strip():
        try:
            pid = int(patient_id_filter)
            df_filtered = df_filtered[df_filtered["patient_id"] == pid]
        except ValueError:
            st.warning("Patient ID must be an integer.")

    st.markdown(f"**Filtered rows: {len(df_filtered)}**")
    st.dataframe(df_filtered, use_container_width=True, height=400)

    st.markdown("### NPi vs GCS (Filtered)")
    fig_filtered = px.scatter(
        df_filtered,
        x="npi",
        y="gcs",
        color="severity",
        hover_data=["patient_id", "diagnosis", "site_id"]
    )
    st.plotly_chart(fig_filtered, use_container_width=True)

# ============================================================
# 4. FDA NARRATIVE SUMMARY
# ============================================================
with tab4:
    st.subheader("LLM-Generated FDA-Style Narrative")

    st.markdown(
        "This section uses Anthropic Claude Sonnet to generate a structured, "
        "regulator-facing summary of dataset and model performance."
    )

    with st.expander("Dataset description used for the LLM prompt"):
        st.write(
            f"- Rows: {len(df_all)}\n"
            f"- Sites: {df_all['site_id'].nunique()}\n"
            f"- Diagnoses: {', '.join(sorted(df_all['diagnosis'].unique()))}\n"
            f"- Outcome: binary `gcs_severe` (GCS ≤ 8)\n"
            f"- Features: NPi, pupil sizes, velocities, latency, GCS, demographics"
        )

    if st.button("Generate FDA-Style Summary"):
        with st.spinner("Calling Claude Sonnet to generate narrative..."):

            # -----------------------------
            # Build dataset description
            # -----------------------------
            dataset_description = (
                f"Synthetic pupillometry dataset with {len(df_all)} rows, "
                f"{df_all['site_id'].nunique()} sites, and diagnoses including "
                f"{', '.join(sorted(df_all['diagnosis'].unique()))}. "
                "Primary endpoint is a binary indicator of severe neurological status "
                "(`gcs_severe`, defined as GCS ≤ 8). Features include age, sex, diagnosis, "
                "NPi, true and measured pupil size, left/right pupil size, "
                "constriction and dilation velocities, latency, and GCS."
            )

            # -----------------------------
            # IMPORT SPLIT FUNCTIONS
            # -----------------------------
            from src.fda_summary import (
                generate_fda_core_summary,
                generate_performance_interpretation
            )

            # -----------------------------
            # CALL CLAUDE — PART 1 (Sections 1–3 + 5)
            # -----------------------------
            core_text = generate_fda_core_summary(dataset_description)

            # -----------------------------
            # BUILD PERFORMANCE TABLE (manual, avoids token waste)
            # -----------------------------
            mLR = models_dict["Logistic Regression"]["metrics"]
            mRF = models_dict["Random Forest"]["metrics"]
            mXGB = models_dict["XGBoost"]["metrics"]

            table_md = f"""
4. Performance Summary

| Metric | Logistic Regression | Random Forest | XGBoost |
|--------|----------------------|---------------|----------|
| Accuracy | {mLR['accuracy']:.3f} | {mRF['accuracy']:.3f} | {mXGB['accuracy']:.3f} |
| Precision | {mLR['precision']:.3f} | {mRF['precision']:.3f} | {mXGB['precision']:.3f} |
| Sensitivity (Recall) | {mLR['recall']:.3f} | {mRF['recall']:.3f} | {mXGB['recall']:.3f} |
| Specificity | {mLR['specificity']:.3f} | {mRF['specificity']:.3f} | {mXGB['specificity']:.3f} |
| F1 Score | {mLR['f1']:.3f} | {mRF['f1']:.3f} | {mXGB['f1']:.3f} |
| ROC AUC | {mLR['roc_auc']:.3f} | {mRF['roc_auc']:.3f} | {mXGB['roc_auc']:.3f} |
"""

            # -----------------------------
            # CALL CLAUDE — PART 2 (Interpretation ONLY)
            # -----------------------------
            perf_text = generate_performance_interpretation(models_dict)

            # -----------------------------
            # COMBINE EVERYTHING
            # -----------------------------
            summary_text = core_text + "\n\n" + table_md + "\n\n" + perf_text

            st.markdown(summary_text)

    st.info(
        "Ensure ANTHROPIC_API_KEY is set in your environment and the Claude model "
        "in fda_summary.py matches your account."
    )