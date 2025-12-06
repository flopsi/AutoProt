"""
pages/2_Visual_EDA.py

Visual exploratory data analysis - uses helpers from viz.py
"""

import streamlit as st
from helpers.core import ProteinData, get_theme
from helpers.transforms import apply_transformation
from helpers.analysis import detect_conditions_from_columns, create_group_dict
from helpers.viz import create_protein_count_stacked_bar, create_boxplot_by_condition
from helpers.audit import log_event
import pandas as pd
# ============================================================================
# CONTROL FUNCTIONS
# ============================================================================

def restart_pipeline():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.switch_page("pages/1_Data_Upload.py")

def reset_eda():
    keys = ['eda_log2_data', 'eda_conditions', 'eda_logged']
    for k in keys:
        st.session_state.pop(k, None)
    st.cache_data.clear()
    st.rerun()

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="Visual EDA", layout="wide")

with st.sidebar:
    st.title("⚙️ Options")
    if st.button("🔄 Reset This Page", width="stretch"):
        reset_eda()
    st.markdown("---")
    if st.button("🏠 Restart Pipeline", width="stretch"):
        restart_pipeline()

# ============================================================================
# LOAD DATA
# ============================================================================

st.title("📊 Visual EDA")

if "protein_data" not in st.session_state:
    st.warning("⚠️ No data loaded")
    if st.button("← Go to Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

protein_data: ProteinData = st.session_state.protein_data
df_raw = protein_data.raw
numeric_cols = protein_data.numeric_cols
species_mapping = protein_data.species_mapping
theme = get_theme(st.session_state.get("theme", "dark"))

st.info(f"📁 **{protein_data.file_path}** | {protein_data.n_proteins:,} proteins × {protein_data.n_samples} samples")
st.markdown("---")

# ============================================================================
# LOG2 TRANSFORMATION (cached)
# ============================================================================

if 'eda_log2_data' not in st.session_state:
    with st.spinner("Applying log2 transformation..."):
        df_log2, _ = apply_transformation(df_raw, numeric_cols, method="log2")
        st.session_state.eda_log2_data = df_log2
else:
    df_log2 = st.session_state.eda_log2_data

# ============================================================================
# DETECT CONDITIONS (cached)
# ============================================================================

if 'eda_conditions' not in st.session_state:
    conditions = detect_conditions_from_columns(numeric_cols)
    condition_dict = create_group_dict(numeric_cols, conditions)
    st.session_state.eda_conditions = condition_dict
else:
    condition_dict = st.session_state.eda_conditions

# ============================================================================
# PLOT 1: PROTEIN COUNTS (uses viz helper)
# ============================================================================

# ============================================================================
# SPECIES BREAKDOWN BY SAMPLE
# ============================================================================

if protein_data.species_mapping:
    st.subheader("📊 Species Breakdown by Sample")
    
    # Build counts: for each sample, count proteins per species
    import plotly.express as px
    from helpers.core import get_theme
    
    chart_data = []
    for sample_col in numeric_cols:
        valid_mask = df_raw[sample_col] > 1.0
        for idx in df_raw.loc[valid_mask].index:
            species = df_raw.loc[idx, protein_data.species_col]
            chart_data.append({"Sample": sample_col, "Species": species, "Count": 1})

    
    if chart_data:
        chart_df = pd.DataFrame(chart_data)
        species_counts = chart_df.groupby(["Sample", "Species"])["Count"].sum().reset_index()
        
        theme = get_theme(st.session_state.get("theme", "dark"))
        color_map = {
            "HUMAN": theme["color_human"],
            "YEAST": theme["color_yeast"],
            "ECOLI": theme["color_ecoli"],
        }
        for sp in species_counts["Species"].unique():
            if sp not in color_map:
                color_map[sp] = theme["accent"]
        
        fig = px.bar(
            species_counts,
            x="Sample",
            y="Count",
            color="Species",
            barmode="stack",
            color_discrete_map=color_map,
            height=400,
        )
        fig.update_layout(
            plot_bgcolor=theme["bg_primary"],
            paper_bgcolor=theme["paper_bg"],
            font=dict(size=11, color=theme["text_primary"]),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No valid intensity data")


# ============================================================================
# LOG EVENT
# ============================================================================

if 'eda_logged' not in st.session_state:
    log_event("Visual EDA", "Page viewed", {"n_proteins": protein_data.n_proteins, "n_samples": protein_data.n_samples})
    st.session_state.eda_logged = True

# ============================================================================
# NAVIGATION
# ============================================================================

st.markdown("---")
c1, c2, c3, c4 = st.columns(4)

with c1:
    if st.button("← Upload", width="stretch"):
        st.switch_page("pages/1_Data_Upload.py")
with c2:
    if st.button("Statistical EDA →", width="stretch", type="primary"):
        st.switch_page("pages/3_Statistical_EDA.py")
with c3:
    if st.button("🔄 Reset", width="stretch"):
        reset_eda()
with c4:
    if st.button("🏠 Restart", width="stretch"):
        restart_pipeline()

st.caption("✅ All computations cached | Fast reruns")
