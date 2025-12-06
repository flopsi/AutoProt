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

st.header("1️⃣ Protein Counts per Sample")
st.markdown("Number of quantified proteins in each sample, stacked by species.")

fig1, summary_df1 = create_protein_count_stacked_bar(df_log2, numeric_cols, species_mapping, theme)
st.plotly_chart(fig1, width="stretch")

st.markdown("**Summary by Species:**")
st.dataframe(summary_df1, width="stretch", hide_index=True)

st.markdown("---")

# ============================================================================
# PLOT 2: BOXPLOTS (uses viz helper)
# ============================================================================

st.header("2️⃣ Log2 Intensity Distribution by Condition")
st.markdown("Box plots showing log2-transformed intensities for conditions A and B.")

if condition_dict and len(condition_dict) >= 2:
    conditions_to_plot = sorted(condition_dict.keys())[:2]
    
    fig2, summary_df2 = create_boxplot_by_condition(df_log2, condition_dict, conditions_to_plot, theme)
    st.plotly_chart(fig2, width="stretch")
    
    st.markdown("**Summary Statistics by Condition:**")
    st.dataframe(summary_df2, width="stretch", hide_index=True)
else:
    st.warning("⚠️ Need at least 2 conditions for comparison")

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
