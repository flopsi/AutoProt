"""
pages/2_Visual_EDA.py

Visual exploratory data analysis with transformation selection
Modular design using existing helper functions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from helpers.core import get_theme, ProteinData
from helpers.audit import log_event
from helpers.analysis import detect_conditions_from_columns, group_columns_by_condition
from helpers.transforms import apply_transformation

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Visual EDA", layout="wide")

# ============================================================================
# CACHE MANAGEMENT & RESTART FUNCTIONS
# ============================================================================

def restart_entire_pipeline():
    """Clear all caches and return to upload page."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.switch_page("pages/1_Data_Upload.py")

def reset_eda_page():
    """Clear only EDA-specific cache, keep uploaded data."""
    keys_to_remove = [
        'eda_condition_mapping',
        'eda_log2_data',
        'eda_filtered_data',
        'eda_normalized_data',
        'eda_imputed_data',
        'eda_transform_method',
    ]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ============================================================================
# SIDEBAR: RESTART OPTIONS
# ============================================================================

with st.sidebar:
    st.title("⚙️ Options")
    st.markdown("### 🔄 Reset")
    
    if st.button("🔄 Reset This Page", use_container_width=True):
        reset_eda_page()
    
    st.markdown("---")
    
    if st.button("🏠 Restart Pipeline", use_container_width=True, type="secondary"):
        restart_entire_pipeline()

# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================

st.title("📊 Visual EDA & Transformation")

if "protein_data" not in st.session_state or st.session_state.protein_data is None:
    st.warning("⚠️ No data loaded. Please upload data first.")
    if st.button("← Go to Data Upload", type="primary"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

protein_data: ProteinData = st.session_state.protein_data
df_raw = protein_data.raw.copy()
numeric_cols = protein_data.numeric_cols
theme_name = st.session_state.get("theme", "dark")
theme = get_theme(theme_name)

st.info(f"""
📁 **{protein_data.file_path}** | 
🧬 {len(df_raw):,} proteins × {len(numeric_cols)} samples | 
❌ {protein_data.missing_rate:.1f}% missing
""")

st.markdown("---")

# ============================================================================
# MODULE 1: CONDITION ANNOTATION
# ============================================================================

st.header("1️⃣ Condition Annotation")

# Use helper to detect conditions
conditions = detect_conditions_from_columns(numeric_cols)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**Auto-Detected Conditions:**")
    
    if conditions:
        for condition in conditions:
            samples = group_columns_by_condition(numeric_cols, condition)
            st.markdown(f"- **Condition {condition}:** {', '.join(samples)}")
        
        use_auto = st.checkbox("Use auto-detected conditions", value=True)
        
        if use_auto:
            condition_mapping = {c: group_columns_by_condition(numeric_cols, c) for c in conditions}
        else:
            st.info("Manual condition definition not yet implemented")
            condition_mapping = {c: group_columns_by_condition(numeric_cols, c) for c in conditions}
    else:
        st.warning("⚠️ Could not auto-detect conditions.")
        condition_mapping = {}

with col2:
    if condition_mapping:
        st.metric("Total Conditions", len(condition_mapping))
        st.metric("Samples per Condition", 
                  f"{min(len(v) for v in condition_mapping.values())}-{max(len(v) for v in condition_mapping.values())}")

st.session_state.eda_condition_mapping = condition_mapping

st.markdown("---")

# ============================================================================
# MODULE 2: LOG2 TRANSFORMATION & BOX PLOT
# ============================================================================

st.header("2️⃣ Log2 Transformation & Sample Distributions")

# Use existing transform helper
with st.spinner("🔄 Applying log2 transformation..."):
    df_log2, trans_cols = apply_transformation(df_raw, numeric_cols, method="log2")

st.success("✅ Log2 transformation applied")
st.session_state.eda_log2_data = df_log2

# --- BOX PLOT ---
st.subheader("📊 Sample Distribution by Condition")

if condition_mapping:
    fig = go.Figure()
    
    colors = {
        'A': theme['color_human'],
        'B': theme['color_yeast'],
        'C': theme['color_ecoli'],
        'D': '#9467bd',
        'E': '#8c564b',
        'F': '#e377c2',
    }
    
    for condition, samples in sorted(condition_mapping.items()):
        for sample in samples:
            if sample in trans_cols:
                fig.add_trace(go.Box(
                    y=df_log2[sample].dropna(),
                    name=sample,
                    marker_color=colors.get(condition, theme['primary']),
                    legendgroup=condition,
                    legendgrouptitle_text=f"Condition {condition}",
                    boxmean='sd'
                ))
    
    fig.update_layout(
        title="Log2 Intensity Distribution by Sample and Condition",
        xaxis_title="Sample",
        yaxis_title="Log2 Intensity",
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['paper_bg'],
        font=dict(family="Arial", size=14, color=theme['text_primary']),
        showlegend=True,
        height=500,
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=theme['grid'], tickangle=-45)
    fig.update_yaxes(showgrid=True, gridcolor=theme['grid'])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary stats
    st.markdown("**Summary Statistics by Condition:**")
    summary_rows = []
    
    for condition, samples in sorted(condition_mapping.items()):
        all_values = np.concatenate([df_log2[s].dropna().values for s in samples if s in trans_cols])
        
        summary_rows.append({
            'Condition': condition,
            'N Samples': len(samples),
            'Mean Log2': np.mean(all_values),
            'Median Log2': np.median(all_values),
            'Std Dev': np.std(all_values),
        })
    
    st.dataframe(
        pd.DataFrame(summary_rows).style.format({
            'Mean Log2': '{:.2f}',
            'Median Log2': '{:.2f}',
            'Std Dev': '{:.2f}',
        }),
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("⚠️ No conditions defined.")

log_event(
    page="Visual EDA",
    action="Log2 transformation applied",
    details={"n_proteins": len(df_log2), "n_samples": len(numeric_cols)}
)

# ============================================================================
# PLACEHOLDER FOR FUTURE MODULES
# ============================================================================

st.markdown("---")
st.info("🚧 **More modules coming:** Normality testing, PCA, Correlation heatmap")

# ============================================================================
# NAVIGATION BUTTONS
# ============================================================================

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("← Previous: Upload", use_container_width=True):
        st.switch_page("pages/1_Data_Upload.py")

with col2:
    if st.button("Next: Statistical EDA →", use_container_width=True, type="primary"):
        if 'eda_log2_data' in st.session_state:
            st.switch_page("pages/3_Statistical_EDA.py")
        else:
            st.error("Complete transformation first")

with col3:
    if st.button("🔄 Reset Page", use_container_width=True):
        reset_eda_page()

with col4:
    if st.button("🏠 Restart", use_container_width=True):
        restart_entire_pipeline()

st.markdown("---")
st.caption("**Current:** Visual EDA → **Next:** Normality, PCA, Correlation")
