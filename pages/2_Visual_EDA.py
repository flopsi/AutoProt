"""
pages/2_Visual_EDA.py

Visual exploratory data analysis with log2 transformation
Modular design using existing helper functions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from helpers.core import ProteinData, get_theme
from helpers.transforms import apply_transformation, TRANSFORM_NAMES
from helpers.analysis import detect_conditions_from_columns, create_group_dict
from helpers.audit import log_event

# ============================================================================
# CACHE & RESTART FUNCTIONS
# ============================================================================

def restart_pipeline():
    """Clear all session state and return to upload."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.switch_page("pages/1_Data_Upload.py")

def reset_eda():
    """Clear EDA cache only."""
    keys = ['eda_log2_data', 'eda_conditions', 'eda_filtered']
    for k in keys:
        st.session_state.pop(k, None)
    st.rerun()

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="Visual EDA", layout="wide")

with st.sidebar:
    st.title("⚙️ Options")
    if st.button("🔄 Reset This Page", use_container_width=True):
        reset_eda()
    st.markdown("---")
    if st.button("🏠 Restart Pipeline", use_container_width=True):
        restart_pipeline()

# ============================================================================
# LOAD DATA
# ============================================================================

st.title("📊 Visual EDA & Transformation")

if "protein_data" not in st.session_state:
    st.warning("⚠️ No data loaded")
    if st.button("← Go to Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

protein_data: ProteinData = st.session_state.protein_data
df_raw = protein_data.raw.copy()
numeric_cols = protein_data.numeric_cols
theme = get_theme(st.session_state.get("theme", "dark"))

st.info(f"📁 **{protein_data.file_path}** | {len(df_raw):,} proteins × {len(numeric_cols)} samples")

st.markdown("---")

# ============================================================================
# MODULE 1: CONDITION DETECTION
# ============================================================================

st.header("1️⃣ Condition Annotation")

# Use helper to detect conditions
conditions = detect_conditions_from_columns(numeric_cols)  # Helper!
condition_dict = create_group_dict(numeric_cols, conditions)  # Helper!

if conditions:
    st.markdown("**Auto-Detected:**")
    for cond, samples in condition_dict.items():
        st.markdown(f"- **{cond}:** {', '.join(samples)}")
    
    c1, c2 = st.columns(2)
    c1.metric("Conditions", len(conditions))
    c2.metric("Samples/Condition", f"{min(len(v) for v in condition_dict.values())}-{max(len(v) for v in condition_dict.values())}")
else:
    st.warning("Could not detect conditions")
    condition_dict = {}

st.session_state.eda_conditions = condition_dict

st.markdown("---")

# ============================================================================
# MODULE 2: LOG2 TRANSFORMATION & BOX PLOT
# ============================================================================

st.header("2️⃣ Log2 Transformation")

# Apply log2 using helper (cached!)
with st.spinner("Applying log2..."):
    df_log2, trans_cols = apply_transformation(df_raw, numeric_cols, method="log2")  # Helper!

st.success("✅ Log2 applied")
st.session_state.eda_log2_data = df_log2

# Box plot by condition
st.subheader("📊 Sample Distributions by Condition")

if condition_dict:
    fig = go.Figure()
    
    # Colors by condition
    colors = {
        'A': theme['color_human'],
        'B': theme['color_yeast'],
        'C': theme['color_ecoli'],
        'D': '#9467bd',
        'E': '#8c564b'
    }
    
    for cond, samples in sorted(condition_dict.items()):
        for sample in samples:
            if sample in trans_cols:
                fig.add_trace(go.Box(
                    y=df_log2[sample].dropna(),
                    name=sample,
                    marker_color=colors.get(cond, theme['primary']),
                    legendgroup=cond,
                    legendgrouptitle_text=f"Condition {cond}",
                    boxmean='sd'
                ))
    
    fig.update_layout(
        title="Log2 Intensity Distribution",
        xaxis_title="Sample",
        yaxis_title="Log2 Intensity",
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['paper_bg'],
        font=dict(family="Arial", size=14, color=theme['text_primary']),
        showlegend=True,
        height=500
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=theme['grid'], tickangle=-45)
    fig.update_yaxes(showgrid=True, gridcolor=theme['grid'])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
    st.markdown("**Summary by Condition:**")
    summary = []
    for cond, samples in sorted(condition_dict.items()):
        vals = np.concatenate([df_log2[s].dropna().values for s in samples if s in trans_cols])
        summary.append({
            'Condition': cond,
            'N Samples': len(samples),
            'Mean': np.mean(vals),
            'Median': np.median(vals),
            'Std Dev': np.std(vals)
        })
    
    st.dataframe(
        pd.DataFrame(summary).style.format({
            'Mean': '{:.2f}',
            'Median': '{:.2f}',
            'Std Dev': '{:.2f}'
        }),
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("Define conditions in Step 1")

# Log event
log_event("Visual EDA", "Log2 applied", {"n_proteins": len(df_log2), "n_samples": len(numeric_cols)})

# ============================================================================
# FUTURE MODULES PLACEHOLDER
# ============================================================================

st.markdown("---")
st.info("🚧 **Coming next:** Normality testing, PCA, Correlation heatmap, Missing data viz")

# ============================================================================
# NAVIGATION
# ============================================================================

st.markdown("---")

c1, c2, c3, c4 = st.columns(4)

with c1:
    if st.button("← Upload", use_container_width=True):
        st.switch_page("pages/1_Data_Upload.py")

with c2:
    if st.button("Statistical EDA →", use_container_width=True, type="primary"):
        if 'eda_log2_data' in st.session_state:
            st.switch_page("pages/3_Statistical_EDA.py")
        else:
            st.error("Complete transformation first")

with c3:
    if st.button("🔄 Reset", use_container_width=True):
        reset_eda()

with c4:
    if st.button("🏠 Restart", use_container_width=True):
        restart_pipeline()

st.caption("**Module 1:** Condition detection ✓ | **Module 2:** Log2 transformation ✓")
