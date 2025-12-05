"""
pages/2_EDA.py - Uses your existing helper functions
"""

import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from helpers.constants import get_theme, TRANSFORMS
from helpers.statistics import test_normality_shapiro
from helpers.plots_advanced import create_heatmap_simple
from helpers.audit import log_event
from helpers.constants import get_theme, TRANSFORMS
from helpers.transforms import apply_transform, get_transform_info, recommend_transform


st.set_page_config(page_title="Visual EDA", layout="wide")

if "protein_data" not in st.session_state:
    st.warning("⚠️ Upload data first")
    st.stop()

# ============================================================================
# LOAD DATA
# ============================================================================

protein_data = st.session_state.protein_data
df = protein_data.raw.copy()
numeric_cols = protein_data.numeric_cols

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("📊 EDA Settings")
    
    transform_method = st.selectbox("Transform", TRANSFORMS, index=0)
    theme_name = st.selectbox("Theme", ["light", "dark", "colorblind", "journal"])
    st.session_state.theme = theme_name
    
    # Species filter
    species_col = getattr(protein_data, 'species_col', None)
    if species_col and species_col in df.columns:
        available_species = sorted(df[species_col].dropna().unique().tolist())
        selected_species = st.multiselect("Species", available_species, default=available_species)
        if selected_species:
            df = df[df[species_col].isin(selected_species)]
            st.info(f"{len(df):,} proteins from {len(selected_species)} species")

# Get theme
theme = get_theme(theme_name)

# ============================================================================
# TRANSFORM DATA (using your helper)
# ============================================================================

st.title("📊 Visual EDA")

if transform_method == "raw":
    df_for_plots = protein_data.raw.copy()
else:
    df_for_plots = apply_transform(
        protein_data.raw,
        protein_data.numeric_cols,
        method=transform_method,
    )

# Auto-detect groups
group_a = [c for c in numeric_cols if 'A' in c.upper()] or numeric_cols[:len(numeric_cols)//2]
group_b = [c for c in numeric_cols if 'B' in c.upper()] or numeric_cols[len(numeric_cols)//2:]

st.metric("Transform", transform_method)




# ============================================================================
# PLOT 1: 6 INDIVIDUAL DISTRIBUTIONS (Custom - not in helpers)
# ============================================================================

st.subheader("Individual Sample Distributions")

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=numeric_cols[:6],
    vertical_spacing=0.15,
    horizontal_spacing=0.10
)

for idx, col in enumerate(numeric_cols[:6]):
    row = (idx // 3) + 1
    col_pos = (idx % 3) + 1
    
    values = df_for_plots[col][df_for_plots[col] > 1.0]
    
    fig.add_trace(
        go.Histogram(
            x=values,
            name=col,
            nbinsx=40,
            opacity=0.75,
            marker_color=theme['color_human']
        ),
        row=row, col=col_pos
    )

fig.update_layout(
    showlegend=False,
    plot_bgcolor=theme['bg_primary'],
    paper_bgcolor=theme['paper_bg'],
    font=dict(family="Arial", size=11, color=theme['text_primary']),
    height=650
)

fig.update_xaxes(showgrid=True, gridcolor=theme['grid'])
fig.update_yaxes(showgrid=True, gridcolor=theme['grid'])

st.plotly_chart(fig, width="stretch")

# ============================================================================
# PLOT 2 & 3: RANKED INTENSITY (Custom)
# ============================================================================

st.subheader("Ranked Intensity Plots")

def get_ranked(df, cols):
    median = df[cols].median(axis=1)
    valid = (df[cols] > 1.0).any(axis=1)
    return median[valid].sort_values(ascending=False).reset_index(drop=True)

col1, col2 = st.columns(2)

with col1:
    ranked_a = get_ranked(df_for_plots, group_a)
    fig_a = go.Figure(go.Scatter(x=list(range(1, len(ranked_a)+1)), y=ranked_a.values, 
                                  mode='lines', line=dict(color=theme['color_human'], width=2)))
    fig_a.update_layout(title="Group A", xaxis_title="Rank", yaxis_title=f"{transform_method} Intensity",
                        plot_bgcolor=theme['bg_primary'], height=450)
    st.plotly_chart(fig_a, width="stretch")

with col2:
    ranked_b = get_ranked(df_for_plots, group_b)
    fig_b = go.Figure(go.Scatter(x=list(range(1, len(ranked_b)+1)), y=ranked_b.values,
                                  mode='lines', line=dict(color=theme['color_yeast'], width=2)))
    fig_b.update_layout(title="Group B", xaxis_title="Rank", yaxis_title=f"{transform_method} Intensity",
                        plot_bgcolor=theme['bg_primary'], height=450)
    st.plotly_chart(fig_b, width="stretch")

# ============================================================================
# PLOT 4: HEATMAP (using your helper!)
# ============================================================================

st.subheader("Intensity Heatmap")

fig = create_heatmap_simple(
    df_for_plots,
    protein_data.numeric_cols,
    theme_name=theme_name,
)
st.plotly_chart(fig,width="stretch")


# ============================================================================
# NORMALITY TESTS (using your helper!)
# ============================================================================

st.subheader("🔬 Normality Tests")

results = []
for col in numeric_cols:
    if transform_method == "raw":
        values = protein_data.raw[col].dropna()
    else:
        values = df_for_plots[col].dropna()

    res = test_normality_shapiro(values)
    results.append({
        "Sample": col,
        "Statistic": res["statistic"],
        "p-value": res["p_value"],
        "is_normal": res["is_normal"],
        "n": res["n"],
    })
st.dataframe(pd.DataFrame(results), width="stretch")


def highlight_normal(val):
    # val is True/False or truthy/falsy
    if bool(val):
        return "background-color: #d1fae5;"  # light green
    return ""

st.dataframe(
    results_df.style.applymap(highlight_normal, subset=["is_normal"]),
    width="stretch",
    height=400,
)

# ============================================================================
# AUDIT
# ============================================================================

log_event("EDA", f"{transform_method} analysis", {"n_proteins": len(df), "theme": theme_name})
