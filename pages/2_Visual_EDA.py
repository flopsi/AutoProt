"""
pages/2_EDA.py
Visual Exploratory Data Analysis
- 6 individual sample distributions (A1-A3, B1-B3)
- 2 ranked intensity plots (A group, B group)
- Intensity heatmap (1000 random proteins)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from helpers.constants import get_theme, TRANSFORMS
from helpers.transforms import apply_transform
from helpers.audit import log_event
# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Visual EDA", layout="wide")

# Check if data is loaded
if "protein_data" not in st.session_state:
    st.warning("⚠️ Please upload data first (Page 1: Data Upload)")
    st.stop()

# ============================================================================
# LOAD DATA
# ============================================================================

protein_data = st.session_state.protein_data
df = protein_data.raw.copy()
numeric_cols = protein_data.numeric_cols


# ============================================================================
# SIDEBAR: SETTINGS
# ============================================================================

with st.sidebar:
    st.title("📊 EDA Settings")
    
    # Transform selection
    st.subheader("Transform")
    transform_method = st.selectbox(
        "Select transformation",
        options=TRANSFORMS,
        index=0,  # Default: log2
        help="Transform data before visualization"
    )
    
    # Theme selection
    st.subheader("Theme")
    theme_name = st.selectbox(
        "Color scheme",
        options=["light", "dark", "colorblind", "journal"],
        index=0
    )
    
    st.session_state.theme = theme_name
    
    # Group assignment
    st.subheader("Sample Groups")
    st.info("Assign samples to groups A or B")
    
    sample_groups = {}
    for col in numeric_cols:
        group = st.radio(
            f"{col}",
            options=["A", "B"],
            key=f"group_{col}",
            horizontal=True,
            index=0 if "A" in col.upper() else 1
        )
        sample_groups[col] = group

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("📊 Visual Exploratory Data Analysis")

# Get theme
theme = get_theme(theme_name)

# ============================================================================
# DATA TRANSFORMATION
# ============================================================================

st.subheader("1️⃣ Data Transformation")

with st.spinner(f"Applying {transform_method} transformation..."):
    df_transformed = apply_transform(df, numeric_cols, method=transform_method)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Transform", transform_method)
with col2:
    group_a_samples = [c for c, g in sample_groups.items() if g == "A"]
    group_b_samples = [c for c, g in sample_groups.items() if g == "B"]
    st.metric("Group A", len(group_a_samples))
with col3:
    st.metric("Group B", len(group_b_samples))

# ============================================================================
# PLOT 1: 6 INDIVIDUAL DISTRIBUTIONS (3x2 GRID)
# ============================================================================

st.subheader("2️⃣ Individual Sample Distributions")

# Create 3x2 subplot grid
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=[f"{col}" for col in numeric_cols[:6]],
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

# Add each sample as subplot
for idx, col in enumerate(numeric_cols[:6]):
    row = (idx // 3) + 1
    col_pos = (idx % 3) + 1
    
    # Get values (filter out missing)
    values = df_transformed[col][df_transformed[col] > 1.0]
    
    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=values,
            name=col,
            nbinsx=50,
            opacity=0.7,
            marker_color=theme['color_human']
        ),
        row=row, col=col_pos
    )

# Update layout
fig.update_layout(
    title_text=f"{transform_method.upper()}-Transformed Intensity Distributions",
    showlegend=False,
    plot_bgcolor=theme['bg_primary'],
    paper_bgcolor=theme['paper_bg'],
    font=dict(family="Arial", size=12, color=theme['text_primary']),
    height=600
)

# Update axes
fig.update_xaxes(title_text=f"{transform_method} Intensity", showgrid=True, gridcolor=theme['grid'])
fig.update_yaxes(title_text="Count", showgrid=True, gridcolor=theme['grid'])

st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PLOT 2: RANKED INTENSITY PLOTS (GROUP A & B)
# ============================================================================

st.subheader("3️⃣ Ranked Intensity Plots")

st.markdown("""
Proteins ranked by median intensity. Shows dynamic range and data quality.
""")

# Function to calculate ranked intensities
def get_ranked_intensities(df, cols):
    """Calculate median intensity per protein and rank."""
    # Get median across samples
    median_intensities = df[cols].median(axis=1)
    
    # Filter valid proteins (at least one value > 1.0)
    valid_mask = (df[cols] > 1.0).any(axis=1)
    median_intensities = median_intensities[valid_mask]
    
    # Rank
    ranked = median_intensities.sort_values(ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1  # Start from 1
    
    return ranked

# Create side-by-side plots
col1, col2 = st.columns(2)

# Group A
with col1:
    st.markdown("**Group A**")
    
    ranked_a = get_ranked_intensities(df_transformed, group_a_samples)
    
    fig_a = go.Figure()
    fig_a.add_trace(go.Scatter(
        x=list(range(1, len(ranked_a) + 1)),
        y=ranked_a.values,
        mode='lines',
        line=dict(color=theme['color_human'], width=2),
        name='Group A'
    ))
    
    fig_a.update_layout(
        title="Group A: Ranked Protein Intensities",
        xaxis_title="Protein Rank",
        yaxis_title=f"{transform_method} Intensity",
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['paper_bg'],
        font=dict(family="Arial", size=12, color=theme['text_primary']),
        height=400,
        yaxis_type="log"  # Log scale for y-axis
    )
    
    fig_a.update_xaxes(showgrid=True, gridcolor=theme['grid'])
    fig_a.update_yaxes(showgrid=True, gridcolor=theme['grid'])
    
    st.plotly_chart(fig_a, use_container_width=True)
    st.metric("Total Proteins", f"{len(ranked_a):,}")

# Group B
with col2:
    st.markdown("**Group B**")
    
    ranked_b = get_ranked_intensities(df_transformed, group_b_samples)
    
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(
        x=list(range(1, len(ranked_b) + 1)),
        y=ranked_b.values,
        mode='lines',
        line=dict(color=theme['color_yeast'], width=2),
        name='Group B'
    ))
    
    fig_b.update_layout(
        title="Group B: Ranked Protein Intensities",
        xaxis_title="Protein Rank",
        yaxis_title=f"{transform_method} Intensity",
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['paper_bg'],
        font=dict(family="Arial", size=12, color=theme['text_primary']),
        height=400,
        yaxis_type="log"
    )
    
    fig_b.update_xaxes(showgrid=True, gridcolor=theme['grid'])
    fig_b.update_yaxes(showgrid=True, gridcolor=theme['grid'])
    
    st.plotly_chart(fig_b, use_container_width=True)
    st.metric("Total Proteins", f"{len(ranked_b):,}")

# ============================================================================
# PLOT 3: INTENSITY HEATMAP (1000 RANDOM PROTEINS)
# ============================================================================

st.subheader("4️⃣ Intensity Heatmap")

st.markdown("""
Random sample of 1000 proteins showing intensity patterns across all samples.
""")

# Sample 1000 random proteins
n_sample = min(1000, len(df_transformed))
sampled_df = df_transformed.sample(n=n_sample, random_state=42)

# Get intensity matrix (filter valid values)
heatmap_data = sampled_df[numeric_cols].copy()

# Replace missing values (1.0) with NaN for better visualization
heatmap_data = heatmap_data.replace(1.0, np.nan)

# Create heatmap
fig_heatmap = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=numeric_cols,
    y=[f"P{i+1}" for i in range(len(heatmap_data))],
    colorscale='Viridis',
    colorbar=dict(title=f"{transform_method}<br>Intensity"),
    hoverongaps=False
))

fig_heatmap.update_layout(
    title=f"Intensity Heatmap ({n_sample} Random Proteins)",
    xaxis_title="Samples",
    yaxis_title="Proteins",
    plot_bgcolor=theme['bg_primary'],
    paper_bgcolor=theme['paper_bg'],
    font=dict(family="Arial", size=12, color=theme['text_primary']),
    height=700,
    yaxis=dict(showticklabels=False)  # Hide protein labels (too many)
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

with st.expander("📊 Summary Statistics"):
    
    stats = []
    for col in numeric_cols:
        values = df_transformed[col][df_transformed[col] > 1.0]
        group = sample_groups.get(col, "Unknown")
        stats.append({
            'Sample': col,
            'Group': group,
            'Mean': values.mean(),
            'Median': values.median(),
            'Std Dev': values.std(),
            'Min': values.min(),
            'Max': values.max(),
            'N Valid': len(values)
        })
    
    stats_df = pd.DataFrame(stats)
    
    st.dataframe(
        stats_df.style.format({
            'Mean': '{:.2f}',
            'Median': '{:.2f}',
            'Std Dev': '{:.2f}',
            'Min': '{:.2f}',
            'Max': '{:.2f}',
            'N Valid': '{:,.0f}'
        }),
        use_container_width=True,
        height=300
    )

# ============================================================================
# INTERPRETATION GUIDE
# ============================================================================

with st.expander("💡 Interpretation Guide"):
    st.markdown("""
    **Distribution Plots:**
    - Similar shapes → Good batch consistency
    - Different peaks → Potential batch effects
    - Wide spread → High variance
    
    **Ranked Intensity Plots:**
    - Smooth curve → Good data quality
    - Wide dynamic range → Many orders of magnitude
    - Flat regions → Detection limit reached
    
    **Heatmap:**
    - Vertical patterns → Sample-specific effects
    - Horizontal patterns → Protein groups with similar behavior
    - Missing values (white) → Proteins not detected
    """)

# ============================================================================
# AUDIT LOGGING
# ============================================================================

log_event(
    "Visual EDA",
    f"Generated 3 plot types with {transform_method} transform",
    {
        "transform": transform_method,
        "theme": theme_name,
        "n_samples": len(numeric_cols),
        "n_proteins": len(df),
        "group_a_samples": len(group_a_samples),
        "group_b_samples": len(group_b_samples)
    }
)

# ============================================================================
# NAVIGATION
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⬅️ Back to Upload"):
        st.switch_page("pages/1_Data_Upload.py")

with col2:
    st.info("Current: Visual EDA")

with col3:
    if st.button("Next: Preprocessing ➡️"):
        st.switch_page("pages/3_Preprocessing.py")
