"""
pages/2_EDA.py
Visual Exploratory Data Analysis
Distribution plot with transform selection
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("📊 Visual Exploratory Data Analysis")

st.markdown("""
Visualize intensity distributions across samples to assess data quality 
and identify potential batch effects or outliers.
""")

# ============================================================================
# DATA TRANSFORMATION
# ============================================================================

st.subheader("1️⃣ Data Transformation")

col1, col2 = st.columns(2)

with col1:
    st.metric("Transform", transform_method)
    st.metric("Total Proteins", f"{len(df):,}")

with col2:
    st.metric("Samples", len(numeric_cols))
    st.metric("Missing %", f"{protein_data.missing_rate:.1f}%")

# Apply transformation
with st.spinner(f"Applying {transform_method} transformation..."):
    df_transformed = apply_transform(df, numeric_cols, method=transform_method)

st.success(f"✅ {transform_method} transformation applied")

# ============================================================================
# DISTRIBUTION PLOT
# ============================================================================

st.subheader("2️⃣ Intensity Distribution")

st.markdown(f"""
Distribution of **{transform_method}-transformed** intensities across all samples.
Each curve represents one sample.
""")

# Get theme
theme = get_theme(theme_name)

# Prepare data for plotting
fig = go.Figure()

# Add histogram for each sample
for i, col in enumerate(numeric_cols):
    # Filter out missing values (1.0 after imputation)
    values = df_transformed[col][df_transformed[col] > 1.0]
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=values,
        name=col,
        opacity=0.6,
        histnorm='probability density',
        nbinsx=50
    ))

# Update layout with theme
fig.update_layout(
    title=f"{transform_method.upper()}-Transformed Intensity Distribution",
    xaxis_title=f"{transform_method.upper()} Intensity",
    yaxis_title="Density",
    barmode='overlay',
    plot_bgcolor=theme['bg_primary'],
    paper_bgcolor=theme['paper_bg'],
    font=dict(
        family="Arial",
        size=14,
        color=theme['text_primary']
    ),
    height=500,
    showlegend=True,
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02
    )
)

fig.update_xaxes(
    showgrid=True,
    gridcolor=theme['grid'],
    gridwidth=1
)

fig.update_yaxes(
    showgrid=True,
    gridcolor=theme['grid'],
    gridwidth=1
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

st.subheader("3️⃣ Summary Statistics")

# Calculate stats on transformed data
stats = []
for col in numeric_cols:
    values = df_transformed[col][df_transformed[col] > 1.0]
    stats.append({
        'Sample': col,
        'Mean': values.mean(),
        'Median': values.median(),
        'Std Dev': values.std(),
        'Min': values.min(),
        'Max': values.max(),
        'N Valid': len(values)
    })

stats_df = pd.DataFrame(stats)

# Display as table
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
# INSIGHTS
# ============================================================================

with st.expander("💡 Interpretation Guide"):
    st.markdown("""
    **What to look for:**
    
    - **Similar distributions** → Good batch consistency
    - **Different peaks** → Potential batch effects
    - **Skewed distributions** → May need different transform
    - **Multiple peaks** → Possible contamination or subpopulations
    
    **Next steps:**
    - If distributions are similar → Proceed to preprocessing
    - If batch effects visible → Consider normalization
    - If outliers present → Check data quality (Page 4: Filtering)
    """)

# ============================================================================
# AUDIT LOGGING
# ============================================================================

log_event(
    "Visual EDA",
    f"Applied {transform_method} transform and viewed distribution",
    {
        "transform": transform_method,
        "theme": theme_name,
        "n_samples": len(numeric_cols),
        "n_proteins": len(df)
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
