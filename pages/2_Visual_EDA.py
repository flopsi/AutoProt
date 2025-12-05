"""
pages/2_Visual_EDA.py
Visual exploratory data analysis with distribution plots and normality testing
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from helpers.transforms import compute_all_transforms_cached, TRANSFORM_NAMES, TRANSFORM_DESCRIPTIONS
from helpers.statistics import test_normality_all_samples
from helpers.constants import get_theme

# ============================================================================
# LOAD DATA
# ============================================================================

st.title("📊 Visual Exploratory Data Analysis")

protein_data = st.session_state.get("protein_data")
if not protein_data:
    st.error("❌ No data loaded. Please upload data first on the Data Upload page.")
    st.stop()

st.success(f"✅ Loaded: {len(protein_data.raw)} proteins × {len(protein_data.numeric_cols)} samples")

# Debug: Show what we have
with st.expander("🔍 Debug Info"):
    st.write("**Protein Data Object:**")
    st.write(f"- Numeric columns: {protein_data.numeric_cols}")
    st.write(f"- Species column: {protein_data.species_col}")
    st.write(f"- Index column: {protein_data.index_col}")
    st.write(f"- Raw data shape: {protein_data.raw.shape}")
    st.write(f"- Raw data columns: {list(protein_data.raw.columns)}")
    
    # Show first few rows
    st.write("**First 3 rows of raw data:**")
    st.dataframe(protein_data.raw.head(3))

# ============================================================================
# COMPUTE ALL TRANSFORMS (CACHED - RUNS ONCE)
# ============================================================================

with st.spinner("Computing transformations..."):
    all_transforms = compute_all_transforms_cached(
        df=protein_data.raw,
        numeric_cols=protein_data.numeric_cols,
        _hash_key=protein_data.file_path  # Cache key
    )

st.info(f"💾 Cached {len(all_transforms)} transformations for instant switching")

# Debug: Check transforms
with st.expander("🔍 Transform Debug"):
    for transform_name, transform_df in all_transforms.items():
        st.write(f"**{transform_name}**: shape {transform_df.shape}")
        st.write(f"Sample values from first numeric column:")
        if len(protein_data.numeric_cols) > 0:
            first_col = protein_data.numeric_cols[0]
            st.write(transform_df[first_col].head(5))

# ============================================================================
# USER CONTROLS
# ============================================================================

st.subheader("🎛️ Analysis Controls")

col1, col2 = st.columns(2)

with col1:
    # Transform selector
    transform_options = list(all_transforms.keys())
    selected_transform = st.selectbox(
        "📊 Select Transformation",
        options=transform_options,
        format_func=lambda x: TRANSFORM_NAMES.get(x, x),
        index=0,  # Default: raw
        help="Choose transformation method to apply to data"
    )
    
    # Show description
    st.caption(TRANSFORM_DESCRIPTIONS.get(selected_transform, ""))

with col2:
    # Species filter
    if protein_data.species_mapping:
        species_available = list(set(protein_data.species_mapping.values()))
        species_available = [s for s in species_available if s and s != 'Unknown']
        
        if species_available:
            selected_species = st.multiselect(
                "🧬 Filter by Species",
                options=sorted(species_available),
                default=sorted(species_available),  # All selected by default
                help="Select species to include in analysis"
            )
        else:
            selected_species = None
            st.warning("No species information available - showing all data")
    else:
        selected_species = None
        st.warning("No species information available - showing all data")

st.divider()

# ============================================================================
# PREPARE DATA FOR PLOTTING
# ============================================================================

# Get the transformed dataframe
df_to_plot = all_transforms[selected_transform].copy()

st.write(f"**Initial data shape:** {df_to_plot.shape}")
st.write(f"**Columns:** {list(df_to_plot.columns)}")

# Apply species filtering if needed
if selected_species and protein_data.species_mapping:
    # Get protein IDs for selected species
    protein_ids_to_keep = [
        protein_id 
        for protein_id, species in protein_data.species_mapping.items()
        if species in selected_species
    ]
    
    st.write(f"**Filtering to {len(protein_ids_to_keep)} proteins from species: {selected_species}**")
    
    # Filter the dataframe
    # Check if index_col is in the columns or if it's the index
    if protein_data.index_col in df_to_plot.columns:
        df_to_plot = df_to_plot[df_to_plot[protein_data.index_col].isin(protein_ids_to_keep)]
    else:
        # Assume index is the protein ID
        df_to_plot = df_to_plot[df_to_plot.index.isin(protein_ids_to_keep)]
    
    st.write(f"**After species filter:** {df_to_plot.shape}")
else:
    st.write("**No species filtering applied - using all data**")

# Check if we have data
if len(df_to_plot) == 0:
    st.error("❌ No data after filtering! Check your species selection.")
    st.stop()

# Show sample of data
with st.expander("📊 View Sample Data (After Filtering)"):
    st.dataframe(df_to_plot.head(10))
    
    # Show statistics per column
    st.write("**Column Statistics:**")
    for col in protein_data.numeric_cols:
        if col in df_to_plot.columns:
            vals = df_to_plot[col].dropna()
            st.write(f"- **{col}**: {len(vals)} values, range [{vals.min():.2f}, {vals.max():.2f}]")


# ============================================================================
# NORMALITY TEST TABLE
# ============================================================================

st.subheader("1️⃣ Normality Test Results")
st.markdown("""
**Shapiro-Wilk Test** (p > 0.05 indicates normally distributed data)
- Tests whether each sample follows a normal distribution
- Transformation goal: Improve normality for statistical tests
""")

# Compute normality tests
normality_df = test_normality_all_samples(
    df_raw=all_transforms['raw'],
    df_transformed=all_transforms[selected_transform],
    numeric_cols=protein_data.numeric_cols,
    alpha=0.05
)

# Format for display
display_df = normality_df.copy()
display_df['Raw_P_Value'] = display_df['Raw_P_Value'].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
display_df['Trans_P_Value'] = display_df['Trans_P_Value'].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
display_df['Raw_Statistic'] = display_df['Raw_Statistic'].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
display_df['Trans_Statistic'] = display_df['Trans_Statistic'].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")

# Color code the table
def highlight_normality(row):
    colors = []
    for col in row.index:
        if col == 'Raw_Normal':
            colors.append('background-color: #90EE90' if row[col] else 'background-color: #FFB6C6')
        elif col == 'Trans_Normal':
            colors.append('background-color: #90EE90' if row[col] else 'background-color: #FFB6C6')
        elif col == 'Improvement':
            if '✅' in str(row[col]):
                colors.append('background-color: #90EE90; font-weight: bold')
            elif '⚠️' in str(row[col]):
                colors.append('background-color: #FFB6C6')
            else:
                colors.append('')
        else:
            colors.append('')
    return colors

styled_df = display_df.style.apply(highlight_normality, axis=1)
st.dataframe(styled_df, width="stretch", height=300)

# Summary metrics
# Summary metrics - FIX st.columns error
n_normal_raw = normality_df['Raw_Normal'].sum()
n_normal_trans = normality_df['Trans_Normal'].sum()
n_improved = len([x for x in normality_df['Improvement'] if '✅' in str(x)])

# Only create columns if we have data
if len(normality_df) > 0:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Samples Tested", len(normality_df))
    with c2:
        st.metric("Normal (Raw)", f"{n_normal_raw}/{len(normality_df)}")
    with c3:
        st.metric("Normal (Transformed)", f"{n_normal_trans}/{len(normality_df)}")
    with c4:
        delta_val = n_improved - n_normal_raw
        st.metric("Improved", f"{n_improved}", delta=f"{delta_val:+d}")
else:
    st.warning("No data to analyze")


st.divider()

# ============================================================================
# FILTER DATA BY SPECIES
# ============================================================================

df_to_plot = all_transforms[selected_transform].copy()

if selected_species:
    # Create species series
    species_series = pd.Series(protein_data.species_mapping)
    # Filter to selected species
    mask = species_series.isin(selected_species)
    protein_ids = species_series[mask].index
    df_to_plot = df_to_plot[df_to_plot.index.isin(protein_ids)]
    
    n_proteins_filtered = len(df_to_plot)
    species_str = ', '.join(selected_species)
    st.info(f"📊 Plotting **{n_proteins_filtered:,} proteins** from species: **{species_str}**")
else:
    st.info(f"📊 Plotting **{len(df_to_plot):,} proteins** (all species)")

# ============================================================================
# 6-PANEL DISTRIBUTION PLOTS
# ============================================================================

st.subheader("2️⃣ Individual Sample Distributions")
st.markdown("""
Each plot shows:
- **Histogram**: Distribution of intensity values
- **Red dashed line**: Mean value
- **Shaded region**: ±2 standard deviations (95% of data if normal)
""")

theme = get_theme(st.session_state.get("theme", "light"))

# Determine how many samples to plot (max 6 for 2x3 grid)
n_samples = min(len(protein_data.numeric_cols), 6)
numeric_cols = protein_data.numeric_cols[:n_samples]

# Create subplot grid
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=[f"<b>{col}</b>" for col in numeric_cols],
    vertical_spacing=0.15,
    horizontal_spacing=0.10
)

# Plot each sample
for idx, col in enumerate(numeric_cols):
    row = (idx // 3) + 1
    col_pos = (idx % 3) + 1
    
    # Get filtered data for this sample (drop NaN and values == 1.0)
    values = df_to_plot[col][df_to_plot[col] > 1.0].dropna().values
    
    if len(values) == 0:
        # Add "No data" annotation if empty
        fig.add_annotation(
            text="No data",
            xref=f"x{idx+1}" if idx > 0 else "x",
            yref=f"y{idx+1}" if idx > 0 else "y",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=row,
            col=col_pos
        )
        continue
    
    # Calculate statistics
    mean_val = np.mean(values)
    std_val = np.std(values)
    lower_bound = mean_val - 2 * std_val
    upper_bound = mean_val + 2 * std_val
    
    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=values,
            name=col,
            nbinsx=40,
            opacity=0.75,
            marker_color=theme['primary'],
            showlegend=False,
            hovertemplate="Intensity: %{x:.2f}<br>Count: %{y}<extra></extra>"
        ),
        row=row, col=col_pos
    )
    
    # Add mean line (vertical line)
    # Get histogram to determine y-axis range
    hist_values, bin_edges = np.histogram(values, bins=40)
    y_max = max(hist_values) if len(hist_values) > 0 else 1
    
    fig.add_trace(
        go.Scatter(
            x=[mean_val, mean_val],
            y=[0, y_max * 1.1],
            mode='lines',
            line=dict(dash="dash", color="red", width=2),
            showlegend=False,
            hovertemplate=f"Mean: {mean_val:.2f}<extra></extra>",
            name="Mean"
        ),
        row=row, col=col_pos
    )
    
    # Add ±2σ shaded region as a shape
    # Calculate axis reference names
    xaxis_ref = f"x{idx+1}" if idx > 0 else "x"
    yaxis_ref = f"y{idx+1}" if idx > 0 else "y"
    
    fig.add_shape(
        type="rect",
        xref=xaxis_ref,
        yref=f"{yaxis_ref} domain",  # Use domain for y (0 to 1)
        x0=lower_bound,
        x1=upper_bound,
        y0=0,
        y1=1,
        fillcolor=theme['primary'],
        opacity=0.2,
        line_width=0,
        layer="below"
    )
    
    # Add annotation for mean
    fig.add_annotation(
        text=f"μ = {mean_val:.2f}",
        xref=xaxis_ref,
        yref=yaxis_ref,
        x=mean_val,
        y=y_max * 0.95,
        showarrow=False,
        font=dict(size=10, color="red", family="Arial"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        borderpad=2
    )
    
    # Add annotation for ±2σ
    fig.add_annotation(
        text=f"±2σ",
        xref=xaxis_ref,
        yref=yaxis_ref,
        x=upper_bound,
        y=y_max * 0.1,
        showarrow=False,
        font=dict(size=9, color=theme['primary'], family="Arial"),
        bgcolor="rgba(255, 255, 255, 0.7)",
        borderpad=2
    )

# Update layout
fig.update_layout(
    showlegend=False,
    plot_bgcolor=theme['bg_primary'],
    paper_bgcolor=theme['paper_bg'],
    font=dict(family="Arial", size=11, color=theme['text_primary']),
    height=650,
    title_text=f"<b>Sample Distributions - {TRANSFORM_NAMES[selected_transform]}</b>",
    title_font_size=16,
    hovermode="closest"
)

# Update all axes
fig.update_xaxes(
    title_text="Intensity",
    showgrid=True,
    gridcolor=theme['grid'],
    gridwidth=1
)
fig.update_yaxes(
    title_text="Count",
    showgrid=True,
    gridcolor=theme['grid'],
    gridwidth=1
)

st.plotly_chart(fig, width="stretch")

# Show statistics summary below plots
st.markdown("**Statistics Summary:**")
stats_cols = st.columns(n_samples)
for idx, col in enumerate(numeric_cols):
    values = df_to_plot[col][df_to_plot[col] > 1.0].dropna().values
    if len(values) > 0:
        with stats_cols[idx]:
            mean_val = np.mean(values)
            std_val = np.std(values)
            st.caption(f"**{col}**")
            st.caption(f"μ: {mean_val:.2f}")
            st.caption(f"σ: {std_val:.2f}")
            st.caption(f"n: {len(values):,}")

# ============================================================================
# SUMMARY STATISTICS TABLE
# ============================================================================

st.subheader("3️⃣ Summary Statistics")

summary_data = []
for col in protein_data.numeric_cols:
    sample_data = df_to_plot[col].dropna()
    
    if len(sample_data) > 0:
        summary_data.append({
            'Sample': col,
            'N': int(len(sample_data)),
            'Mean': float(sample_data.mean()),
            'Median': float(sample_data.median()),
            'Std_Dev': float(sample_data.std()),
            'Min': float(sample_data.min()),
            'Max': float(sample_data.max()),
            'Q1': float(sample_data.quantile(0.25)),
            'Q3': float(sample_data.quantile(0.75)),
            'IQR': float(sample_data.quantile(0.75) - sample_data.quantile(0.25))
        })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    
    # Format numeric columns
    format_cols = ['Mean', 'Median', 'Std_Dev', 'Min', 'Max', 'Q1', 'Q3', 'IQR']
    for col in format_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(lambda x: f"{x:.2f}")
    
    # Rename for display
    summary_df = summary_df.rename(columns={'Std_Dev': 'Std Dev'})
    
    st.dataframe(summary_df, width='stretch')
else:
    st.warning("No data available for summary statistics")


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
### 💡 Interpretation Tips

**Normality Tests:**
- p > 0.05: Data likely normally distributed (good for t-tests)
- p ≤ 0.05: Data likely not normal (consider non-parametric tests)

**Distributions:**
- **Symmetric + bell-shaped**: Good normality
- **Right-skewed**: Consider log transformation
- **Left-skewed**: Consider power transformation
- **Bimodal**: Possible batch effects or subpopulations

**±2σ Region:**
- Should contain ~95% of data if normally distributed
- Outliers beyond ±3σ may need investigation
""")

st.info("💾 **All transformations are cached** - changing plots/filters is instant!")
