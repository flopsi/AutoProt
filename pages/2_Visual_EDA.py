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
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Visual EDA", layout="wide")

# ============================================================================
# LOAD DATA
# ============================================================================

st.title("📊 Visual Exploratory Data Analysis")

protein_data = st.session_state.get("protein_data")
if not protein_data:
    st.error("❌ No data loaded. Please upload data first on the Data Upload page.")
    st.stop()

st.success(f"✅ Loaded: {len(protein_data.raw)} proteins × {len(protein_data.numeric_cols)} samples")

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
            st.info("No species information available")
    else:
        selected_species = None
        st.info("No species information available")

st.divider()

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
st.dataframe(styled_df, use_container_width=True, height=300)

# Summary metrics
n_normal_raw = normality_df['Raw_Normal'].sum()
n_normal_trans = normality_df['Trans_Normal'].sum()
n_improved = len([x for x in normality_df['Improvement'] if '✅' in x])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples Tested", len(normality_df))
c2.metric("Normal (Raw)", f"{n_normal_raw}/{len(normality_df)}")
c3.metric("Normal (Transformed)", f"{n_normal_trans}/{len(normality_df)}")
c4.metric("Improved", f"{n_improved}", delta=f"+{n_improved - n_normal_raw}")

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

st.subheader("2️⃣ Distribution Analysis (All Samples)")
st.markdown("""
Each plot shows:
- **Histogram**: Distribution of intensity values
- **Red dashed line**: Mean value
- **Shaded region**: ±2 standard deviations (95% of data if normal)
""")

theme = get_theme(st.session_state.get("theme", "light"))

# Determine grid layout based on number of samples
n_samples = len(protein_data.numeric_cols)
if n_samples <= 3:
    n_rows, n_cols = 1, n_samples
elif n_samples <= 6:
    n_rows, n_cols = 2, 3
elif n_samples <= 9:
    n_rows, n_cols = 3, 3
else:
    n_rows, n_cols = (n_samples + 3) // 4, 4

# Create subplots
fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    subplot_titles=[f"<b>{col}</b>" for col in protein_data.numeric_cols],
    vertical_spacing=0.12 if n_rows > 1 else 0,
    horizontal_spacing=0.08
)

# Plot each sample
for idx, sample_col in enumerate(protein_data.numeric_cols):
    row = (idx // n_cols) + 1
    col_pos = (idx % n_cols) + 1
    
    # Get data for this sample
    sample_data = df_to_plot[sample_col].dropna().values
    
    if len(sample_data) == 0:
        continue
    
    # Calculate statistics
    mean_val = np.mean(sample_data)
    std_val = np.std(sample_data)
    lower_bound = mean_val - 2 * std_val
    upper_bound = mean_val + 2 * std_val
    
    # Create histogram
    fig.add_trace(
        go.Histogram(
            x=sample_data,
            nbinsx=50,
            opacity=0.7,
            name=sample_col,
            marker=dict(
                color=theme["primary"],
                line=dict(color=theme["text_primary"], width=0.5)
            ),
            showlegend=False,
            hovertemplate="Value: %{x:.2f}<br>Count: %{y}<extra></extra>"
        ),
        row=row, col=col_pos
    )
    
    # Add mean line
    fig.add_vline(
        x=mean_val,
        line=dict(dash="dash", color="red", width=2),
        row=row, col=col_pos,
        annotation_text=f"μ={mean_val:.2f}",
        annotation_position="top",
        annotation_font_size=10
    )
    
    # Add ±2 StdDev shaded region
    fig.add_vrect(
        x0=lower_bound,
        x1=upper_bound,
        fillcolor=theme["primary"],
        opacity=0.15,
        line_width=0,
        row=row, col=col_pos,
        annotation_text=f"±2σ",
        annotation_position="bottom right",
        annotation_font_size=9
    )
    
    # Update axes for this subplot
    fig.update_xaxes(
        title_text="Intensity",
        showgrid=True,
        gridcolor=theme["grid"],
        row=row, col=col_pos
    )
    fig.update_yaxes(
        title_text="Count",
        showgrid=True,
        gridcolor=theme["grid"],
        row=row, col=col_pos
    )

# Update overall layout
fig.update_layout(
    height=400 * n_rows,
    title_text=f"<b>Sample Distributions - {TRANSFORM_NAMES[selected_transform]}</b>",
    title_font_size=18,
    showlegend=False,
    plot_bgcolor=theme["bg_primary"],
    paper_bgcolor=theme["paper_bg"],
    font=dict(
        family="Arial",
        size=12,
        color=theme["text_primary"]
    ),
    hovermode="closest"
)

st.plotly_chart(fig, use_container_width=True)

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
            'N': len(sample_data),
            'Mean': sample_data.mean(),
            'Median': sample_data.median(),
            'Std Dev': sample_data.std(),
            'Min': sample_data.min(),
            'Max': sample_data.max(),
            'Q1': sample_data.quantile(0.25),
            'Q3': sample_data.quantile(0.75),
            'IQR': sample_data.quantile(0.75) - sample_data.quantile(0.25)
        })

summary_df = pd.DataFrame(summary_data)

# Format numeric columns
for col in ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3', 'IQR']:
    summary_df[col] = summary_df[col].apply(lambda x: f"{x:.2f}")

st.dataframe(summary_df, use_container_width=True)

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
