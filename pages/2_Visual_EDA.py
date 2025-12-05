"""
pages/2_Visual_EDA.py
Visual exploratory data analysis with normality testing and density plots
"""

import streamlit as st
import pandas as pd
import numpy as np

from helpers.transforms import compute_all_transforms_cached, TRANSFORM_NAMES, TRANSFORM_DESCRIPTIONS
from helpers.normality import analyze_all_transformations, find_best_transformation
from helpers.plots import create_density_histograms
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

st.success(f"✅ Loaded: {len(protein_data.raw):,} proteins × {len(protein_data.numeric_cols)} samples")

# ============================================================================
# COMPUTE ALL TRANSFORMS (CACHED)
# ============================================================================

with st.spinner("Computing transformations..."):
    all_transforms = compute_all_transforms_cached(
        df=protein_data.raw,
        numeric_cols=protein_data.numeric_cols,
        _hash_key=protein_data.file_path
    )

st.info(f"💾 Cached {len(all_transforms)} transformations")

# ============================================================================
# NORMALITY ANALYSIS
# ============================================================================

st.subheader("1️⃣ Normality Analysis")
st.caption("Testing which transformation best normalizes intensity distributions")

# Analyze all transformations
stats_df = analyze_all_transformations(
    all_transforms=all_transforms,
    numeric_cols=protein_data.numeric_cols,
    _hash_key=protein_data.file_path
)

# Find best transformation
best_transform, best_W = find_best_transformation(stats_df)

# Format and display table
display_df = stats_df.copy()

# Highlight best transformation
def highlight_best(row):
    if row["Transformation"] == best_transform:
        return ["background-color: #B5BD00; color: white; font-weight: bold"] * len(row)
    return [""] * len(row)

styled_df = display_df.style.apply(highlight_best, axis=1).format({
    "Kurtosis": "{:.3f}",
    "Skewness": "{:.3f}",
    "Shapiro_W": "{:.4f}",
    "Shapiro_p": "{:.2e}",
    "Normal_Count": "{:.0f}",
    "Total_Samples": "{:.0f}"
})

st.dataframe(styled_df, width='stretch', hide_index=True)

# Show recommendation
st.success(f"✅ **Recommended: {TRANSFORM_NAMES.get(best_transform, best_transform)}** (Shapiro W = {best_W:.4f})")

st.markdown("""
**Interpretation:**
- **Shapiro W**: Higher is better (closer to 1.0 = more normal)
- **Shapiro p**: > 0.05 indicates normal distribution
- **Kurtosis**: Closer to 0 is better (measures tail heaviness)
- **Skewness**: Closer to 0 is better (measures asymmetry)
- **Normal Count**: Number of samples passing normality test (p > 0.05)
""")

st.divider()

# ============================================================================
# VISUALIZATION CONTROLS
# ============================================================================

st.subheader("2️⃣ Distribution Visualization")

col1, col2 = st.columns(2)

with col1:
    # Transform selector
    transform_options = list(all_transforms.keys())
    selected_transform = st.selectbox(
        "Select Transformation",
        options=transform_options,
        format_func=lambda x: TRANSFORM_NAMES.get(x, x),
        index=transform_options.index(best_transform) if best_transform in transform_options else 0,
        help="Choose transformation to visualize"
    )
    
    st.caption(TRANSFORM_DESCRIPTIONS.get(selected_transform, ""))

with col2:
    # Species filter
    if protein_data.species_mapping:
        species_available = sorted(list(set(protein_data.species_mapping.values())))
        species_available = [s for s in species_available if s and s != 'Unknown']
        
        if species_available:
            selected_species = st.multiselect(
                "Filter by Species",
                options=species_available,
                default=species_available,
                help="Select species to include"
            )
        else:
            selected_species = None
            st.info("No species information available")
    else:
        selected_species = None
        st.info("No species information available")

# ============================================================================
# PREPARE DATA FOR PLOTTING
# ============================================================================

df_to_plot = all_transforms[selected_transform].copy()

# Apply species filter
if selected_species and protein_data.species_mapping:
    # Get protein IDs for selected species
    protein_ids_to_keep = [
        pid for pid, species in protein_data.species_mapping.items()
        if species in selected_species
    ]
    
    # Filter dataframe by index
    df_to_plot = df_to_plot[df_to_plot.index.isin(protein_ids_to_keep)]
    
    species_str = ', '.join(selected_species)
    st.info(f"📊 Plotting **{len(df_to_plot):,} proteins** from: **{species_str}**")
else:
    st.info(f"📊 Plotting **{len(df_to_plot):,} proteins** (all species)")

# Check if we have data
if len(df_to_plot) == 0:
    st.error("❌ No data after filtering!")
    st.stop()

# ============================================================================
# DENSITY HISTOGRAMS
# ============================================================================

theme = get_theme(st.session_state.get("theme", "light"))

with st.spinner("Creating plots..."):
    fig = create_density_histograms(
        df=df_to_plot,
        numeric_cols=protein_data.numeric_cols,
        transform_name=TRANSFORM_NAMES.get(selected_transform, selected_transform),
        theme=theme,
        max_plots=6
    )

st.plotly_chart(fig, width='stretch')

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

st.subheader("3️⃣ Summary Statistics")

summary_data = []
for col in protein_data.numeric_cols[:6]:  # Match plot limit
    sample_data = df_to_plot[col][df_to_plot[col] > 1.0].dropna()
    
    if len(sample_data) > 0:
        summary_data.append({
            'Sample': col,
            'N': int(len(sample_data)),
            'Mean': float(sample_data.mean()),
            'Median': float(sample_data.median()),
            'Std_Dev': float(sample_data.std()),
            'Min': float(sample_data.min()),
            'Max': float(sample_data.max())
        })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    
    # Format numeric columns
    for col in ['Mean', 'Median', 'Std_Dev', 'Min', 'Max']:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(lambda x: f"{x:.2f}")
    
    summary_df = summary_df.rename(columns={'Std_Dev': 'Std Dev'})
    st.dataframe(summary_df, width='stretch')
else:
    st.warning("No data for summary statistics")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
### 💡 Tips

**Normality Testing:**
- **Shapiro-Wilk test**: Most reliable for sample sizes 20-5000
- **p > 0.05**: Data likely normally distributed
- **Higher W value**: Better fit to normal distribution

**Choosing Transformations:**
- **Log2**: Standard for proteomics fold-change analysis
- **Yeo-Johnson**: Best for data with zeros and negatives
- **VST**: Stabilizes variance across intensity range

**Visual Inspection:**
- **Symmetric bell curve**: Good normality
- **Red line (mean)** should be centered
- **Red curve (KDE)**: Smoothed density estimate
""")
