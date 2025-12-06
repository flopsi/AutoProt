"""
pages/2_Visual_EDA.py

Visual exploratory data analysis with transformation selection
Includes normality testing, distribution plots, and PCA
"""

import streamlit as st
import pandas as pd
import numpy as np
from helpers.transforms import (
    apply_transformation, TRANSFORM_NAMES, TRANSFORM_DESCRIPTIONS,
    list_available_transforms
)
from helpers.stats import test_normality_all_samples
from helpers.viz import (
    create_density_histograms, create_raw_row_figure,
    create_transformed_row_figure, create_pca_plot, create_heatmap_clustered
)
from helpers.core import get_theme, TransformCache
from helpers.ui import show_data_summary, download_button_csv
from helpers.audit import log_transformation_selected, log_species_filter
from helpers.analysis import filter_by_species, count_proteins_by_species

# ============================================================================
# PAGE CONFIGURATION
# Check if data is loaded
# ============================================================================

st.title("📊 Visual EDA & Transformation")

# Check for data in session state
if "protein_data" not in st.session_state or st.session_state.protein_data is None:
    st.warning("⚠️ No data loaded. Please upload data first.")
    if st.button("← Go to Data Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

protein_data = st.session_state.protein_data
df_raw = protein_data.raw
numeric_cols = protein_data.numeric_cols
theme_name = st.session_state.get("theme", "light")
theme = get_theme(theme_name)

st.markdown(f"""
Loaded: **{protein_data.file_path}** | 
{protein_data.n_proteins} proteins × {protein_data.n_samples} samples | 
{protein_data.missing_rate:.1f}% missing
""")

# ============================================================================
# SECTION: SPECIES FILTERING (OPTIONAL)
# Filter by species if species column detected
# ============================================================================

if protein_data.species_col is not None:
    st.header("1️⃣ Species Filtering (Optional)")
    
    # Count proteins per species
    species_counts = count_proteins_by_species(df_raw, protein_data.species_mapping)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        available_species = list(species_counts.index)
        selected_species = st.multiselect(
            "Select species to include:",
            options=available_species,
            default=available_species,
            help="Filter dataset to specific species"
        )
    
    with col2:
        st.dataframe(species_counts, use_container_width=True)
    
    # Apply filter
    if set(selected_species) != set(available_species):
        df_filtered = filter_by_species(df_raw, protein_data.species_mapping, selected_species)
        st.info(f"Filtered: {len(df_filtered)} proteins ({len(df_raw) - len(df_filtered)} removed)")
        
        # Log filter event
        log_species_filter(selected_species, len(df_filtered))
    else:
        df_filtered = df_raw.copy()
    
    st.markdown("---")
else:
    df_filtered = df_raw.copy()

# ============================================================================
# SECTION: TRANSFORMATION SELECTION
# Choose transformation method with description
# ============================================================================

st.header("2️⃣ Data Transformation")

col1, col2 = st.columns([2, 1])

with col1:
    # Transformation selector
    transform_options = list_available_transforms()
    selected_transform = st.selectbox(
        "Select transformation method:",
        options=transform_options,
        format_func=lambda x: TRANSFORM_NAMES.get(x, x),
        index=transform_options.index("log2"),  # Default to log2
        help="Choose mathematical transformation to stabilize variance and normalize data"
    )
    
    # Show description
    st.info(f"ℹ️ {TRANSFORM_DESCRIPTIONS.get(selected_transform, 'No description')}")

with col2:
    # Show transformation formula
    formulas = {
        "log2": r"$y = \log_2(x)$",
        "log10": r"$y = \log_{10}(x)$",
        "ln": r"$y = \ln(x)$",
        "sqrt": r"$y = \sqrt{x}$",
        "arcsinh": r"$y = \sinh^{-1}(x)$",
        "vst": r"$y = \sinh^{-1}(x / 2\mu)$",
    }
    if selected_transform in formulas:
        st.latex(formulas[selected_transform])

# --- Apply transformation with caching ---
with st.spinner("🔄 Applying transformation..."):
    df_transformed, trans_cols = apply_transformation(
        df_filtered,
        numeric_cols,
        method=selected_transform
    )

# For "raw", use original columns
if selected_transform == "raw":
    trans_cols = numeric_cols

st.success(f"✅ Transformation applied: **{TRANSFORM_NAMES[selected_transform]}**")

# Store in session state
st.session_state.df_transformed = df_transformed
st.session_state.trans_cols = trans_cols
st.session_state.current_transform = selected_transform

# ============================================================================
# SECTION: NORMALITY ASSESSMENT
# Shapiro-Wilk tests before and after transformation
# ============================================================================

st.header("3️⃣ Normality Assessment")

with st.spinner("📊 Running Shapiro-Wilk tests..."):
    normality_df = test_normality_all_samples(
        df_filtered,
        df_transformed,
        numeric_cols,
        alpha=0.05
    )

# Summary metrics
col1, col2, col3 = st.columns(3)

with col1:
    n_normal_raw = normality_df['Raw_Normal'].sum()
    st.metric(
        "Normal (Raw)",
        f"{n_normal_raw}/{len(normality_df)}",
        delta=f"{n_normal_raw/len(normality_df)*100:.0f}%"
    )

with col2:
    n_normal_trans = normality_df['Trans_Normal'].sum()
    st.metric(
        "Normal (Transformed)",
        f"{n_normal_trans}/{len(normality_df)}",
        delta=f"+{n_normal_trans - n_normal_raw}" if n_normal_trans > n_normal_raw else f"{n_normal_trans - n_normal_raw}"
    )

with col3:
    improved = (normality_df['Improvement'] == "✅ Yes").sum()
    st.metric(
        "Samples Improved",
        improved,
        delta=f"{improved/len(normality_df)*100:.0f}%"
    )

# Detailed table
with st.expander("📋 View Detailed Results"):
    st.dataframe(
        normality_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Raw_P_Value": st.column_config.NumberColumn(format="%.4f"),
            "Trans_P_Value": st.column_config.NumberColumn(format="%.4f"),
        }
    )

# Log transformation
if 'Trans_P_Value' in normality_df.columns:
    mean_p_trans = normality_df['Trans_P_Value'].mean()
    log_transformation_selected(selected_transform, mean_p_trans)

# ============================================================================
# SECTION: DIAGNOSTIC PLOTS
# Before/after comparison: distributions, Q-Q plots, mean-variance
# ============================================================================

st.header("4️⃣ Diagnostic Plots")

tab1, tab2 = st.tabs(["🔴 Before Transformation", "🟢 After Transformation"])

with tab1:
    with st.spinner("Generating raw data diagnostics..."):
        fig_raw = create_raw_row_figure(
            df_filtered,
            numeric_cols,
            title="Raw Data Diagnostics"
        )
        st.plotly_chart(fig_raw, use_container_width=True)

with tab2:
    with st.spinner("Generating transformed data diagnostics..."):
        fig_trans = create_transformed_row_figure(
            df_transformed,
            trans_cols,
            title=TRANSFORM_NAMES[selected_transform]
        )
        st.plotly_chart(fig_trans, use_container_width=True)

# ============================================================================
# SECTION: SAMPLE DISTRIBUTIONS
# Density histograms for each sample
# ============================================================================

st.header("5️⃣ Sample Distributions")

with st.spinner("Creating density plots..."):
    fig_density = create_density_histograms(
        df_transformed,
        trans_cols,
        TRANSFORM_NAMES[selected_transform],
        theme,
        max_plots=6
    )
    st.plotly_chart(fig_density, use_container_width=True)

# ============================================================================
# SECTION: MULTIVARIATE ANALYSIS
# PCA and hierarchical clustering
# ============================================================================

st.header("6️⃣ Multivariate Analysis")

tab1, tab2 = st.tabs(["🎯 PCA", "🔥 Heatmap"])

with tab1:
    st.subheader("Principal Component Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        pca_dim = st.radio("Dimensions:", [2, 3], index=0)
    
    with col1:
        with st.spinner("Computing PCA..."):
            # Create group mapping for coloring (if species available)
            group_mapping = None
            if protein_data.species_col:
                group_mapping = {
                    col: protein_data.species_mapping.get(protein_data.raw.index[0], "UNKNOWN")
                    for col in numeric_cols
                }
            
            fig_pca = create_pca_plot(
                df_transformed,
                trans_cols,
                group_mapping=group_mapping,
                theme_name=theme_name,
                dim=pca_dim
            )
            st.plotly_chart(fig_pca, use_container_width=True)

with tab2:
    st.subheader("Hierarchical Clustered Heatmap")
    
    # Limit to top N proteins for performance
    n_proteins_heatmap = st.slider(
        "Number of proteins to display:",
        min_value=50,
        max_value=min(500, len(df_transformed)),
        value=min(200, len(df_transformed)),
        step=50,
        help="Limiting protein count improves rendering performance"
    )
    
    with st.spinner("Generating heatmap..."):
        df_heatmap = df_transformed.head(n_proteins_heatmap)
        fig_heatmap = create_heatmap_clustered(
            df_heatmap,
            trans_cols,
            theme_name=theme_name
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

# ============================================================================
# SECTION: EXPORT OPTIONS
# Download transformed data
# ============================================================================

st.header("7️⃣ Export Transformed Data")

col1, col2 = st.columns(2)

with col1:
    download_button_csv(
        df_transformed[trans_cols],
        filename=f"transformed_{selected_transform}.csv",
        label="📥 Download Transformed Data (CSV)"
    )

with col2:
    # Export normality test results
    download_button_csv(
        normality_df,
        filename="normality_tests.csv",
        label="📥 Download Normality Tests (CSV)"
    )

# ============================================================================
# FOOTER
# Navigation hints
# ============================================================================

st.markdown("---")
st.caption("**Next Step:** Proceed to **🧪 3_Statistical_EDA** for differential expression analysis.")
