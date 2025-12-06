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
from helpers.audit import log_transformation_selected, log_species_filter, log_event
from helpers.analysis import filter_by_species, count_proteins_by_species
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIGURATION
# Check if data is loaded
# ============================================================================

st.set_page_config(page_title="Visual EDA", layout="wide")

st.title("📊 Visual EDA & Transformation")

# Check for data in session state
if "protein_data" not in st.session_state or st.session_state.protein_data is None:
    st.warning("⚠️ No data loaded. Please upload data first.")
    if st.button("← Go to Data Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

# ============================================================================
# LOAD DATA FROM CACHE EFFICIENTLY
# ============================================================================

# Cache protein_data reference (avoid repeated access)
@st.cache_data
def get_protein_data_summary(file_path, n_proteins, n_samples, missing_rate):
    """Cache protein data summary for display."""
    return {
        "file_path": file_path,
        "n_proteins": n_proteins,
        "n_samples": n_samples,
        "missing_rate": missing_rate
    }

protein_data = st.session_state.protein_data
df_raw = protein_data.raw.copy()  # Work with copy to avoid mutations
numeric_cols = protein_data.numeric_cols
theme_name = st.session_state.get("theme", "light")
theme = get_theme(theme_name)

# Display data summary
summary = get_protein_data_summary(
    protein_data.file_path,
    protein_data.n_proteins,
    protein_data.n_samples,
    protein_data.missing_rate
)

st.markdown(f"""
Loaded: **{summary['file_path']}** | 
{summary['n_proteins']} proteins × {summary['n_samples']} samples | 
{summary['missing_rate']:.1f}% missing
""")

# ============================================================================
# SIDEBAR: PAGE-SPECIFIC CONTROLS
# ============================================================================

with st.sidebar:
    st.header("🎛️ EDA Settings")
    
    # Reset this page only
    if st.button("🔄 Reset This Page", use_container_width=True, help="Clear all selections and restart Visual EDA"):
        # Clear page-specific session state
        keys_to_clear = [
            'df_transformed', 'trans_cols', 'current_transform',
            'selected_species_eda', 'df_filtered_eda'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        log_event(
            page="Visual EDA",
            action="Page Reset",
            details={"cleared_keys": keys_to_clear}
        )
        st.rerun()
    
    st.markdown("---")
    st.info("""
    **This page:**
    - Species filtering
    - Transformation selection
    - Normality assessment
    - Distribution visualization
    - PCA & clustering
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
        
        # Initialize from session state if available
        default_species = st.session_state.get('selected_species_eda', available_species)
        
        selected_species = st.multiselect(
            "Select species to include:",
            options=available_species,
            default=default_species,
            help="Filter dataset to specific species",
            key="species_selector_eda"
        )
        
        # Store selection
        st.session_state.selected_species_eda = selected_species
    
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
    
    # Cache filtered data
    st.session_state.df_filtered_eda = df_filtered
    
    st.markdown("---")
else:
    df_filtered = df_raw.copy()
    st.session_state.df_filtered_eda = df_filtered

# ============================================================================
# SECTION: TRANSFORMATION SELECTION
# Choose transformation method with description
# ============================================================================

st.header("2️⃣ Data Transformation")

col1, col2 = st.columns([2, 1])

with col1:
    # Transformation selector
    transform_options = list_available_transforms()
    
    # Get default from session state
    default_transform = st.session_state.get('current_transform', 'log2')
    default_idx = transform_options.index(default_transform) if default_transform in transform_options else transform_options.index("log2")
    
    selected_transform = st.selectbox(
        "Select transformation method:",
        options=transform_options,
        format_func=lambda x: TRANSFORM_NAMES.get(x, x),
        index=default_idx,
        help="Choose mathematical transformation to stabilize variance and normalize data",
        key="transform_selector"
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
# MODULE: BOXPLOT OF LOG2 INTENSITIES BY CONDITION
# ============================================================================

st.header("4️⃣ Distribution by Condition")

st.markdown("**Box plot of log2 intensities grouped by experimental condition**")

# Detect conditions from column names (assumes A1, A2, B1, B2 format)
@st.cache_data
def detect_conditions(cols):
    """Extract condition labels from column names."""
    conditions = {}
    for col in cols:
        if len(col) > 0 and col[0].isalpha():
            condition = col[0]  # First letter = condition
            if condition not in conditions:
                conditions[condition] = []
            conditions[condition].append(col)
    return conditions

conditions = detect_conditions(trans_cols)

if len(conditions) > 0:
    # Prepare data for boxplot
    boxplot_data = []
    for condition, cols in sorted(conditions.items()):
        for col in cols:
            values = df_transformed[col].dropna()
            for val in values:
                boxplot_data.append({
                    "Condition": condition,
                    "Sample": col,
                    "Log2 Intensity": val
                })
    
    if boxplot_data:
        df_boxplot = pd.DataFrame(boxplot_data)
        
        # Create figure
        fig = go.Figure()
        
        # Add box for each condition
        for condition in sorted(df_boxplot["Condition"].unique()):
            condition_data = df_boxplot[df_boxplot["Condition"] == condition]["Log2 Intensity"]
            
            fig.add_trace(go.Box(
                y=condition_data,
                name=condition,
                boxmean='sd',  # Show mean and std dev
                marker=dict(
                    color=theme.get(f"color_{condition.lower()}", theme["color_primary"])
                )
            ))
        
        fig.update_layout(
            title="Log2 Intensity Distribution by Condition",
            yaxis_title="Log2 Intensity",
            xaxis_title="Condition",
            plot_bgcolor=theme["bg_primary"],
            paper_bgcolor=theme["paper_bg"],
            font=dict(color=theme["text_primary"]),
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics by condition
        with st.expander("📊 View Condition Statistics"):
            stats_by_condition = df_boxplot.groupby("Condition")["Log2 Intensity"].agg([
                ('Count', 'count'),
                ('Mean', 'mean'),
                ('Median', 'median'),
                ('Std Dev', 'std'),
                ('Min', 'min'),
                ('Max', 'max')
            ]).round(3)
            
            st.dataframe(stats_by_condition, use_container_width=True)
else:
    st.info("ℹ️ No condition structure detected. Column names should follow format: A1, A2, B1, B2, etc.")

# ============================================================================
# SECTION: DIAGNOSTIC PLOTS
# Before/after comparison: distributions, Q-Q plots, mean-variance
# ============================================================================

st.header("5️⃣ Diagnostic Plots")

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

st.header("6️⃣ Sample Distributions")

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

st.header("7️⃣ Multivariate Analysis")

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

st.header("8️⃣ Export Transformed Data")

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
# NAVIGATION & RESET BUTTONS
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("⬅️ Back to Upload", use_container_width=True):
        st.switch_page("pages/1_Data_Upload.py")

with col2:
    if st.button("➡️ Continue to Statistical EDA", type="primary", use_container_width=True):
        st.switch_page("pages/3_Statistical_EDA.py")

with col3:
    if st.button("🔄 Restart Pipeline", use_container_width=True, help="Clear ALL data and restart from beginning"):
        # Clear entire session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        log_event(
            page="Visual EDA",
            action="Pipeline Restart",
            details={"cleared_all_state": True}
        )
        
        st.success("✅ Pipeline restarted! Redirecting...")
        st.switch_page("pages/1_Data_Upload.py")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("""
**Workflow Progress:** Upload ✅ → Visual EDA (current) → Statistical EDA → Results

💡 **Tip:** Use the sidebar reset button to restart this page with fresh settings.
""")
