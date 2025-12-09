"""
pages/2_Visual_EDA.py - OPTIMIZED VISUAL EXPLORATORY DATA ANALYSIS
Production-ready EDA with plotnine, interactive controls, and comprehensive visualizations

Features:
- Distribution plots (histograms, density, box plots)
- Species-stratified analysis
- Transformation comparison (log2, yeo-johnson, box-cox)
- Normality assessment (Q-Q plots, Shapiro-Wilk test)
- PCA and t-SNE with species/condition coloring
- Heatmaps with hierarchical clustering
- Missing data visualization
- Export-ready publication-quality figures
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import boxcox
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotnine import *
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Visual EDA - AutoProt",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Visual Exploratory Data Analysis")
st.markdown("Visualize and understand your proteomics data distributions, transformations, and structure")

# ============================================================================
# DATA VALIDATION
# ============================================================================

if 'data_ready' not in st.session_state or not st.session_state.data_ready:
    st.warning("⚠️ No data loaded. Please upload data on the **📁 Data Upload** page first.")
    st.stop()

# Load from session state
df_raw = st.session_state.df_raw
df_polars = st.session_state.get('df_raw_polars', pl.from_pandas(df_raw))
numeric_cols = st.session_state.numeric_cols
id_col = st.session_state.id_col
species_col = st.session_state.species_col
data_type = st.session_state.data_type

st.success(f"✅ Loaded {data_type} data: {len(df_raw):,} rows × {len(numeric_cols)} samples")
st.info(f"ID: **{id_col}** | Species: **{species_col}** | Data Type: **{data_type.upper()}**")
st.markdown("---")

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

with st.sidebar:
    st.header("⚙️ Visualization Settings")
    
    # Plot type selection
    plot_section = st.radio(
        "Select visualization:",
        options=[
            "Distribution",
            "Box & Violin",
            "Transformations",
            "Q-Q Plots",
            "PCA",
            "t-SNE",
            "Heatmap",
            "Missing Data"
        ],
        key="plot_section"
    )
    
    st.divider()
    
    # Color scheme
    color_scheme = st.selectbox(
        "Color scheme:",
        options=["Viridis", "Plasma", "Inferno", "Turbo", "Set2", "Pastel1"],
        key="color_scheme"
    )
    
    # Figure size
    fig_width = st.slider(
        "Figure width:",
        min_value=8,
        max_value=16,
        value=12,
        step=1,
        key="fig_width"
    )
    
    fig_height = st.slider(
        "Figure height:",
        min_value=4,
        max_value=10,
        value=6,
        step=1,
        key="fig_height"
    )
    
    # Transparency
    alpha = st.slider(
        "Transparency:",
        min_value=0.3,
        max_value=1.0,
        value=0.7,
        step=0.1,
        key="alpha"
    )
    
    # Bins for histograms
    hist_bins = st.slider(
        "Histogram bins:",
        min_value=20,
        max_value=100,
        value=50,
        step=10,
        key="hist_bins"
    )

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Long format for plotnine
df_long = df_raw.melt(
    id_vars=[id_col, species_col],
    value_vars=numeric_cols,
    var_name='Sample',
    value_name='Intensity'
)

# Log2 transformation
df_long['Log2_Intensity'] = np.log2(df_long['Intensity'] + 1)  # +1 to avoid log(0)

# Create numeric version of intensity (drop NaN for numeric operations)
df_numeric = df_raw[numeric_cols].copy()
df_numeric = df_numeric.replace(1.0, np.nan)  # Clean placeholder values

# ============================================================================
# SECTION 1: DISTRIBUTION PLOTS
# ============================================================================

if plot_section == "Distribution":
    st.header("1️⃣ Intensity Distributions")
    
    col1, col2 = st.columns(2)
    
    # Histogram (all data)
    with col1:
        st.subheader("Histogram - All Samples")
        st.caption("Log2-transformed intensity distribution across all samples")
        
        plot_hist = (
            ggplot(df_long, aes(x='Log2_Intensity')) +
            geom_histogram(bins=hist_bins, fill='steelblue', alpha=alpha, color='black', size=0.2) +
            labs(
                title='Log2 Intensity Distribution',
                x='Log2(Intensity + 1)',
                y='Count'
            ) +
            theme_minimal() +
            theme(
                figure_size=(fig_width/2 - 0.5, fig_height),
                axis_text_x=element_text(angle=45, hjust=1)
            )
        )
        st.pyplot(plot_hist)
    
    # Density plot by species
    with col2:
        st.subheader("Density - By Species")
        st.caption("Overlay of intensity distributions by species")
        
        plot_dens = (
            ggplot(df_long.dropna(subset=['Log2_Intensity']), aes(x='Log2_Intensity', color='factor(' + species_col + ')')) +
            geom_density(size=1, alpha=0.3) +
            labs(
                title='Density by Species',
                x='Log2(Intensity + 1)',
                y='Density',
                color=species_col
            ) +
            theme_minimal() +
            theme(
                figure_size=(fig_width/2 - 0.5, fig_height),
                axis_text_x=element_text(angle=45, hjust=1)
            )
        )
        st.pyplot(plot_dens)
    
    # Statistics by sample
    st.subheader("Sample Statistics")
    stats_df = pd.DataFrame({
        'Sample': numeric_cols,
        'N': [df_raw[col].notna().sum() for col in numeric_cols],
        'Mean': [df_raw[col].mean() for col in numeric_cols],
        'Median': [df_raw[col].median() for col in numeric_cols],
        'Std': [df_raw[col].std() for col in numeric_cols],
        'Min': [df_raw[col].min() for col in numeric_cols],
        'Max': [df_raw[col].max() for col in numeric_cols],
    }).round(2)
    
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

# ============================================================================
# SECTION 2: BOX & VIOLIN PLOTS
# ============================================================================

elif plot_section == "Box & Violin":
    st.header("2️⃣ Box & Violin Plots by Sample")
    
    col1, col2 = st.columns(2)
    
    # Box plot
    with col1:
        st.subheader("Box Plot")
        st.caption("Intensity distribution per sample showing quartiles and outliers")
        
        plot_box = (
            ggplot(df_long, aes(x='reorder(Sample, Log2_Intensity, na.rm=True)', y='Log2_Intensity', fill='Sample')) +
            geom_boxplot(alpha=alpha) +
            coord_flip() +
            labs(
                title='Log2 Intensity by Sample',
                x='Sample',
                y='Log2(Intensity + 1)'
            ) +
            theme_minimal() +
            theme(
                figure_size=(fig_width/2 - 0.5, fig_height),
                legend_position='none'
            )
        )
        st.pyplot(plot_box)
    
    # Violin plot
    with col2:
        st.subheader("Violin Plot")
        st.caption("Probability density by sample")
        
        plot_violin = (
            ggplot(df_long, aes(x='reorder(Sample, Log2_Intensity, na.rm=True)', y='Log2_Intensity', fill=species_col)) +
            geom_violin(alpha=alpha) +
            coord_flip() +
            labs(
                title='Distribution by Sample & Species',
                x='Sample',
                y='Log2(Intensity + 1)',
                fill=species_col
            ) +
            theme_minimal() +
            theme(
                figure_size=(fig_width/2 - 0.5, fig_height),
                legend_position='right'
            )
        )
        st.pyplot(plot_violin)

# ============================================================================
# SECTION 3: TRANSFORMATION COMPARISON
# ============================================================================

elif plot_section == "Transformations":
    st.header("3️⃣ Transformation Comparison")
    st.caption("Assess normality of different transformations to identify optimal preprocessing")
    
    # Combine all numeric data for transformation
    sample_data = df_numeric.values.flatten()
    sample_data = sample_data[~np.isnan(sample_data)]
    
    if len(sample_data) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        # Original
        with col1:
            _, pval_orig = stats.shapiro(sample_data[:5000] if len(sample_data) > 5000 else sample_data)
            st.metric("Original", f"p={pval_orig:.2e}", delta="No transform")
        
        # Log2
        with col2:
            log2_data = np.log2(sample_data + 1)
            _, pval_log2 = stats.shapiro(log2_data[:5000] if len(log2_data) > 5000 else log2_data)
            st.metric("Log2", f"p={pval_log2:.2e}", delta=f"{'+' if pval_log2 > pval_orig else ''}{(pval_log2-pval_orig):.2e}")
        
        # Yeo-Johnson
        with col3:
            pt = PowerTransformer(method='yeo-johnson')
            yj_data = pt.fit_transform(sample_data.reshape(-1, 1)).flatten()
            _, pval_yj = stats.shapiro(yj_data[:5000] if len(yj_data) > 5000 else yj_data)
            st.metric("Yeo-Johnson", f"p={pval_yj:.2e}", delta=f"{'+' if pval_yj > pval_orig else ''}{(pval_yj-pval_orig):.2e}")
        
        # Box-Cox (if all positive)
        with col4:
            if np.all(sample_data > 0):
                bc_data, _ = boxcox(sample_data)
                _, pval_bc = stats.shapiro(bc_data[:5000] if len(bc_data) > 5000 else bc_data)
                st.metric("Box-Cox", f"p={pval_bc:.2e}", delta=f"{'+' if pval_bc > pval_orig else ''}{(pval_bc-pval_orig):.2e}")
            else:
                st.metric("Box-Cox", "N/A", delta="(requires all positive)")
        
        # Visualization
        st.subheader("Transform Distributions")
        
        transforms = {
            'Original': sample_data,
            'Log2': np.log2(sample_data + 1),
            'Yeo-Johnson': pt.fit_transform(sample_data.reshape(-1, 1)).flatten(),
        }
        
        transform_df = pd.DataFrame({
            'Value': np.concatenate([v for v in transforms.values()]),
            'Transform': np.repeat(list(transforms.keys()), len(sample_data))
        })
        
        plot_trans = (
            ggplot(transform_df, aes(x='Value', fill='Transform')) +
            geom_histogram(bins=hist_bins, alpha=alpha, position='identity') +
            facet_wrap('~Transform', scales='free_x', ncol=3) +
            labs(
                title='Transformation Comparison',
                x='Value',
                y='Count'
            ) +
            theme_minimal() +
            theme(
                figure_size=(fig_width, fig_height),
                subplots_adjust={'wspace': 0.3}
            )
        )
        st.pyplot(plot_trans)

# ============================================================================
# SECTION 4: Q-Q PLOTS
# ============================================================================

elif plot_section == "Q-Q Plots":
    st.header("4️⃣ Q-Q Plots - Normality Assessment")
    st.caption("Compare data quantiles to theoretical normal distribution")
    
    # Select samples for Q-Q plotting
    n_samples = min(4, len(numeric_cols))
    selected_samples = st.multiselect(
        "Select samples for Q-Q plots:",
        options=numeric_cols,
        default=numeric_cols[:n_samples],
        key="qq_samples"
    )
    
    if selected_samples:
        n_cols = min(2, len(selected_samples))
        n_rows = (len(selected_samples) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols) if n_rows > 1 else axes.reshape(1, -1)
        
        for idx, sample in enumerate(selected_samples):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if axes.ndim == 2 else axes[idx]
            
            data = df_raw[sample].dropna()
            if len(data) > 0:
                stats.probplot(data, dist="norm", plot=ax)
                ax.set_title(f'Q-Q Plot: {sample}', fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(selected_samples), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if axes.ndim == 2 else axes[idx]
            ax.set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================================
# SECTION 5: PCA
# ============================================================================

elif plot_section == "PCA":
    st.header("5️⃣ Principal Component Analysis (PCA)")
    st.caption("Dimensionality reduction colored by species")
    
    # Prepare data for PCA
    df_pca = df_raw[numeric_cols].fillna(0)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_pca)
    
    # Fit PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df_scaled)
    
    # Create PCA dataframe
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'PC3': pca_result[:, 2],
        species_col: df_raw[species_col]
    })
    
    col1, col2 = st.columns(2)
    
    # 2D PCA
    with col1:
        st.subheader("PCA: PC1 vs PC2")
        fig_2d = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color=species_col,
            title=f'PCA (PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%})',
            labels={
                'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
            }
        )
        st.plotly_chart(fig_2d, use_container_width=True)
    
    # Scree plot
    with col2:
        st.subheader("Scree Plot")
        var_exp = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(var_exp)
        
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(y=var_exp, name='Variance', marker_color='steelblue'))
        fig_scree.add_trace(go.Scatter(y=cumsum_var, name='Cumulative', mode='lines+markers', yaxis='y2'))
        fig_scree.update_layout(
            title='Explained Variance by PC',
            xaxis_title='Principal Component',
            yaxis_title='Variance Explained',
            yaxis2=dict(title='Cumulative', overlaying='y', side='right'),
            hovermode='x unified',
            height=450
        )
        st.plotly_chart(fig_scree, use_container_width=True)

# ============================================================================
# SECTION 6: t-SNE
# ============================================================================

elif plot_section == "t-SNE":
    st.header("6️⃣ t-SNE Plot")
    st.caption("Non-linear dimensionality reduction (takes 30-60 seconds to compute)")
    
    with st.spinner("Computing t-SNE..."):
        df_tsne = df_raw[numeric_cols].fillna(0)
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_tsne)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_result = tsne.fit_transform(df_scaled)
        
        tsne_df = pd.DataFrame({
            't-SNE 1': tsne_result[:, 0],
            't-SNE 2': tsne_result[:, 1],
            species_col: df_raw[species_col]
        })
        
        fig_tsne = px.scatter(
            tsne_df,
            x='t-SNE 1',
            y='t-SNE 2',
            color=species_col,
            title='t-SNE: Sample Relationships',
            labels={
                't-SNE 1': 't-SNE Dimension 1',
                't-SNE 2': 't-SNE Dimension 2'
            }
        )
        st.plotly_chart(fig_tsne, use_container_width=True)

# ============================================================================
# SECTION 7: HEATMAP
# ============================================================================

elif plot_section == "Heatmap":
    st.header("7️⃣ Hierarchical Clustering Heatmap")
    st.caption("Top variable proteins with hierarchical clustering")
    
    # Select top N proteins by variance
    n_proteins = st.slider(
        "Number of proteins to show:",
        min_value=10,
        max_value=100,
        value=30,
        step=10,
        key="n_proteins_heatmap"
    )
    
    # Get top proteins by variance
    var_per_protein = df_raw[numeric_cols].var(axis=1)
    top_idx = var_per_protein.nlargest(n_proteins).index
    df_heatmap = df_raw.loc[top_idx, numeric_cols]
    
    # Z-score normalize
    df_heatmap_norm = (df_heatmap - df_heatmap.mean(axis=1).values.reshape(-1, 1)) / (df_heatmap.std(axis=1).values.reshape(-1, 1) + 1e-10)
    
    # Create heatmap
    fig_heat = px.imshow(
        df_heatmap_norm,
        color_continuous_scale='RdBu_r',
        title=f'Top {n_proteins} Proteins by Variance (Z-score normalized)',
        labels=dict(x='Sample', y='Protein', color='Z-score'),
        aspect='auto'
    )
    fig_heat.update_layout(height=max(400, n_proteins * 5), width=fig_width * 100)
    st.plotly_chart(fig_heat, use_container_width=True)

# ============================================================================
# SECTION 8: MISSING DATA
# ============================================================================

elif plot_section == "Missing Data":
    st.header("8️⃣ Missing Data Analysis")
    
    col1, col2 = st.columns(2)
    
    # Missing data percentage
    with col1:
        st.subheader("Missing Data by Sample")
        missing_pct = (df_raw[numeric_cols].isna().sum() / len(df_raw) * 100).sort_values(ascending=False)
        
        fig_missing = px.bar(
            x=missing_pct.index,
            y=missing_pct.values,
            title='% Missing Values per Sample',
            labels={'x': 'Sample', 'y': '% Missing'},
            color=missing_pct.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_missing, use_container_width=True)
    
    # Valid counts by sample
    with col2:
        st.subheader("Valid Data by Sample")
        valid_counts = df_raw[numeric_cols].notna().sum().sort_values(ascending=False)
        
        fig_valid = px.bar(
            x=valid_counts.index,
            y=valid_counts.values,
            title='Valid Values per Sample',
            labels={'x': 'Sample', 'y': 'Count'},
            color=valid_counts.values,
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_valid, use_container_width=True)
    
    # Summary statistics
    st.subheader("Missing Data Summary")
    missing_summary = pd.DataFrame({
        'Sample': numeric_cols,
        'Total': [len(df_raw)] * len(numeric_cols),
        'Valid': df_raw[numeric_cols].notna().sum().values,
        'Missing': df_raw[numeric_cols].isna().sum().values,
        '% Missing': (df_raw[numeric_cols].isna().sum() / len(df_raw) * 100).round(2).values
    })
    st.dataframe(missing_summary, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("💡 **Tip:** Use the sidebar to customize plot appearance. Export high-resolution figures for publications.")
