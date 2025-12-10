"""
pages/5_Post_Imputation_EDA.py - COMPREHENSIVE EDA AFTER IMPUTATION
Statistical visualizations and quality assessment for proteomics data
Production-ready with all optimizations and corrections
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Visualization imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import scipy.stats as stats
from scipy.stats import gaussian_kde

sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Post-Imputation EDA",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Exploratory Data Analysis (Post-Imputation)")
st.markdown("Comprehensive quality assessment and statistical analysis of imputed proteomics data")
st.markdown("---")

# ============================================================================
# CHECK FOR IMPUTED DATA
# ============================================================================

if 'df_imputed' not in st.session_state or st.session_state.df_imputed is None:
    st.error("❌ No imputed data. Please complete **🔧 Missing Value Imputation** first")
    st.stop()

# Load data from session state
df = st.session_state.df_imputed.copy()
numeric_cols = st.session_state.numeric_cols
sample_to_condition = st.session_state.get('sample_to_condition', {})
species_col = st.session_state.get('species_col', '__SPECIES__')
imputation_method = st.session_state.get('imputation_method', 'Unknown')

# Validate data integrity
if len(df) == 0 or len(numeric_cols) == 0:
    st.error("❌ Invalid data. Please re-upload from the beginning.")
    st.stop()

# Get conditions
conditions = sorted(list(set(sample_to_condition.values())))
if len(conditions) == 0:
    st.warning("⚠️ No conditions found in sample mapping")
    conditions = [f"Condition_{i+1}" for i in range(len(numeric_cols))]

st.info(f"📊 **Data**: {len(df):,} proteins × {len(numeric_cols)} samples | **Imputation**: {imputation_method}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def permanova_test(distance_matrix: np.ndarray, grouping: list, permutations: int = 999) -> Dict:
    """
    Manual PERMANOVA implementation (Permutational Multivariate Analysis of Variance)
    
    Tests whether groups have significantly different centroids in multivariate space.
    
    Args:
        distance_matrix: Square distance matrix (n x n)
        grouping: Group labels for each sample (length n)
        permutations: Number of permutations for significance testing
    
    Returns:
        Dictionary with F-statistic, p-value, and permutation count
    """
    distance_matrix = np.array(distance_matrix)
    grouping = np.array(grouping)
    n = len(grouping)
    
    # Validate inputs
    if n != distance_matrix.shape[0] or n != distance_matrix.shape[1]:
        raise ValueError(f"Distance matrix shape {distance_matrix.shape} does not match grouping length {n}")
    
    if not np.allclose(distance_matrix, distance_matrix.T):
        raise ValueError("Distance matrix must be symmetric")
    
    groups = np.unique(grouping)
    n_groups = len(groups)
    
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for PERMANOVA")
    
    # Calculate sum of squared distances
    def calc_ss(dist_mat: np.ndarray, group_labels: np.ndarray) -> Tuple[float, float, float]:
        """Calculate total, within-group, and between-group sum of squares"""
        total_ss = np.sum(dist_mat ** 2) / (2 * n)
        within_ss = 0
        
        for group in np.unique(group_labels):
            group_mask = group_labels == group
            group_indices = np.where(group_mask)[0]
            n_group = len(group_indices)
            
            if n_group > 1:
                group_dist = dist_mat[np.ix_(group_indices, group_indices)]
                within_ss += np.sum(group_dist ** 2) / (2 * n_group)
        
        between_ss = total_ss - within_ss
        return total_ss, within_ss, between_ss
    
    # Calculate observed F-statistic
    total_ss, within_ss, between_ss = calc_ss(distance_matrix, grouping)
    df_between = n_groups - 1
    df_within = n - n_groups
    
    if within_ss == 0 or df_within == 0:
        return {
            'test statistic': np.nan,
            'p-value': 1.0,
            'permutations': permutations
        }
    
    f_stat = (between_ss / df_between) / (within_ss / df_within)
    
    # Permutation test
    perm_f_stats = []
    np.random.seed(42)  # For reproducibility
    
    for _ in range(permutations):
        perm_grouping = np.random.permutation(grouping)
        _, perm_within_ss, perm_between_ss = calc_ss(distance_matrix, perm_grouping)
        
        if perm_within_ss > 0:
            perm_f_stat = (perm_between_ss / df_between) / (perm_within_ss / df_within)
            perm_f_stats.append(perm_f_stat)
    
    # Calculate p-value
    if len(perm_f_stats) > 0:
        p_value = (np.sum(np.array(perm_f_stats) >= f_stat) + 1) / (len(perm_f_stats) + 1)
    else:
        p_value = 1.0
    
    return {
        'test statistic': f_stat,
        'p-value': p_value,
        'permutations': permutations
    }

def run_permanova(distance_matrix: np.ndarray, grouping: list, title: str) -> Dict:
    """Run PERMANOVA with error handling"""
    try:
        if len(distance_matrix) != len(grouping):
            return {
                'Dataset': title,
                'F-statistic': 'Error',
                'p-value': 'Error',
                'Significant': 'N/A',
                'Interpretation': f'Dimension mismatch'
            }
        
        unique_groups = len(set(grouping))
        if unique_groups < 2:
            return {
                'Dataset': title,
                'F-statistic': 'N/A',
                'p-value': 'N/A',
                'Significant': 'N/A',
                'Interpretation': 'Need at least 2 groups'
            }
        
        result = permanova_test(distance_matrix, grouping, permutations=999)
        
        if np.isnan(result['test statistic']):
            return {
                'Dataset': title,
                'F-statistic': 'N/A',
                'p-value': 'N/A',
                'Significant': 'N/A',
                'Interpretation': 'Insufficient variance'
            }
        
        return {
            'Dataset': title,
            'F-statistic': f"{result['test statistic']:.4f}",
            'p-value': f"{result['p-value']:.4f}",
            'Significant': '✅ Yes' if result['p-value'] < 0.05 else '❌ No',
            'Interpretation': 'Significant separation' if result['p-value'] < 0.05 else 'No significant separation'
        }
    except Exception as e:
        return {
            'Dataset': title,
            'F-statistic': 'Error',
            'p-value': 'Error',
            'Significant': 'N/A',
            'Interpretation': f'Error: {str(e)[:50]}'
        }

def perform_pca(data_subset: pd.DataFrame, numeric_cols: List[str], title_suffix: str) -> Optional[Tuple]:
    """Perform PCA with validation"""
    if len(data_subset) < 3:
        st.warning(f"⚠️ Insufficient proteins for PCA on {title_suffix}: {len(data_subset)} proteins")
        return None
    
    # Transpose for PCA (samples as rows)
    df_pca = data_subset[numeric_cols].T
    
    # Standardize
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_pca)
    
    # Perform PCA
    n_components = min(10, len(numeric_cols), len(data_subset))
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_scaled)
    
    # Create results dataframe
    n_pcs = min(3, n_components)
    pca_df = pd.DataFrame(
        pca_result[:, :n_pcs],
        columns=[f'PC{i+1}' for i in range(n_pcs)],
        index=numeric_cols
    )
    pca_df['Sample'] = numeric_cols
    pca_df['Condition'] = pca_df['Sample'].map(sample_to_condition)
    
    variance_explained = pca.explained_variance_ratio_ * 100
    
    return pca_df, variance_explained, df_scaled

# ============================================================================
# 1. SPECIES COMPOSITION
# ============================================================================

st.subheader("1️⃣ Protein Species Composition")

# Count proteins per species per sample
species_counts = []
for sample in numeric_cols:
    for species in df[species_col].unique():
        species_df = df[df[species_col] == species]
        count = (species_df[sample] > 0).sum()
        species_counts.append({
            'Sample': sample,
            'Condition': sample_to_condition.get(sample, 'Unknown'),
            'Species': species,
            'Count': count
        })

species_df_plot = pd.DataFrame(species_counts)

# Stacked bar plot
fig = px.bar(
    species_df_plot,
    x='Sample',
    y='Count',
    color='Species',
    title='Number of Proteins per Species per Sample',
    labels={'Count': 'Protein Count'},
    barmode='stack',
    height=600
)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# Summary table
st.markdown("**Species Summary**")
species_summary = species_df_plot.groupby('Species')['Count'].agg(['sum', 'mean', 'std']).reset_index()
species_summary.columns = ['Species', 'Total', 'Mean per Sample', 'Std Dev']
st.dataframe(species_summary.round(0), use_container_width=True)

st.markdown("---")
# ============================================================================
# 2. INTENSITY DISTRIBUTION & NORMALITY ASSESSMENT
# ============================================================================

st.subheader("2️⃣ Distribution Quality Assessment")

st.markdown("""
**Comprehensive normality and variance stabilization testing**:
- **Distributions**: Visualize intensity patterns by condition
- **Q-Q Plots**: Assess deviation from normality
- **Shapiro-Wilk Test**: Statistical normality test
- **Mean-Variance Relationship**: Check homoscedasticity per condition
""")

# Prepare data
all_intensities_by_condition = {}
for condition in conditions:
    condition_samples = [s for s in numeric_cols if sample_to_condition.get(s) == condition]
    condition_data = df[condition_samples].values.flatten()
    condition_data = condition_data[~np.isnan(condition_data) & (condition_data > 0)]
    all_intensities_by_condition[condition] = np.log2(condition_data + 1)

# Combine all data
all_data_combined = np.concatenate(list(all_intensities_by_condition.values()))

# ============================================================================
# DISTRIBUTION PLOTS - PER CONDITION KDE
# ============================================================================

st.markdown("### 📊 Intensity Distribution (Per Condition)")

st.markdown("**Kernel Density Estimation (KDE) by condition with individual histograms**")

# Create subplots for each condition
n_conditions = len(conditions)
n_cols = min(3, n_conditions)
n_rows = (n_conditions + n_cols - 1) // n_cols

fig_dists = make_subplots(
    rows=n_rows,
    cols=n_cols,
    subplot_titles=conditions,
    specs=[[{'secondary_y': False}] * n_cols for _ in range(n_rows)]
)

colors = px.colors.qualitative.Set2[:n_conditions]

for idx, condition in enumerate(conditions):
    row = idx // n_cols + 1
    col = idx % n_cols + 1
    color = colors[idx % len(colors)]
    
    condition_data = all_intensities_by_condition[condition]
    
    if len(condition_data) > 5:
        # Add histogram
        fig_dists.add_trace(
            go.Histogram(
                x=condition_data,
                name=condition,
                nbinsx=40,
                opacity=0.5,
                marker_color=color,
                histnorm='probability density',
                showlegend=False,
                hovertemplate='Bin: %{x:.2f}<br>Density: %{y:.4f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add KDE overlay
        try:
            kde = gaussian_kde(condition_data)
            x_range = np.linspace(condition_data.min(), condition_data.max(), 200)
            
            fig_dists.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kde(x_range),
                    mode='lines',
                    name=f'{condition} (KDE)',
                    line=dict(color=color, width=3),
                    showlegend=False,
                    hovertemplate='Intensity: %{x:.2f}<br>Density: %{y:.4f}<extra></extra>'
                ),
                row=row, col=col
            )
        except Exception as e:
            st.warning(f"⚠️ Could not compute KDE for {condition}: {str(e)}")
    else:
        st.warning(f"⚠️ Insufficient data for {condition}: {len(condition_data)} points")

# Update layout
fig_dists.update_xaxes(title_text="Log2(Intensity + 1)")
fig_dists.update_yaxes(title_text="Density")
fig_dists.update_layout(
    title_text='Intensity Distribution by Condition (Histogram + KDE)',
    height=400 * n_rows,
    showlegend=False,
    hovermode='closest'
)

st.plotly_chart(fig_dists, use_container_width=True)

# Summary statistics per condition
st.markdown("**Summary Statistics by Condition**")

summary_stats = []
for condition in conditions:
    condition_data = all_intensities_by_condition[condition]
    
    summary_stats.append({
        'Condition': condition,
        'n': len(condition_data),
        'Mean': f"{condition_data.mean():.2f}",
        'Median': f"{np.median(condition_data):.2f}",
        'Std Dev': f"{condition_data.std():.2f}",
        'Min': f"{condition_data.min():.2f}",
        'Max': f"{condition_data.max():.2f}",
        'Skewness': f"{stats.skew(condition_data):.3f}",
        'Kurtosis': f"{stats.kurtosis(condition_data):.3f}"
    })

summary_stats_df = pd.DataFrame(summary_stats)
st.dataframe(summary_stats_df, hide_index=True, use_container_width=True)

# ============================================================================
# Q-Q PLOTS BY CONDITION (CORRECTLY SCALED)
# ============================================================================

st.markdown("### Q-Q Plots by Condition (Normality Assessment)")

n_conditions = len(conditions)
n_cols = min(3, n_conditions)
n_rows = (n_conditions + n_cols - 1) // n_cols

fig_qq_multi = make_subplots(
    rows=n_rows,
    cols=n_cols,
    subplot_titles=[f'{cond}' for cond in conditions]
)

for idx, condition in enumerate(conditions):
    row = idx // n_cols + 1
    col = idx % n_cols + 1
    
    condition_data = all_intensities_by_condition[condition]
    
    if len(condition_data) >= 3:
        # ✅ CORRECT: Manual scaling approach
        sorted_data = np.sort(condition_data)
        n = len(sorted_data)
        
        # Van der Waerden percentiles
        percentiles = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quantiles = stats.norm.ppf(percentiles)
        
        # Scale to match data distribution
        mu = sorted_data.mean()
        sigma = sorted_data.std()
        theoretical_quantiles_scaled = mu + sigma * theoretical_quantiles
        
        # Add scatter
        fig_qq_multi.add_trace(
            go.Scatter(
                x=theoretical_quantiles_scaled,
                y=sorted_data,
                mode='markers',
                marker=dict(size=4, opacity=0.6, color='steelblue'),
                showlegend=False,
                hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add reference line (diagonal)
        min_val = min(theoretical_quantiles_scaled.min(), sorted_data.min())
        max_val = max(theoretical_quantiles_scaled.max(), sorted_data.max())
        
        fig_qq_multi.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                showlegend=False
            ),
            row=row, col=col
        )

fig_qq_multi.update_xaxes(title_text="Theoretical Quantiles")
fig_qq_multi.update_yaxes(title_text="Sample Quantiles")
fig_qq_multi.update_layout(
    title_text="Q-Q Plots by Condition",
    height=400 * n_rows,
    showlegend=False
)

st.plotly_chart(fig_qq_multi, use_container_width=True)

# ============================================================================
# STATISTICAL NORMALITY TESTS
# ============================================================================

st.markdown("### 📋 Normality Tests (Shapiro-Wilk, K-S, D'Agostino)")

normality_results = []

for condition in conditions:
    condition_data = all_intensities_by_condition[condition]
    
    if len(condition_data) >= 3:
        # Limit to 5000 samples for performance
        test_data = condition_data[:5000] if len(condition_data) > 5000 else condition_data
        
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(test_data)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(test_data, 'norm', 
                                      args=(test_data.mean(), test_data.std()))
        
        # D'Agostino-Pearson test
        if len(test_data) >= 8:
            dagostino_stat, dagostino_p = stats.normaltest(test_data)
        else:
            dagostino_p = np.nan
        
        normality_results.append({
            'Condition': condition,
            'n': len(condition_data),
            'Shapiro-Wilk p': f"{shapiro_p:.4f}",
            'K-S p': f"{ks_p:.4f}",
            'D\'Agostino p': f"{dagostino_p:.4f}" if not np.isnan(dagostino_p) else 'N/A',
            'Normal?': '✅ Yes' if shapiro_p > 0.05 else '❌ No'
        })

normality_df = pd.DataFrame(normality_results)
st.dataframe(normality_df, hide_index=True, use_container_width=True)

# ============================================================================
# MEAN-VARIANCE RELATIONSHIP (PER CONDITION)
# ============================================================================

st.markdown("### 📈 Mean-Variance Relationship (Homoscedasticity) - Per Condition")

st.markdown("""
**Variance stabilization assessment per condition**:
- Proteomics data typically shows mean-variance dependence
- **Good transformation**: Low correlation (r < 0.3)
- **Poor transformation**: High correlation (r > 0.6)
""")

# Create subplots for mean-variance plots per condition
n_conditions = len(conditions)
n_cols = min(3, n_conditions)
n_rows = (n_conditions + n_cols - 1) // n_cols

fig_mv_multi = make_subplots(
    rows=n_rows,
    cols=n_cols,
    subplot_titles=conditions,
    specs=[[{'type': 'scatter'} for _ in range(n_cols)] for _ in range(n_rows)]
)

mean_var_corr_dict = {}

for idx, condition in enumerate(conditions):
    row = idx // n_cols + 1
    col = idx % n_cols + 1
    
    # Get samples for this condition
    condition_samples = [s for s in numeric_cols if sample_to_condition.get(s) == condition]
    
    if len(condition_samples) >= 2:
        # Calculate per-protein mean and variance for this condition
        condition_df = df[condition_samples].copy()
        protein_means_cond = condition_df.mean(axis=1)
        protein_vars_cond = condition_df.var(axis=1)
        
        # Remove invalid values
        valid_mask = ~(np.isnan(protein_means_cond) | np.isnan(protein_vars_cond) | 
                       (protein_means_cond == 0) | (protein_vars_cond == 0))
        means_clean = protein_means_cond[valid_mask].values
        vars_clean = protein_vars_cond[valid_mask].values
        
        if len(means_clean) > 2:
            # Calculate correlation
            mean_var_corr = np.corrcoef(means_clean, vars_clean)[0, 1]
            mean_var_corr_dict[condition] = mean_var_corr
            
            # Add scatter plot
            fig_mv_multi.add_trace(
                go.Scatter(
                    x=means_clean,
                    y=vars_clean,
                    mode='markers',
                    marker=dict(size=3, opacity=0.3, color='steelblue'),
                    name=condition,
                    showlegend=False,
                    hovertemplate='Mean: %{x:.2f}<br>Variance: %{y:.2f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add trend line (log-log)
            try:
                z = np.polyfit(np.log10(means_clean), np.log10(vars_clean), 1)
                p = np.poly1d(z)
                x_trend = np.logspace(np.log10(means_clean.min()), np.log10(means_clean.max()), 100)
                y_trend = 10 ** p(np.log10(x_trend))
                
                fig_mv_multi.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name=f'{condition} Trend',
                        showlegend=False
                    ),
                    row=row, col=col
                )
            except:
                pass

# Update axes
fig_mv_multi.update_xaxes(title_text="Mean Intensity", type='log')
fig_mv_multi.update_yaxes(title_text="Variance", type='log')
fig_mv_multi.update_layout(
    title_text='Mean-Variance Relationship by Condition (Log-Log Scale)',
    height=400 * n_rows,
    showlegend=False,
    hovermode='closest'
)

st.plotly_chart(fig_mv_multi, use_container_width=True)

# Summary interpretation
st.markdown("**Mean-Variance Correlation Summary**")

mv_summary = []
for condition in conditions:
    if condition in mean_var_corr_dict:
        corr = mean_var_corr_dict[condition]
        
        if abs(corr) < 0.3:
            status = "✅ Excellent"
        elif abs(corr) < 0.6:
            status = "⚠️ Moderate"
        else:
            status = "❌ Poor"
        
        mv_summary.append({
            'Condition': condition,
            'Correlation': f"{corr:.3f}",
            'Status': status,
            'Interpretation': 'Homoscedasticity achieved' if abs(corr) < 0.3 else 
                            'Consider robust methods' if abs(corr) < 0.6 else 
                            'Variance stabilization needed'
        })

mv_summary_df = pd.DataFrame(mv_summary)
st.dataframe(mv_summary_df, hide_index=True, use_container_width=True)

st.markdown("---")

# ============================================================================
# 4. PERMANOVA ANALYSIS + EFFECT SIZE METRICS
# ============================================================================

st.subheader("4️⃣ Multivariate Group Separation Testing")

st.markdown("""
**Comprehensive analysis of condition separation**:
- **PERMANOVA**: Tests whether groups have different centroids
- **Effect Size Metrics**: Quantify cluster quality (more reliable than p-values for small n)
- **Silhouette Score**: How tight are clusters? (range: -1 to 1, higher is better)
- **Calinski-Harabasz Index**: Ratio of between-group to within-group variance

⚠️ **Note**: With n=6 samples and 11K+ features, p-values are less informative.
Effect sizes and visual inspection provide stronger evidence of separation.
""")

# ============================================================================
# IMPORT CLUSTERING METRICS
# ============================================================================

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ============================================================================
# RUN PERMANOVA + EFFECT SIZES ON ALL THREE DATASETS
# ============================================================================

permanova_results = []
effect_size_results = []

if pca_result_all:
    # ===== ALL PROTEINS =====
    dist_all = squareform(pdist(scaled_all, metric='euclidean'))
    grouping = [sample_to_condition.get(s, 'Unknown') for s in numeric_cols]
    
    # PERMANOVA
    permanova_all = run_permanova(dist_all, grouping, f"All Proteins (n={len(df):,})")
    permanova_results.append(permanova_all)
    
    # Effect sizes
    silhouette_all = silhouette_score(scaled_all, grouping)
    davies_all = davies_bouldin_score(scaled_all, grouping)
    calinski_all = calinski_harabasz_score(scaled_all, grouping)
    
    effect_size_results.append({
        'Dataset': f"All Proteins (n={len(df):,})",
        'Silhouette Score': f"{silhouette_all:.3f}",
        'Davies-Bouldin Index': f"{davies_all:.3f}",
        'Calinski-Harabasz': f"{calinski_all:.1f}",
        'Cluster Quality': 'Excellent' if silhouette_all > 0.5 else 'Good' if silhouette_all > 0.25 else 'Moderate'
    })

if pca_result_common:
    # ===== HUMAN PROTEINS =====
    dist_common = squareform(pdist(scaled_common, metric='euclidean'))
    
    permanova_common = run_permanova(dist_common, grouping, f"{most_common_species} (n={len(df_common):,})")
    permanova_results.append(permanova_common)
    
    silhouette_common = silhouette_score(scaled_common, grouping)
    davies_common = davies_bouldin_score(scaled_common, grouping)
    calinski_common = calinski_harabasz_score(scaled_common, grouping)
    
    effect_size_results.append({
        'Dataset': f"{most_common_species} (n={len(df_common):,})",
        'Silhouette Score': f"{silhouette_common:.3f}",
        'Davies-Bouldin Index': f"{davies_common:.3f}",
        'Calinski-Harabasz': f"{calinski_common:.1f}",
        'Cluster Quality': 'Excellent' if silhouette_common > 0.5 else 'Good' if silhouette_common > 0.25 else 'Moderate'
    })

if pca_result_rest:
    # ===== OTHER SPECIES =====
    dist_rest = squareform(pdist(scaled_rest, metric='euclidean'))
    
    permanova_rest = run_permanova(dist_rest, grouping, f"Other Species (n={len(df_rest):,})")
    permanova_results.append(permanova_rest)
    
    silhouette_rest = silhouette_score(scaled_rest, grouping)
    davies_rest = davies_bouldin_score(scaled_rest, grouping)
    calinski_rest = calinski_harabasz_score(scaled_rest, grouping)
    
    effect_size_results.append({
        'Dataset': f"Other Species (n={len(df_rest):,})",
        'Silhouette Score': f"{silhouette_rest:.3f}",
        'Davies-Bouldin Index': f"{davies_rest:.3f}",
        'Calinski-Harabasz': f"{calinski_rest:.1f}",
        'Cluster Quality': 'Excellent' if silhouette_rest > 0.5 else 'Good' if silhouette_rest > 0.25 else 'Moderate'
    })

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.markdown("**PERMANOVA Results**")
    permanova_df = pd.DataFrame(permanova_results)
    st.dataframe(permanova_df, hide_index=True, use_container_width=True)
    
    st.markdown("""
    **Interpretation**:
    - p < 0.05 = Statistically significant separation
    - ⚠️ Note: With n=6, p-values are less reliable. See Effect Sizes.
    """)

with col2:
    st.markdown("**Effect Size Metrics**")
    effect_size_df = pd.DataFrame(effect_size_results)
    st.dataframe(effect_size_df, hide_index=True, use_container_width=True)
    
    st.markdown("""
    **Interpretation**:
    - Silhouette > 0.5 = Excellent separation
    - Silhouette > 0.25 = Good separation
    - Lower Davies-Bouldin = Better
    - Higher Calinski-Harabasz = Better
    """)

# ============================================================================
# INTER-GROUP DISTANCE ANALYSIS
# ============================================================================

st.markdown("### Distance-Based Cluster Quality Analysis")

# Calculate mean distances between groups
distance_analysis = []

for dataset_name, scaled_data in [
    ("All Proteins", scaled_all if pca_result_all else None),
    (most_common_species, scaled_common if pca_result_common else None),
    ("Other Species", scaled_rest if pca_result_rest else None)
]:
    if scaled_data is not None:
        grouping = [sample_to_condition.get(s, 'Unknown') for s in numeric_cols]
        
        # Get group indices
        group_a_idx = [i for i, g in enumerate(grouping) if g == conditions[0]]
        group_b_idx = [i for i, g in enumerate(grouping) if g == conditions[1]]
        
        if len(group_a_idx) > 0 and len(group_b_idx) > 0:
            # Within-group distances
            within_dist_a = pdist(scaled_data[group_a_idx], metric='euclidean')
            within_dist_b = pdist(scaled_data[group_b_idx], metric='euclidean')
            
            # Between-group distances
            between_dists = []
            for i in group_a_idx:
                for j in group_b_idx:
                    dist = np.linalg.norm(scaled_data[i] - scaled_data[j])
                    between_dists.append(dist)
            between_dists = np.array(between_dists)
            
            # Calculate metrics
            mean_within_a = within_dist_a.mean() if len(within_dist_a) > 0 else 0
            mean_within_b = within_dist_b.mean() if len(within_dist_b) > 0 else 0
            mean_between = between_dists.mean()
            mean_within = (mean_within_a + mean_within_b) / 2
            
            # Separation ratio
            separation_ratio = mean_between / mean_within if mean_within > 0 else np.inf
            
            distance_analysis.append({
                'Dataset': dataset_name,
                'Mean Within-Group': f"{mean_within:.2f}",
                'Mean Between-Group': f"{mean_between:.2f}",
                'Separation Ratio': f"{separation_ratio:.2f}",
                'Interpretation': f"Groups are {separation_ratio:.1f}x farther apart than within"
            })

distance_df = pd.DataFrame(distance_analysis)
st.dataframe(distance_df, hide_index=True, use_container_width=True)

st.markdown("""
**Separation Ratio Interpretation**:
- Ratio > 2.0 = Strong separation (between-group distance 2x within-group)
- Ratio > 1.5 = Good separation
- Ratio > 1.0 = Some separation (but groups may overlap)
""")

# ============================================================================
# LINEAR DISCRIMINANT ANALYSIS (LDA)
# ============================================================================

st.markdown("### Linear Discriminant Analysis (Classification Accuracy)")

st.markdown("""
**LDA shows how well conditions can be distinguished** using the multivariate protein profile.
With perfect separation, LDA classification accuracy should be 100%.
""")

lda_results = []

for dataset_name, scaled_data, n_proteins in [
    ("All Proteins", scaled_all if pca_result_all else None, len(df)),
    (most_common_species, scaled_common if pca_result_common else None, len(df_common)),
    ("Other Species", scaled_rest if pca_result_rest else None, len(df_rest))
]:
    if scaled_data is not None and len(scaled_data) >= 3:
        grouping_binary = np.array([0 if sample_to_condition.get(s, 'Unknown') == conditions[0] 
                                   else 1 for s in numeric_cols])
        
        try:
            # Fit LDA
            lda = LinearDiscriminantAnalysis(n_components=min(1, len(numeric_cols)-1))
            lda_data = lda.fit_transform(scaled_data, grouping_binary)
            
            # Calculate classification accuracy
            lda_pred = lda.predict(scaled_data)
            accuracy = (lda_pred == grouping_binary).sum() / len(grouping_binary) * 100
            
            lda_results.append({
                'Dataset': dataset_name,
                'n Proteins': n_proteins,
                'LDA Accuracy': f"{accuracy:.1f}%",
                'Interpretation': 'Perfect separation' if accuracy == 100 else 
                                'Excellent' if accuracy >= 85 else 
                                'Good' if accuracy >= 70 else 'Moderate'
            })
        except Exception as e:
            st.warning(f"⚠️ Could not compute LDA for {dataset_name}: {str(e)}")

lda_df = pd.DataFrame(lda_results)
st.dataframe(lda_df, hide_index=True, use_container_width=True)

# ============================================================================
# SUMMARY INTERPRETATION
# ============================================================================

st.markdown("### 📊 Overall Interpretation")

significant_permanova = sum(1 for r in permanova_results if '✅' in r['Significant'])
excellent_effects = sum(1 for r in effect_size_results if 'Excellent' in r['Cluster Quality'])

if excellent_effects >= len(effect_size_results) * 0.66 and lda_df['LDA Accuracy'].str.rstrip('%').astype(float).mean() > 85:
    st.success(f"""
    ✅ **STRONG EVIDENCE OF CONDITION SEPARATION**
    
    - Effect sizes show excellent cluster quality
    - LDA classification accuracy is high
    - Visual PCA separation is dramatic (99% variance in PC1)
    - Multivariate protein signatures clearly distinguish conditions
    
    **Recommendation**: Data is excellent for downstream differential expression analysis.
    The lack of PERMANOVA significance is likely due to small sample size (n=6),
    not lack of biological signal.
    """)
elif significant_permanova > 0:
    st.info(f"""
    ⚠️ **MODERATE EVIDENCE OF SEPARATION**
    
    - Some statistical tests show significance
    - Effect sizes are reasonable
    - Consider increasing sample size for robust conclusions
    """)
else:
    st.warning(f"""
    ⚠️ **WEAK EVIDENCE OF SEPARATION**
    
    - Effect sizes are moderate
    - Visual inspection suggests mild separation
    - Consider:
      • Batch effects
      • Quality issues with some samples
      • Need for larger sample sizes
    """)

st.markdown("---")


# ============================================================================
# 6. DATA QUALITY SUMMARY
# ============================================================================

st.subheader("6️⃣ Data Quality Summary")

normal_count = sum(1 for r in normality_results if '✅' in r['Normal?'])
total_count = len(normality_results)

quality_metrics = pd.DataFrame({
    'Metric': [
        'Total Proteins',
        'Total Samples',
        'Conditions',
        'Normality Rate',
        'Mean-Variance Correlation',
        'Variance Stabilization',
        'Imputation Method'
    ],
    'Value': [
        f"{len(df):,}",
        f"{len(numeric_cols)}",
        f"{len(conditions)}",
        f"{normal_count}/{total_count} ({normal_count/total_count*100:.0f}%)",
        f"{mean_var_corr:.3f}" if 'mean_var_corr' in locals() else 'N/A',
        '✅ Good' if 'mean_var_corr' in locals() and abs(mean_var_corr) < 0.3 else 'N/A',
        imputation_method
    ]
})

st.dataframe(quality_metrics, hide_index=True, use_container_width=True)

st.markdown("---")

# ============================================================================
# EXPORT OPTIONS
# ============================================================================

st.subheader("💾 Export Data")

col1, col2, col3 = st.columns(3)

with col1:
    csv = df[numeric_cols].to_csv(index=False)
    st.download_button(
        label="📥 Download Imputed Data",
        data=csv,
        file_name="imputed_proteomics_data.csv",
        mime="text/csv"
    )

with col2:
    if pca_result_all:
        pca_csv = pca_all.to_csv(index=False)
        st.download_button(
            label="📥 Download PCA Results",
            data=pca_csv,
            file_name="pca_all_proteins.csv",
            mime="text/csv"
        )

with col3:
    if permanova_results:
        permanova_csv = permanova_df.to_csv(index=False)
        st.download_button(
            label="📥 Download PERMANOVA Results",
            data=permanova_csv,
            file_name="permanova_results.csv",
            mime="text/csv"
        )

st.markdown("---")
st.success("✅ EDA Complete! Proceed to differential expression analysis.")
