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
- **Mean-Variance Relationship**: Check homoscedasticity
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
# DISTRIBUTION PLOTS
# ============================================================================

st.markdown("### 📊 Distribution Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Histogram with KDE Overlay**")
    fig_hist = go.Figure()
    
    for condition in conditions:
        condition_data = all_intensities_by_condition[condition]
        
        fig_hist.add_trace(go.Histogram(
            x=condition_data,
            name=condition,
            opacity=0.5,
            nbinsx=50,
            histnorm='probability density'
        ))
        
        # Add KDE overlay
        if len(condition_data) > 10:
            kde = gaussian_kde(condition_data)
            x_range = np.linspace(condition_data.min(), condition_data.max(), 200)
            
            fig_hist.add_trace(go.Scatter(
                x=x_range,
                y=kde(x_range),
                mode='lines',
                name=f'{condition} (KDE)',
                line=dict(width=3)
            ))
    
    fig_hist.update_layout(
        title='Log2(Intensity) Distribution',
        xaxis_title='Log2(Intensity + 1)',
        yaxis_title='Density',
        barmode='overlay',
        height=450,
        hovermode='x unified'
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.markdown("**Q-Q Plot (Combined Data)**")
    
    # Generate Q-Q plot using scipy.stats.probplot (CORRECT implementation)
    theoretical_q, sample_q = stats.probplot(all_data_combined, dist='norm')[0]
    
    fig_qq = go.Figure()
    
    # Add scatter
    fig_qq.add_trace(go.Scatter(
        x=theoretical_q,
        y=sample_q,
        mode='markers',
        marker=dict(size=4, opacity=0.5, color='steelblue'),
        name='Data',
        hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
    ))
    
    # Add reference line (y=x)
    min_val = min(theoretical_q.min(), sample_q.min())
    max_val = max(theoretical_q.max(), sample_q.max())
    fig_qq.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect Normal'
    ))
    
    fig_qq.update_layout(
        title='Q-Q Plot (All Data)',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles',
        height=450
    )
    st.plotly_chart(fig_qq, use_container_width=True)

# ============================================================================
# Q-Q PLOTS BY CONDITION
# ============================================================================

st.markdown("### Q-Q Plots by Condition")

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
        # CORRECT: Use scipy.stats.probplot
        theoretical_q, sample_q = stats.probplot(condition_data, dist='norm')[0]
        
        # Add scatter
        fig_qq_multi.add_trace(
            go.Scatter(
                x=theoretical_q,
                y=sample_q,
                mode='markers',
                marker=dict(size=4, opacity=0.6, color='steelblue'),
                showlegend=False,
                hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add reference line
        min_val = min(theoretical_q.min(), sample_q.min())
        max_val = max(theoretical_q.max(), sample_q.max())
        
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

st.markdown("### 📋 Normality Tests")

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
# MEAN-VARIANCE RELATIONSHIP
# ============================================================================

st.markdown("### 📈 Mean-Variance Relationship (Homoscedasticity)")

st.markdown("""
**Variance stabilization assessment**:
- Proteomics data typically shows mean-variance dependence
- **Good transformation**: Low correlation (r < 0.3)
- **Poor transformation**: High correlation (r > 0.6)
""")

# Calculate per-protein statistics
protein_means = df[numeric_cols].mean(axis=1)
protein_vars = df[numeric_cols].var(axis=1)

# Remove invalid values
valid_mask = ~(np.isnan(protein_means) | np.isnan(protein_vars) | (protein_means == 0) | (protein_vars == 0))
means_clean = protein_means[valid_mask].values
vars_clean = protein_vars[valid_mask].values

if len(means_clean) > 2:
    # Calculate correlation
    mean_var_corr = np.corrcoef(means_clean, vars_clean)[0, 1]
    
    # Create scatter plot
    fig_mv = go.Figure()
    
    fig_mv.add_trace(go.Scatter(
        x=means_clean,
        y=vars_clean,
        mode='markers',
        marker=dict(size=3, opacity=0.3, color='steelblue'),
        name='Proteins',
        hovertemplate='Mean: %{x:.2f}<br>Variance: %{y:.2f}<extra></extra>'
    ))
    
    # Add trend line (log-log)
    z = np.polyfit(np.log10(means_clean), np.log10(vars_clean), 1)
    p = np.poly1d(z)
    x_trend = np.logspace(np.log10(means_clean.min()), np.log10(means_clean.max()), 100)
    y_trend = 10 ** p(np.log10(x_trend))
    
    fig_mv.add_trace(go.Scatter(
        x=x_trend,
        y=y_trend,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name=f'Trend (r={mean_var_corr:.3f})'
    ))
    
    fig_mv.update_layout(
        title='Mean-Variance Relationship',
        xaxis_title='Mean Intensity',
        yaxis_title='Variance',
        xaxis_type='log',
        yaxis_type='log',
        height=500
    )
    
    st.plotly_chart(fig_mv, use_container_width=True)
    
    # Interpretation
    if abs(mean_var_corr) < 0.3:
        st.success(f"""
        ✅ **Excellent variance stabilization** (r = {mean_var_corr:.3f})
        
        Low correlation indicates homoscedasticity. Ideal for parametric tests.
        """)
    elif abs(mean_var_corr) < 0.6:
        st.info(f"""
        ⚠️ **Moderate variance stabilization** (r = {mean_var_corr:.3f})
        
        Consider variance-stabilizing transformation or robust methods.
        """)
    else:
        st.warning(f"""
        ❌ **Poor variance stabilization** (r = {mean_var_corr:.3f})
        
        Apply VSN or use non-parametric tests.
        """)
else:
    st.warning("⚠️ Insufficient data for mean-variance analysis")

st.markdown("---")

# ============================================================================
# 3. PCA ANALYSIS
# ============================================================================

st.subheader("3️⃣ Principal Component Analysis (PCA)")

# Get most common species
if species_col in df.columns:
    species_counts_total = df[species_col].value_counts()
    most_common_species = species_counts_total.index[0]
    st.info(f"🔬 **Most Common Proteome**: {most_common_species} ({species_counts_total[most_common_species]:,} proteins)")
else:
    most_common_species = None
    st.warning("⚠️ Species column not found")

# Perform PCA on three datasets
st.markdown("### PCA on Three Protein Subsets")

# 1. All proteins
pca_result_all = perform_pca(df, numeric_cols, "All Proteins")

# 2. Most common species
if most_common_species:
    df_common = df[df[species_col] == most_common_species].copy()
    pca_result_common = perform_pca(df_common, numeric_cols, most_common_species)
else:
    pca_result_common = None

# 3. Other species
if most_common_species:
    df_rest = df[df[species_col] != most_common_species].copy()
    pca_result_rest = perform_pca(df_rest, numeric_cols, "Other Species")
else:
    pca_result_rest = None

# Create three PCA plots
col1, col2, col3 = st.columns(3)

if pca_result_all:
    pca_all, var_all, scaled_all = pca_result_all
    
    with col1:
        st.markdown(f"**All Proteins** (n={len(df):,})")
        fig1 = px.scatter(
            pca_all,
            x='PC1',
            y='PC2',
            color='Condition',
            text='Sample',
            title=f'PC1 ({var_all[0]:.1f}%) vs PC2 ({var_all[1]:.1f}%)',
            labels={
                'PC1': f'PC1 ({var_all[0]:.1f}%)',
                'PC2': f'PC2 ({var_all[1]:.1f}%)'
            },
            height=500
        )
        fig1.update_traces(textposition='top center', marker=dict(size=10))
        st.plotly_chart(fig1, use_container_width=True)

if pca_result_common:
    pca_common, var_common, scaled_common = pca_result_common
    
    with col2:
        st.markdown(f"**{most_common_species} Only** (n={len(df_common):,})")
        fig2 = px.scatter(
            pca_common,
            x='PC1',
            y='PC2',
            color='Condition',
            text='Sample',
            title=f'PC1 ({var_common[0]:.1f}%) vs PC2 ({var_common[1]:.1f}%)',
            labels={
                'PC1': f'PC1 ({var_common[0]:.1f}%)',
                'PC2': f'PC2 ({var_common[1]:.1f}%)'
            },
            height=500
        )
        fig2.update_traces(textposition='top center', marker=dict(size=10))
        st.plotly_chart(fig2, use_container_width=True)

if pca_result_rest:
    pca_rest, var_rest, scaled_rest = pca_result_rest
    
    with col3:
        st.markdown(f"**Other Species** (n={len(df_rest):,})")
        fig3 = px.scatter(
            pca_rest,
            x='PC1',
            y='PC2',
            color='Condition',
            text='Sample',
            title=f'PC1 ({var_rest[0]:.1f}%) vs PC2 ({var_rest[1]:.1f}%)',
            labels={
                'PC1': f'PC1 ({var_rest[0]:.1f}%)',
                'PC2': f'PC2 ({var_rest[1]:.1f}%)'
            },
            height=500
        )
        fig3.update_traces(textposition='top center', marker=dict(size=10))
        st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ============================================================================
# 4. PERMANOVA ANALYSIS
# ============================================================================

st.subheader("4️⃣ PERMANOVA - Statistical Testing of Group Separation")

st.markdown("""
**PERMANOVA** tests whether groups have different centroids in multivariate space.
- **p < 0.05**: Significant separation between conditions
- **p ≥ 0.05**: No significant separation
""")

permanova_results = []

# Run PERMANOVA on all three datasets
if pca_result_all:
    dist_all = squareform(pdist(scaled_all, metric='euclidean'))
    grouping = [sample_to_condition.get(s, 'Unknown') for s in numeric_cols]
    permanova_results.append(run_permanova(dist_all, grouping, f"All Proteins (n={len(df):,})"))

if pca_result_common:
    dist_common = squareform(pdist(scaled_common, metric='euclidean'))
    permanova_results.append(run_permanova(dist_common, grouping, f"{most_common_species} (n={len(df_common):,})"))

if pca_result_rest:
    dist_rest = squareform(pdist(scaled_rest, metric='euclidean'))
    permanova_results.append(run_permanova(dist_rest, grouping, f"Other Species (n={len(df_rest):,})"))

if permanova_results:
    permanova_df = pd.DataFrame(permanova_results)
    st.dataframe(permanova_df, hide_index=True, use_container_width=True)
    
    # Interpretation
    significant_count = sum(1 for r in permanova_results if '✅' in r['Significant'])
    
    if significant_count == len(permanova_results):
        st.success(f"""
        **Strong Evidence of Condition Separation**: All analyses show significant group differences.
        Data quality is excellent and ready for differential expression analysis.
        """)
    elif significant_count > 0:
        st.info(f"""
        **Moderate Evidence**: {significant_count}/{len(permanova_results)} analyses show significant separation.
        Consider species-stratified analysis.
        """)
    else:
        st.warning(f"""
        **Weak Evidence**: No significant separation detected.
        Check for batch effects or consider alternative experimental design.
        """)

st.markdown("---")

# ============================================================================
# 5. HIERARCHICAL CLUSTERING HEATMAP
# ============================================================================

st.subheader("5️⃣ Hierarchical Clustering Heatmap")

st.markdown("**Sample-to-sample correlation with hierarchical clustering**")

# Calculate correlation matrix
corr_matrix = df[numeric_cols].corr()

# Calculate linkage
linkage_samples = linkage(corr_matrix, method='ward')

# Get reorder from dendrogram
dend = dendrogram(linkage_samples, labels=numeric_cols, no_plot=True)
reordered_idx = dend['leaves']
corr_reordered = corr_matrix.iloc[reordered_idx, reordered_idx]

# Create heatmap
fig_heatmap = go.Figure(data=go.Heatmap(
    z=corr_reordered.values,
    x=corr_reordered.columns,
    y=corr_reordered.columns,
    colorscale='RdBu_r',
    zmid=0,
    zmin=-1,
    zmax=1,
    colorbar=dict(title="Correlation"),
    hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
))

fig_heatmap.update_layout(
    title='Hierarchical Clustering Heatmap (Ward Linkage)',
    xaxis_title='Sample',
    yaxis_title='Sample',
    height=800,
    xaxis={'side': 'bottom', 'tickangle': -45}
)

st.plotly_chart(fig_heatmap, use_container_width=True)

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
