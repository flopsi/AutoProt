"""
pages/5_Post_Imputation_EDA.py - COMPREHENSIVE EDA AFTER IMPUTATION
Statistical visualizations for quality-controlled proteomics data
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Visualization imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import scipy.stats as stats
from skbio.stats.distance import permanova

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
st.markdown("Comprehensive visual analysis of clean, imputed proteomics data")
st.markdown("---")

# ============================================================================
# CHECK FOR IMPUTED DATA
# ============================================================================

if 'df_imputed' not in st.session_state or st.session_state.df_imputed is None:
    st.error("❌ No imputed data. Please complete **🔧 Missing Value Imputation** first")
    st.stop()

# Load data
df = st.session_state.df_imputed.copy()
numeric_cols = st.session_state.numeric_cols
sample_to_condition = st.session_state.get('sample_to_condition', {})
species_col = st.session_state.species_col
imputation_method = st.session_state.get('imputation_method', 'Unknown')

# Get conditions
conditions = sorted(list(set(sample_to_condition.values())))
condition_samples = {}
for sample, condition in sample_to_condition.items():
    if sample in numeric_cols:
        if condition not in condition_samples:
            condition_samples[condition] = []
        condition_samples[condition].append(sample)

st.info(f"📊 **Data**: {len(df):,} proteins × {len(numeric_cols)} samples | **Imputation**: {imputation_method}")

# ============================================================================
# 1. SPECIES COMPOSITION - STACKED BAR PLOT
# ============================================================================

st.subheader("1️⃣ Protein Species Composition per Sample")

# Count proteins per species per sample
species_counts = []
for sample in numeric_cols:
    # Count non-zero/non-null proteins per species
    for species in df[species_col].unique():
        species_df = df[df[species_col] == species]
        # Count proteins with valid intensity in this sample
        count = (species_df[sample] > 0).sum()
        species_counts.append({
            'Sample': sample,
            'Condition': sample_to_condition.get(sample, 'Unknown'),
            'Species': species,
            'Count': count
        })

species_df = pd.DataFrame(species_counts)

# Create stacked bar plot
fig = px.bar(
    species_df,
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
species_summary = species_df.groupby('Species')['Count'].agg(['sum', 'mean', 'std'])
species_summary.columns = ['Total', 'Mean per Sample', 'Std Dev']
st.dataframe(species_summary.round(0), use_container_width=True)

st.markdown("---")

# ============================================================================
# 2. INTENSITY DISTRIBUTION - HISTOGRAM WITH KDE
# ============================================================================

st.subheader("2️⃣ Intensity Distribution Analysis")

st.markdown("**Histogram with Kernel Density Estimation (KDE)**")

# Prepare data for histogram
all_intensities = []
for col in numeric_cols:
    intensities = df[col].dropna()
    log_intensities = np.log2(intensities[intensities > 0] + 1)
    for val in log_intensities:
        all_intensities.append({
            'Sample': col,
            'Condition': sample_to_condition.get(col, 'Unknown'),
            'Log2_Intensity': val
        })

intensity_df = pd.DataFrame(all_intensities)

# Create histogram with KDE overlay
fig = go.Figure()

# Add histogram for each condition
for condition in conditions:
    condition_data = intensity_df[intensity_df['Condition'] == condition]['Log2_Intensity']
    
    fig.add_trace(go.Histogram(
        x=condition_data,
        name=condition,
        opacity=0.6,
        nbinsx=50,
        histnorm='probability density'
    ))

# Add KDE overlay for each condition
from scipy.stats import gaussian_kde

for condition in conditions:
    condition_data = intensity_df[intensity_df['Condition'] == condition]['Log2_Intensity'].values
    
    if len(condition_data) > 0:
        kde = gaussian_kde(condition_data)
        x_range = np.linspace(condition_data.min(), condition_data.max(), 200)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde(x_range),
            mode='lines',
            name=f'{condition} (KDE)',
            line=dict(width=3)
        ))

fig.update_layout(
    title='Intensity Distribution with KDE Overlay',
    xaxis_title='Log2(Intensity + 1)',
    yaxis_title='Density',
    barmode='overlay',
    height=600,
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

# Distribution statistics
st.markdown("**Distribution Statistics by Condition**")
dist_stats = intensity_df.groupby('Condition')['Log2_Intensity'].agg(['mean', 'median', 'std', 'min', 'max'])
dist_stats.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
st.dataframe(dist_stats.round(2), use_container_width=True)

st.markdown("---")

# ============================================================================
# 3. PCA ANALYSIS - THREE DATASETS
# ============================================================================

st.subheader("3️⃣ Principal Component Analysis (PCA)")

# Identify most common species
species_counts_total = df[species_col].value_counts()
most_common_species = species_counts_total.index[0]

st.info(f"🔬 **Most Common Proteome**: {most_common_species} ({species_counts_total[most_common_species]:,} proteins)")

# Helper function for PCA
def perform_pca(data_subset, title_suffix):
    """Perform PCA and return results"""
    # Transpose for PCA (samples as rows)
    df_pca = data_subset[numeric_cols].T
    
    # Standardize data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_pca)
    
    # Perform PCA
    pca = PCA(n_components=min(10, len(numeric_cols), len(data_subset)))
    pca_result = pca.fit_transform(df_scaled)
    
    # Create PCA dataframe
    pca_df = pd.DataFrame(
        pca_result[:, :3],
        columns=['PC1', 'PC2', 'PC3'],
        index=numeric_cols
    )
    pca_df['Sample'] = numeric_cols
    pca_df['Condition'] = pca_df['Sample'].map(sample_to_condition)
    
    variance_explained = pca.explained_variance_ratio_ * 100
    
    return pca_df, variance_explained, df_scaled

# Perform PCA on three datasets
st.markdown("### PCA on Three Protein Subsets")

# 1. All proteins
df_all = df.copy()
pca_all, var_all, scaled_all = perform_pca(df_all, "All Proteins")

# 2. Most common species only
df_common = df[df[species_col] == most_common_species].copy()
pca_common, var_common, scaled_common = perform_pca(df_common, most_common_species)

# 3. All except most common (rest)
df_rest = df[df[species_col] != most_common_species].copy()
pca_rest, var_rest, scaled_rest = perform_pca(df_rest, "Other Species")

# Create three PCA plots side by side
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**All Proteins** (n={len(df_all):,})")
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
**PERMANOVA** (Permutational Multivariate Analysis of Variance) tests whether groups have different centroids.
- **Null hypothesis**: No difference between condition centroids
- **p < 0.05**: Significant separation between conditions
""")

# Helper function for PERMANOVA
def run_permanova(distance_matrix, grouping, title):
    """Run PERMANOVA test"""
    try:
        # Create distance matrix
        from skbio import DistanceMatrix
        dm = DistanceMatrix(distance_matrix, ids=[str(i) for i in range(len(distance_matrix))])
        
        # Run PERMANOVA
        result = permanova(dm, grouping, permutations=999)
        
        return {
            'Dataset': title,
            'F-statistic': f"{result['test statistic']:.4f}",
            'p-value': f"{result['p-value']:.4f}",
            'Significant': '✅ Yes' if result['p-value'] < 0.05 else '❌ No',
            'Interpretation': 'Significant separation between conditions' if result['p-value'] < 0.05 else 'No significant separation'
        }
    except Exception as e:
        return {
            'Dataset': title,
            'F-statistic': 'Error',
            'p-value': 'Error',
            'Significant': 'N/A',
            'Interpretation': str(e)
        }

# Calculate distance matrices (Euclidean)
dist_all = squareform(pdist(scaled_all, metric='euclidean'))
dist_common = squareform(pdist(scaled_common, metric='euclidean'))
dist_rest = squareform(pdist(scaled_rest, metric='euclidean'))

# Grouping variable
grouping = [sample_to_condition[s] for s in numeric_cols]

# Run PERMANOVA on all three datasets
permanova_results = []
permanova_results.append(run_permanova(dist_all, grouping, f"All Proteins (n={len(df_all):,})"))
permanova_results.append(run_permanova(dist_common, grouping, f"{most_common_species} Only (n={len(df_common):,})"))
permanova_results.append(run_permanova(dist_rest, grouping, f"Other Species (n={len(df_rest):,})"))

permanova_df = pd.DataFrame(permanova_results)
st.dataframe(permanova_df, hide_index=True, use_container_width=True)

# Interpretation statement
st.markdown("### 📊 Statistical Interpretation")

significant_count = sum(1 for r in permanova_results if '✅' in r['Significant'])

if significant_count == 3:
    st.success("""
    **Strong Evidence of Condition Separation**: All three protein subsets show statistically significant 
    separation between experimental conditions (PERMANOVA p < 0.05). This indicates that:
    - Biological differences between conditions are captured across the entire proteome
    - Both major ({}) and minor species contribute to group differences
    - The data quality is sufficient for downstream differential expression analysis
    """.format(most_common_species))
elif significant_count == 2:
    st.info("""
    **Moderate Evidence of Condition Separation**: Two out of three protein subsets show significant 
    separation. This suggests:
    - Major biological differences exist, but may be driven by specific protein subsets
    - Species-specific effects may be present
    - Proceed with caution and consider species-stratified analysis
    """)
elif significant_count == 1:
    st.warning("""
    **Weak Evidence of Condition Separation**: Only one protein subset shows significant separation. 
    Consider:
    - Increasing sample size or biological replicates
    - Checking for batch effects or technical confounders
    - Re-evaluating experimental design and sample quality
    """)
else:
    st.error("""
    **No Significant Condition Separation**: None of the protein subsets show statistically significant 
    differences. This may indicate:
    - Insufficient biological differences between conditions
    - High technical noise obscuring biological signal
    - Potential issues with experimental design or sample preparation
    - Consider additional QC steps before proceeding with differential analysis
    """)

st.markdown("---")

# ============================================================================
# 5. HIERARCHICAL CLUSTERING HEATMAP WITH DENDROGRAM
# ============================================================================

st.subheader("5️⃣ Hierarchical Clustering Heatmap")

st.markdown("**Sample-to-sample correlation with hierarchical clustering**")

# Calculate correlation matrix
corr_matrix = df[numeric_cols].corr()

# Calculate linkage for both rows and columns
linkage_samples = linkage(corr_matrix, method='ward')

# Create dendrogram
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
dend = scipy_dendrogram(linkage_samples, labels=numeric_cols, no_plot=True)

# Reorder correlation matrix based on dendrogram
reordered_idx = dend['leaves']
corr_reordered = corr_matrix.iloc[reordered_idx, reordered_idx]

# Create heatmap with plotly
fig = go.Figure(data=go.Heatmap(
    z=corr_reordered.values,
    x=corr_reordered.columns,
    y=corr_reordered.columns,
    colorscale='RdBu_r',
    zmid=0,
    zmin=-1,
    zmax=1,
    colorbar=dict(title="Correlation")
))

fig.update_layout(
    title='Hierarchical Clustering Heatmap (Ward Linkage)',
    xaxis_title='Sample',
    yaxis_title='Sample',
    height=800,
    xaxis={'side': 'bottom', 'tickangle': -45},
    yaxis={'side': 'left'}
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================================================
# 6. DATA QUALITY SUMMARY
# ============================================================================

st.subheader("6️⃣ Data Quality Summary")

quality_metrics = {
    'Metric': [
        'Total Proteins',
        'Total Samples',
        'Conditions',
        'Most Common Species',
        'Imputation Method',
        'Mean Sample Correlation',
        'All Proteins - PC1 Variance (%)',
        f'{most_common_species} - PC1 Variance (%)',
        'Other Species - PC1 Variance (%)',
        'PERMANOVA Significant Datasets'
    ],
    'Value': [
        f"{len(df):,}",
        f"{len(numeric_cols)}",
        f"{len(conditions)} ({', '.join(conditions)})",
        f"{most_common_species} ({species_counts_total[most_common_species]:,} proteins)",
        imputation_method,
        f"{corr_reordered.values[np.triu_indices_from(corr_reordered.values, k=1)].mean():.3f}",
        f"{var_all[0]:.1f}",
        f"{var_common[0]:.1f}",
        f"{var_rest[0]:.1f}",
        f"{significant_count}/3"
    ]
}

quality_df = pd.DataFrame(quality_metrics)
st.dataframe(quality_df, hide_index=True, use_container_width=True)

st.markdown("---")

# ============================================================================
# EXPORT OPTIONS
# ============================================================================

st.subheader("💾 Export Data")

col1, col2, col3 = st.columns(3)

with col1:
    # Export imputed data
    csv = df[numeric_cols].to_csv(index=False)
    st.download_button(
        label="📥 Download Imputed Data",
        data=csv,
        file_name="imputed_proteomics_data.csv",
        mime="text/csv"
    )

with col2:
    # Export PCA results (all proteins)
    pca_csv = pca_all.to_csv(index=False)
    st.download_button(
        label="📥 Download PCA Results",
        data=pca_csv,
        file_name="pca_all_proteins.csv",
        mime="text/csv"
    )

with col3:
    # Export PERMANOVA results
    permanova_csv = permanova_df.to_csv(index=False)
    st.download_button(
        label="📥 Download PERMANOVA Results",
        data=permanova_csv,
        file_name="permanova_results.csv",
        mime="text/csv"
    )

st.markdown("---")
st.success("✅ EDA Complete! Data shows clear biological structure and is ready for differential expression analysis.")
