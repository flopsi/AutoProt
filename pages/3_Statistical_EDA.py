# pages/3_Statistical_EDA.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Statistical EDA", layout="wide")

st.title("📈 Statistical EDA: Species & Variability")

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
protein_data = st.session_state.get("protein_data")
if protein_data is None:
    st.error("No data found. Please complete the EDA page first.")
    st.stop()

df: pd.DataFrame = protein_data.raw
numeric_cols = protein_data.numeric_cols
species_col = protein_data.species_col
species_mapping = protein_data.species_mapping

if not numeric_cols or species_col is None or not species_mapping:
    st.error("Missing species information or numeric columns.")
    st.stop()

st.success(
    f"{len(df):,} proteins × {len(numeric_cols)} samples | "
    f"Transform: **{st.session_state.get('selected_transform_method', 'raw')}**"
)

# ----------------------------------------------------------------------
# 1) Configuration: Assign samples to conditions
# ----------------------------------------------------------------------
st.subheader("1️⃣ Configuration")

n_cols = len(numeric_cols)
mid = n_cols // 2

col1, col2 = st.columns(2)
with col1:
    cond_a_cols = st.multiselect(
        "Condition A samples",
        options=numeric_cols,
        default=numeric_cols[:mid],
        key="stat_eda_cond_a",
    )
with col2:
    cond_b_cols = st.multiselect(
        "Condition B samples",
        options=numeric_cols,
        default=numeric_cols[mid:],
        key="stat_eda_cond_b",
    )

if not cond_a_cols or not cond_b_cols:
    st.warning("Select at least one sample for each condition.")
    st.stop()

all_samples = cond_a_cols + cond_b_cols

# ----------------------------------------------------------------------
# 2) Species composition per sample (stacked bar chart)
# ----------------------------------------------------------------------
st.subheader("2️⃣ Species Composition per Sample")

# Count proteins per species per sample (presence/absence or intensity-weighted)
species_counts = []

for sample in all_samples:
    # For each sample, count how many proteins of each species are detected (non-zero/non-NaN)
    sample_data = df[[species_col, sample]].copy()
    sample_data = sample_data[sample_data[sample].notna() & (sample_data[sample] > 0)]
    
    counts = sample_data[species_col].value_counts().to_dict()
    
    species_counts.append({
        "Sample": sample,
        "HUMAN": counts.get("HUMAN", 0),
        "YEAST": counts.get("YEAST", 0),
        "ECOLI": counts.get("ECOLI", 0),
    })

species_df = pd.DataFrame(species_counts)

# Stacked bar chart
fig_species = go.Figure()

fig_species.add_trace(go.Bar(
    name='HUMAN',
    x=species_df['Sample'],
    y=species_df['HUMAN'],
    marker_color='#1f77b4',
))

fig_species.add_trace(go.Bar(
    name='YEAST',
    x=species_df['Sample'],
    y=species_df['YEAST'],
    marker_color='#ff7f0e',
))

fig_species.add_trace(go.Bar(
    name='ECOLI',
    x=species_df['Sample'],
    y=species_df['ECOLI'],
    marker_color='#2ca02c',
))

fig_species.update_layout(
    barmode='stack',
    title="Detected Proteins per Species per Sample",
    xaxis_title="Sample",
    yaxis_title="Number of Proteins",
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig_species, use_container_width=True)

# Summary table
st.dataframe(species_df.set_index("Sample"), use_container_width=True)

# ----------------------------------------------------------------------
# 3) Within-condition CV (Coefficient of Variation) violin plots
# ----------------------------------------------------------------------
st.subheader("3️⃣ Within-Condition Variability (CV)")

st.markdown(
    """
    **Coefficient of Variation (CV)** = (std / mean) × 100%  
    Lower CV indicates more consistent measurements within a condition.
    """
)

# Compute CV per protein per condition
def compute_cv(row, cols):
    vals = row[cols]
    vals = vals[vals > 0]  # exclude zeros/negatives
    if len(vals) < 2:
        return np.nan
    return (vals.std() / vals.mean()) * 100 if vals.mean() > 0 else np.nan

df['CV_CondA'] = df.apply(lambda row: compute_cv(row, cond_a_cols), axis=1)
df['CV_CondB'] = df.apply(lambda row: compute_cv(row, cond_b_cols), axis=1)

cv_a = df['CV_CondA'].dropna()
cv_b = df['CV_CondB'].dropna()

# Violin plot
fig_cv = go.Figure()

fig_cv.add_trace(go.Violin(
    y=cv_a,
    name='Condition A',
    box_visible=True,
    meanline_visible=True,
    fillcolor='#1f77b4',
    opacity=0.6,
    x0='Condition A',
))

fig_cv.add_trace(go.Violin(
    y=cv_b,
    name='Condition B',
    box_visible=True,
    meanline_visible=True,
    fillcolor='#ff7f0e',
    opacity=0.6,
    x0='Condition B',
))

fig_cv.update_layout(
    title="Coefficient of Variation (CV) Distribution per Condition",
    yaxis_title="CV (%)",
    xaxis_title="Condition",
    height=500,
    showlegend=False,
)

st.plotly_chart(fig_cv, use_container_width=True)

# Summary stats
col_stats1, col_stats2 = st.columns(2)

with col_stats1:
    st.metric("Condition A - Median CV", f"{cv_a.median():.1f}%")
    st.metric("Condition A - Mean CV", f"{cv_a.mean():.1f}%")

with col_stats2:
    st.metric("Condition B - Median CV", f"{cv_b.median():.1f}%")
    st.metric("Condition B - Mean CV", f"{cv_b.mean():.1f}%")

# Optional: CV per species
with st.expander("View CV by Species"):
    st.markdown("### CV Distribution by Species")
    
    for species in ["HUMAN", "YEAST", "ECOLI"]:
        species_proteins = df[df[species_col] == species]
        
        if len(species_proteins) == 0:
            continue
        
        cv_a_sp = species_proteins['CV_CondA'].dropna()
        cv_b_sp = species_proteins['CV_CondB'].dropna()
        
        st.markdown(f"#### {species}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Cond A - Median CV ({species})", f"{cv_a_sp.median():.1f}%" if len(cv_a_sp) > 0 else "N/A")
        with col2:
            st.metric(f"Cond B - Median CV ({species})", f"{cv_b_sp.median():.1f}%" if len(cv_b_sp) > 0 else "N/A")

# ----------------------------------------------------------------------
# 4) PCA Analysis: ALL | HUMAN only | YEAST+ECOLI
# ----------------------------------------------------------------------
st.subheader("4️⃣ PCA Analysis by Species Group")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f_oneway

st.markdown(
    """
    **PCA** visualizes sample similarity. Each column shows:
    - ALL proteins
    - HUMAN only
    - YEAST + ECOLI (spike-ins)
    
    **PERMANOVA** tests if conditions differ significantly in multivariate space.  
    **Cohen's d** measures effect size between conditions.
    """
)

# Helper: PERMANOVA (simplified)
def permanova_test(X: np.ndarray, labels: list, n_permutations: int = 999) -> dict:
    """
    Simplified PERMANOVA: tests if groups have different centroids in multivariate space.
    Returns p-value from permutation test.
    """
    from scipy.spatial.distance import pdist, squareform
    
    # Compute distance matrix (Euclidean)
    dist_matrix = squareform(pdist(X, metric='euclidean'))
    
    # Observed F-statistic
    unique_labels = np.unique(labels)
    n_groups = len(unique_labels)
    n = len(labels)
    
    # Between-group sum of squares
    ss_between = 0
    group_means = []
    for label in unique_labels:
        group_idx = np.where(labels == label)[0]
        group_center = X[group_idx].mean(axis=0)
        group_means.append(group_center)
        ss_between += len(group_idx) * np.sum((group_center - X.mean(axis=0))**2)
    
    # Within-group sum of squares
    ss_within = 0
    for label in unique_labels:
        group_idx = np.where(labels == label)[0]
        group_center = X[group_idx].mean(axis=0)
        ss_within += np.sum((X[group_idx] - group_center)**2)
    
    df_between = n_groups - 1
    df_within = n - n_groups
    
    if df_within == 0 or ss_within == 0:
        return {"F": np.nan, "p": np.nan}
    
    F_obs = (ss_between / df_between) / (ss_within / df_within)
    
    # Permutation test
    F_perm = []
    for _ in range(n_permutations):
        perm_labels = np.random.permutation(labels)
        
        ss_between_perm = 0
        for label in unique_labels:
            group_idx = np.where(perm_labels == label)[0]
            group_center = X[group_idx].mean(axis=0)
            ss_between_perm += len(group_idx) * np.sum((group_center - X.mean(axis=0))**2)
        
        ss_within_perm = 0
        for label in unique_labels:
            group_idx = np.where(perm_labels == label)[0]
            group_center = X[group_idx].mean(axis=0)
            ss_within_perm += np.sum((X[group_idx] - group_center)**2)
        
        if ss_within_perm > 0:
            F_perm.append((ss_between_perm / df_between) / (ss_within_perm / df_within))
    
    p_value = np.sum(np.array(F_perm) >= F_obs) / len(F_perm)
    
    return {"F": float(F_obs), "p": float(p_value)}


# Helper: Cohen's d
def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Cohen's d effect size between two groups.
    d = (mean1 - mean2) / pooled_std
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return np.nan
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


# Prepare data for PCA
def run_pca_analysis(df_subset: pd.DataFrame, sample_cols: list, title: str, color_map: dict):
    """Run PCA, plot, and return stats."""
    # Transpose: samples as rows, proteins as columns
    X = df_subset[sample_cols].T.values
    
    # Remove proteins with NaN or zero variance
    mask = np.all(np.isfinite(X), axis=0) & (np.var(X, axis=0) > 0)
    X = X[:, mask]
    
    if X.shape[1] < 2:
        return None, None, None
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Labels for samples
    labels = np.array(['A' if s in cond_a_cols else 'B' for s in sample_cols])
    
    # PERMANOVA
    perm_result = permanova_test(X_scaled, labels, n_permutations=999)
    
    # Cohen's d on PC1
    pc1_a = X_pca[labels == 'A', 0]
    pc1_b = X_pca[labels == 'B', 0]
    cohens = cohens_d(pc1_a, pc1_b)
    
    # Plot
    fig = go.Figure()
    
    for label, color in color_map.items():
        mask_label = labels == label
        fig.add_trace(go.Scatter(
            x=X_pca[mask_label, 0],
            y=X_pca[mask_label, 1],
            mode='markers+text',
            marker=dict(size=12, color=color, line=dict(width=1, color='white')),
            text=[sample_cols[i] for i in np.where(mask_label)[0]],
            textposition='top center',
            textfont=dict(size=9),
            name=f'Condition {label}',
        ))
    
    fig.update_layout(
        title=f"{title}<br><sub>PERMANOVA: F={perm_result['F']:.2f}, p={perm_result['p']:.3f} | Cohen's d={cohens:.2f}</sub>",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
        height=500,
        showlegend=True,
    )
    
    return fig, perm_result, cohens


# 3 PCA analyses
col_pca1, col_pca2, col_pca3 = st.columns(3)

color_map = {'A': '#1f77b4', 'B': '#ff7f0e'}

# ALL proteins
with col_pca1:
    st.markdown("#### ALL Proteins")
    df_all = df.copy()
    fig_all, perm_all, cohens_all = run_pca_analysis(df_all, all_samples, "ALL Proteins", color_map)
    if fig_all:
        st.plotly_chart(fig_all, use_container_width=True, key="pca_all")
        st.caption(f"PERMANOVA p={perm_all['p']:.3f}, Cohen's d={cohens_all:.2f}")
    else:
        st.warning("Insufficient data for PCA")

# HUMAN only
with col_pca2:
    st.markdown("#### HUMAN Only")
    df_human = df[df[species_col] == 'HUMAN'].copy()
    fig_human, perm_human, cohens_human = run_pca_analysis(df_human, all_samples, "HUMAN Only", color_map)
    if fig_human:
        st.plotly_chart(fig_human, use_container_width=True, key="pca_human")
        st.caption(f"PERMANOVA p={perm_human['p']:.3f}, Cohen's d={cohens_human:.2f}")
    else:
        st.warning("Insufficient HUMAN proteins")

# YEAST + ECOLI
with col_pca3:
    st.markdown("#### YEAST + ECOLI")
    df_spikes = df[df[species_col].isin(['YEAST', 'ECOLI'])].copy()
    fig_spikes, perm_spikes, cohens_spikes = run_pca_analysis(df_spikes, all_samples, "YEAST + ECOLI", color_map)
    if fig_spikes:
        st.plotly_chart(fig_spikes, use_container_width=True, key="pca_spikes")
        st.caption(f"PERMANOVA p={perm_spikes['p']:.3f}, Cohen's d={cohens_spikes:.2f}")
    else:
        st.warning("Insufficient spike-in proteins")

# Summary stats table
st.markdown("### PCA Summary Statistics")

pca_summary = pd.DataFrame([
    {
        "Group": "ALL Proteins",
        "PERMANOVA F": perm_all['F'] if fig_all else np.nan,
        "PERMANOVA p": perm_all['p'] if fig_all else np.nan,
        "Cohen's d": cohens_all if fig_all else np.nan,
    },
    {
        "Group": "HUMAN Only",
        "PERMANOVA F": perm_human['F'] if fig_human else np.nan,
        "PERMANOVA p": perm_human['p'] if fig_human else np.nan,
        "Cohen's d": cohens_human if fig_human else np.nan,
    },
    {
        "Group": "YEAST + ECOLI",
        "PERMANOVA F": perm_spikes['F'] if fig_spikes else np.nan,
        "PERMANOVA p": perm_spikes['p'] if fig_spikes else np.nan,
        "Cohen's d": cohens_spikes if fig_spikes else np.nan,
    },
])

st.dataframe(pca_summary.round(3), use_container_width=True)

st.markdown(
    """
    **Interpretation:**
    - **PERMANOVA p < 0.05**: Conditions are significantly different in multivariate space.
    - **Cohen's d**: Small (0.2), Medium (0.5), Large (0.8+) effect size.
    """
)

def compute_quality_metrics(row, cond_a_cols, cond_b_cols):
    """Compute multiple quality metrics for filtering."""
    vals_a = row[cond_a_cols]
    vals_b = row[cond_b_cols]
    
    # 1. Missing value rate
    missing_rate_a = vals_a.isna().sum() / len(vals_a)
    missing_rate_b = vals_b.isna().sum() / len(vals_b)
    missing_rate_total = (missing_rate_a + missing_rate_b) / 2
    
    # 2. Detection rate (non-zero, non-NaN)
    detected_a = ((vals_a > 0) & vals_a.notna()).sum() / len(vals_a)
    detected_b = ((vals_b > 0) & vals_b.notna()).sum() / len(vals_b)
    detection_rate = (detected_a + detected_b) / 2
    
    # 3. Mean intensity (log scale if transformed)
    mean_intensity_a = vals_a[vals_a > 0].mean()
    mean_intensity_b = vals_b[vals_b > 0].mean()
    mean_intensity = np.nanmean([mean_intensity_a, mean_intensity_b])
    
    # 4. Intensity range (dynamic range)
    all_vals = pd.concat([vals_a, vals_b])
    all_vals = all_vals[all_vals > 0]
    if len(all_vals) > 0:
        intensity_range = all_vals.max() - all_vals.min()
    else:
        intensity_range = 0
    
    return {
        'missing_rate': missing_rate_total,
        'detection_rate': detection_rate,
        'mean_intensity': mean_intensity,
        'intensity_range': intensity_range,
    }
