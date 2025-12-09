"""
pages/2_Visual_EDA.py - VISUAL EXPLORATORY DATA ANALYSIS
Enhanced normality diagnostics with auto-generated comments
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import warnings
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from helpers.diagnostics import generate_full_comment, generate_condition_summary

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Visual EDA",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# NORMALITY DIAGNOSTICS
# ============================================================================

def normality_diagnostics(values: pd.Series) -> dict:
    """
    Comprehensive normality testing.
    
    Returns dict with:
    - n: sample size
    - shapiro_w: Shapiro-Wilk W statistic
    - shapiro_p: Shapiro-Wilk p-value
    - dagostino_p: D'Agostino K^2 p-value
    - skewness: skewness coefficient
    - kurtosis: excess kurtosis
    - is_normal: True if both tests p > 0.05
    """
    values = values.dropna()
    n = len(values)
    
    if n < 3:
        return {
            "n": n,
            "shapiro_w": np.nan,
            "shapiro_p": np.nan,
            "dagostino_p": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "is_normal": False
        }
    
    # Shapiro-Wilk test
    try:
        shapiro_w, shapiro_p = stats.shapiro(values)
    except:
        shapiro_w = np.nan
        shapiro_p = np.nan
    
    # D'Agostino-Pearson K^2 test (requires n >= 8)
    try:
        if n >= 8:
            _, dagostino_p = stats.normaltest(values)
        else:
            dagostino_p = np.nan
    except:
        dagostino_p = np.nan
    
    # Skewness and kurtosis
    try:
        skewness = stats.skew(values)
        kurtosis = stats.kurtosis(values)  # Excess kurtosis
    except:
        skewness = np.nan
        kurtosis = np.nan
    
    # Combined normality decision
    is_normal = False
    if not np.isnan(shapiro_p) and not np.isnan(dagostino_p):
        is_normal = (shapiro_p > 0.05) and (dagostino_p > 0.05)
    elif not np.isnan(shapiro_p):
        is_normal = shapiro_p > 0.05
    
    return {
        "n": n,
        "shapiro_w": shapiro_w,
        "shapiro_p": shapiro_p,
        "dagostino_p": dagostino_p,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "is_normal": is_normal
    }

# ============================================================================
# HEADER
# ============================================================================

st.title("📊 Visual Exploratory Data Analysis")
st.markdown("Log2 intensity distribution by sample")
st.markdown("---")

# ============================================================================
# DATA VALIDATION
# ============================================================================

if 'data_ready' not in st.session_state or not st.session_state.data_ready:
    st.warning("⚠️ No data loaded. Please upload data on the **📁 Data Upload** page first.")
    st.stop()

df_raw = st.session_state.df_raw
numeric_cols = st.session_state.numeric_cols
id_col = st.session_state.id_col
species_col = st.session_state.species_col
data_type = st.session_state.data_type
replicates_per_condition = st.session_state.replicates_per_condition

st.success(f"✅ Loaded {data_type} data: {len(df_raw):,} rows × {len(numeric_cols)} samples")
st.markdown("---")

# ============================================================================
# DATA PREPARATION
# ============================================================================

df_long = df_raw.melt(
    id_vars=[id_col, species_col],
    value_vars=numeric_cols,
    var_name='Sample',
    value_name='Intensity'
)

df_long['Log2_Intensity'] = np.log2(df_long['Intensity'])

# REPLACE WITH:

# Use pre-computed conditions if available, otherwise extract
if 'sample_to_condition' in st.session_state:
    df_long['Condition'] = df_long['Sample'].map(st.session_state.sample_to_condition)
else:
    # Fallback: extract from sample name
    import re
    df_long['Condition'] = df_long['Sample'].apply(
        lambda x: re.search(r'^([A-Z]+)', str(x)).group(1) if re.search(r'^([A-Z]+)', str(x)) else "Unknown"
    )


# ============================================================================
# VIOLIN PLOT
# ============================================================================

st.header("🎻 Log2 Intensity Distribution by Sample")
st.caption("All proteins per sample, colored by condition")

fig_violin = px.violin(
    df_long,
    x='Sample',
    y='Log2_Intensity',
    color='Condition',
    title='Log2 Intensity Distribution by Sample',
    labels={
        'Log2_Intensity': 'Log2(Intensity)',
        'Sample': 'Sample',
        'Condition': 'Condition'
    },
    box=True,
    points=False,
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig_violin.update_layout(
    height=600,
    hovermode='closest',
    xaxis_tickangle=-45,
    template='plotly_white',
    showlegend=True
)

st.plotly_chart(fig_violin, use_container_width=True)

# ============================================================================
# ENHANCED NORMALITY STATISTICS WITH SHAPIRO W AND LEVENE'S F
# ============================================================================

st.subheader("📈 Sample Statistics with Normality Diagnostics & Comments")

# Prepare data for F-test by condition
log2_df = df_raw[numeric_cols].apply(lambda x: np.log2(x))
log2_df = log2_df.replace([np.inf, -np.inf], np.nan)

# Group samples by condition for F-test
condition_groups = {}
for col in numeric_cols:
    condition = col.split('_')[0] if '_' in col else col[0]
    if condition not in condition_groups:
        condition_groups[condition] = []
    condition_groups[condition].append(col)

results = []
for col in numeric_cols:
    log2_values = log2_df[col].dropna()
    
    diag = normality_diagnostics(log2_values)
    
    # Extract condition for this sample
    condition = col.split('_')[0] if '_' in col else col[0]
    
    # Calculate Levene's F-statistic
    f_stat = np.nan
    f_pvalue = np.nan
    
    if condition in condition_groups and len(condition_groups[condition]) > 1:
        condition_samples = condition_groups[condition]
        if len(condition_samples) >= 2:
            condition_data = [log2_df[s].dropna() for s in condition_samples]
            try:
                f_stat, f_pvalue = stats.levene(*condition_data)
            except:
                f_stat = np.nan
                f_pvalue = np.nan
    
    # Add diagnostic info
    diag['levene_p'] = f_pvalue
    
    results.append({
        "Sample": col,
        "Condition": condition,
        "n": diag["n"],
        "Mean (Log2)": log2_values.mean(),
        "Std (Log2)": log2_values.std(),
        "Shapiro W": diag["shapiro_w"],
        "Shapiro p": diag["shapiro_p"],
        "DAgostino p": diag["dagostino_p"],
        "Skewness": diag["skewness"],
        "Kurtosis": diag["kurtosis"],
        "Levene F": f_stat,
        "Levene p": f_pvalue,
        "Normal?": "✓" if diag["is_normal"] else "✗",
        "_diag_dict": diag  # Store for comment generation
    })

results_df = pd.DataFrame(results).round(4)

# Display main table
display_df = results_df.drop('_diag_dict', axis=1)
st.dataframe(display_df, use_container_width=True, hide_index=True)

# ============================================================================
# AUTO-GENERATED COMMENTS
# ============================================================================

st.subheader("💡 Intelligent Statistical Comments")

# Create tabs for sample comments
tab1, tab2 = st.tabs(["Per-Sample Analysis", "Condition Summary"])

with tab1:
    st.markdown("**Individual sample diagnostics and recommendations:**")
    
    for idx, row in results_df.iterrows():
        with st.expander(f"🔍 {row['Sample']} (Condition: {row['Condition']})"):
            diag_dict = row['_diag_dict']
            comments = generate_full_comment(diag_dict, sample_size=int(row['n']))
            
            # Display comments
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Diagnostic:**")
                st.write(comments['diagnostic'])
                
                st.markdown("**📈 Distribution:**")
                st.write(comments['distribution'])
            
            with col2:
                st.markdown("**⚖️ Variance:**")
                st.write(comments['variance'])
            
            st.markdown("**✅ Recommendations:**")
            for rec in comments['recommendations']:
                st.write(rec)

with tab2:
    st.markdown("**Condition-level summary and quality assessment:**")
    
    conditions = results_df['Condition'].unique()
    
    for condition in sorted(conditions):
        cond_df = results_df[results_df['Condition'] == condition]
        summary = generate_condition_summary(cond_df)
        
        with st.expander(f"🏷️ Condition: {condition}"):
            # Overall assessment
            status_emoji = "✓" if "READY" in summary['recommendation'] else "⚠️" if "MIXED" in summary['recommendation'] else "❌"
            st.markdown(f"### {status_emoji} {summary['recommendation']}")
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Normal Samples", f"{summary['normal_pct']:.0f}%")
            col2.metric("Homogeneous Variance", f"{summary['homogeneous_pct']:.0f}%")
            col3.metric("Mean Skewness", f"{summary['mean_skewness']:.2f}")
            
            st.markdown(summary['summary'])
            
            # Detailed breakdown
            st.markdown("**Sample Breakdown:**")
            cond_display = cond_df[['Sample', 'n', 'Mean (Log2)', 'Shapiro p', 'Levene p', 'Normal?']].copy()
            st.dataframe(cond_display, hide_index=True)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

st.markdown("---")

normal_count = (results_df['Normal?'] == '✓').sum()
homogeneous_count = (results_df['Levene p'] > 0.05).sum()

col1, col2 = st.columns(2)
with col1:
    st.info(f"🔬 **Normality:** {normal_count}/{len(results_df)} samples pass (Shapiro & D'Agostino p > 0.05)")
with col2:
    st.info(f"📊 **Homogeneity:** {homogeneous_count}/{len(results_df)} samples pass variance test (Levene p > 0.05)")

st.markdown("---")
st.caption("💡 **Shapiro W:** Test statistic for normality (closer to 1 = more normal)")
st.caption("💡 **Normality Tests:** Shapiro-Wilk (all n), D'Agostino-Pearson (n≥8) | **Thresholds:** p > 0.05 = normal")
st.caption("💡 **Levene F:** Test statistic for equal variances within condition | p > 0.05 = homogeneous variance")
st.caption("💡 **Skewness:** 0 = symmetric, >0 = right-skewed, <0 = left-skewed | **Kurtosis:** 0 = normal, >0 = heavy-tailed, <0 = light-tailed")

st.markdown("---")
st.markdown("---")

# ============================================================================
# RESET BUTTONS (BOTTOM)
# ============================================================================

def reset_all():
    """Clear all session state and restart from upload page"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.switch_page("pages/1_Data_Upload.py")

def reset_current_page():
    """Clear only current page's cached data"""
    st.cache_data.clear()
    st.rerun()

col1, col2, col3 = st.columns([3, 1, 1])
with col2:
    if st.button("🔄 Reset Page", help="Clear this page's cache and restart", key="reset_page_bottom"):
        reset_current_page()
with col3:
    if st.button("🗑️ Reset All", help="Clear everything and start over", type="secondary", key="reset_all_bottom"):
        reset_all()
