"""
pages/2_Visual_EDA.py - VISUAL EXPLORATORY DATA ANALYSIS
Violin plots showing log2 intensity distribution per sample with normality testing
"""
"""
pages/2_Visual_EDA.py - VISUAL EXPLORATORY DATA ANALYSIS
Enhanced normality diagnostics: Shapiro, D'Agostino, Skewness, Kurtosis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import warnings

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
# RESET FUNCTIONS
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

# ============================================================================
# NORMALITY DIAGNOSTICS
# ============================================================================

def normality_diagnostics(values: pd.Series) -> dict:
    """
    Comprehensive normality testing.
    
    Returns dict with:
    - n: sample size
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
            "shapiro_p": np.nan,
            "dagostino_p": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "is_normal": False
        }
    
    # Shapiro-Wilk test
    try:
        _, shapiro_p = stats.shapiro(values)
    except:
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
        "shapiro_p": shapiro_p,
        "dagostino_p": dagostino_p,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "is_normal": is_normal
    }

# ============================================================================
# HEADER WITH RESET BUTTONS
# ============================================================================

st.title("📊 Visual Exploratory Data Analysis")

col1, col2, col3 = st.columns([3, 1, 1])
with col2:
    if st.button("🔄 Reset Page", help="Clear this page's cache and restart"):
        reset_current_page()
with col3:
    if st.button("🗑️ Reset All", help="Clear everything and start over", type="secondary"):
        reset_all()

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
df_long['Condition'] = df_long['Sample'].str.extract(r'^([A-Z]+)', expand=False)

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
# ENHANCED NORMALITY STATISTICS
# ============================================================================

st.subheader("📈 Sample Statistics with Normality Diagnostics")

results = []
for col in numeric_cols:
    log2_values = np.log2(df_raw[col])
    log2_values = log2_values[np.isfinite(log2_values)]
    
    diag = normality_diagnostics(log2_values)
    
    results.append({
        "Sample": col,
        "n": diag["n"],
        "Mean (Log2)": log2_values.mean(),
        "Std (Log2)": log2_values.std(),
        "Shapiro p": diag["shapiro_p"],
        "DAgostino p": diag["dagostino_p"],
        "Skewness": diag["skewness"],
        "Kurtosis": diag["kurtosis"],
        "Normal?": "✓" if diag["is_normal"] else "✗"
    })

results_df = pd.DataFrame(results).round(4)

st.dataframe(results_df, use_container_width=True, hide_index=True)

# Summary
normal_count = (results_df['Normal?'] == '✓').sum()
st.info(f"🔬 **Normality Summary:** {normal_count}/{len(results_df)} samples pass normality tests (Shapiro & D'Agostino p > 0.05)")

st.markdown("---")
st.caption("💡 **Normality Tests:** Shapiro-Wilk (all n), D'Agostino-Pearson (n≥8) | **Thresholds:** p > 0.05 = normal")
st.caption("💡 **Skewness:** 0 = symmetric, >0 = right-skewed, <0 = left-skewed")
st.caption("💡 **Kurtosis:** 0 = normal, >0 = heavy-tailed, <0 = light-tailed")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Visual EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Visual Exploratory Data Analysis")
st.markdown("Log2 intensity distribution by sample with normality assessment")

# ============================================================================
# DATA VALIDATION
# ============================================================================

if 'data_ready' not in st.session_state or not st.session_state.data_ready:
    st.warning("⚠️ No data loaded. Please upload data on the **📁 Data Upload** page first.")
    st.stop()

# Load from session state
df_raw = st.session_state.df_raw
numeric_cols = st.session_state.numeric_cols
id_col = st.session_state.id_col
species_col = st.session_state.species_col
data_type = st.session_state.data_type

st.success(f"✅ Loaded {data_type} data: {len(df_raw):,} rows × {len(numeric_cols)} samples")
st.markdown("---")

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Long format for plotting
df_long = df_raw.melt(
    id_vars=[id_col, species_col],
    value_vars=numeric_cols,
    var_name='Sample',
    value_name='Intensity'
)

# Log2 transformation
df_long['Log2_Intensity'] = np.log2(df_long['Intensity'])

# Extract condition from sample name (first letter)
df_long['Condition'] = df_long['Sample'].str[0]

# ============================================================================
# VIOLIN PLOT
# ============================================================================

st.header("🎻 Log2 Intensity Distribution by Sample")
st.caption("Log2-transformed intensity values, colored by condition")

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

st.plotly_chart(fig_violin, width="stretch")

# ============================================================================
# SAMPLE STATISTICS WITH NORMALITY TEST
# ============================================================================

st.subheader("📈 Sample Statistics (Log2 Scale)")

# Calculate statistics for each sample
stats_list = []

for col in numeric_cols:
    log2_values = np.log2(df_raw[col].values)
    
    # Remove inf/-inf values for normality test
    valid_log2 = log2_values[np.isfinite(log2_values)]
    
    # Shapiro-Wilk test
    if len(valid_log2) > 3:
        shapiro_stat, shapiro_p = stats.shapiro(valid_log2)
    else:
        shapiro_p = np.nan
    
    # Normality classification
    if pd.isna(shapiro_p):
        normality = "N/A"
    elif shapiro_p > 0.98:
        normality = "Normal"
    elif shapiro_p > 0.95:
        normality = "Quasi-Normal"
    else:
        normality = "Not Normal"
    
    stats_list.append({
        'Sample': col,
        'Condition': col[0],
        'N': df_raw[col].notna().sum(),
        'N (1.00)': (df_raw[col] == 1.00).sum(),
        'Mean (Log2)': np.mean(valid_log2),
        'Median (Log2)': np.median(valid_log2),
        'Std (Log2)': np.std(valid_log2),
        'Min (Log2)': np.min(valid_log2),
        'Max (Log2)': np.max(valid_log2),
        'Shapiro p-value': shapiro_p,
        'Normality': normality
    })

stats_df = pd.DataFrame(stats_list).round(4)

st.dataframe(stats_df, width="stretch", hide_index=True)

st.markdown("---")
st.caption("💡 **Normality:** Normal (p > 0.98) | Quasi-Normal (p > 0.95) | Not Normal (p ≤ 0.95)")
st.caption("💡 **Interactive:** Hover for details, click legend to toggle conditions, download as PNG using camera icon")
