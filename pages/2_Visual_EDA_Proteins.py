"""
pages/2_Visual_EDA_Proteins.py - PRODUCTION VERSION
Exploratory Data Analysis with advanced visualization and analysis functions
Uses plotnine (ggplot2-style) and Polars for high-performance data processing
"""

import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import gc
from plotnine import *

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Visual EDA - Proteins",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Visual EDA - Proteins")
st.markdown("Exploratory Data Analysis: Distributions, Quality, and Patterns")

# ============================================================================
# DATA VALIDATION
# ============================================================================

if 'data_ready' not in st.session_state or not st.session_state.data_ready:
    st.error("📥 Please upload protein data first on the **Data Upload** page")
    st.stop()

if st.session_state.data_type != 'protein':
    st.error("⚠️ This page is for protein data. Please upload protein data on the **Data Upload** page")
    st.stop()

# Get data from session state
df_raw = st.session_state.df_raw
numeric_cols = st.session_state.numeric_cols
id_col = st.session_state.id_col
species_col = st.session_state.species_col
data_type = 'Protein'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clear_plot_memory():
    """Close all matplotlib figures and collect garbage."""
    plt.close('all')
    gc.collect()

# ============================================================================
# CACHE FUNCTIONS FOR HIGH-PERFORMANCE ANALYSIS
# ============================================================================

@st.cache_data
def compute_log2(df_dict: dict, cols: list) -> dict:
    """Cache log2 transformation."""
    df_temp = pl.from_dict(df_dict)
    df_log2 = df_temp.with_columns([
        pl.col(c).clip(lower_bound=1.0).log(2).alias(c) for c in cols
    ])
    return df_log2.to_dict(as_series=False)

@st.cache_data
def compute_valid_counts(df_dict: dict, id_col: str, species_col: str, numeric_cols: list) -> dict:
    """Cache valid protein counts per species per sample."""
    df_temp = pl.from_dict(df_dict)
    
    df_counts = df_temp.select([id_col, species_col] + numeric_cols).unpivot(
        index=[id_col, species_col],
        on=numeric_cols,
        variable_name='sample',
        value_name='value'
    ).filter(
        (pl.col('value') > 1.0) & (pl.col('value').is_finite())
    ).group_by(['sample', species_col]).agg(
        pl.len().alias('count')
    )
    
    # Calculate species order by total
    species_totals = df_counts.group_by(species_col).agg(
        pl.col('count').sum().alias('total')
    ).sort('total', descending=True)
    
    # Join to add sort key, then sort
    df_counts = df_counts.join(
        species_totals.with_row_index('sort_order'),
        on=species_col
    ).sort(['sample', 'sort_order']).with_columns([
        (pl.col('count').cum_sum().over('sample') - pl.col('count') / 2).alias('label_pos')
    ]).drop(['total', 'sort_order'])
    
    return df_counts.to_dict(as_series=False)

@st.cache_data
def compute_valid_table(df_dict: dict, id_col: str, species_col: str, numeric_cols: list) -> dict:
    """Cache valid protein table."""
    df_temp = pl.from_dict(df_dict)
    
    df_counts = df_temp.select([id_col, species_col] + numeric_cols).unpivot(
        index=[id_col, species_col],
        on=numeric_cols,
        variable_name='sample',
        value_name='value'
    ).filter(
        (pl.col('value') > 1.0) & (pl.col('value').is_finite())
    ).group_by(['sample', species_col]).agg(
        pl.len().alias('count')
    )
    
    df_table = df_counts.pivot(
        index=species_col,
        on='sample',
        values='count'
    ).fill_null(0)
    
    species_totals = df_temp.select([id_col, species_col] + numeric_cols).unpivot(
        index=[id_col, species_col],
        on=numeric_cols,
        variable_name='sample',
        value_name='value'
    ).filter(
        (pl.col('value') > 1.0) & (pl.col('value').is_finite())
    ).group_by([id_col, species_col]).agg(
        pl.len()
    ).group_by(species_col).agg(
        pl.len().alias('Total')
    )
    
    df_table = df_table.join(species_totals, on=species_col).sort('Total', descending=True)
    
    return df_table.to_dict(as_series=False)

@st.cache_data
def compute_intensity_stats(df_log2_dict: dict, id_col: str, numeric_cols: list) -> dict:
    """Cache intensity statistics."""
    df_log2 = pl.from_dict(df_log2_dict)
    
    df_long = df_log2.select([id_col] + numeric_cols).unpivot(
        index=[id_col],
        on=numeric_cols,
        variable_name='sample',
        value_name='log2_intensity'
    ).filter(
        pl.col('log2_intensity').is_finite()
    ).with_columns(
        pl.when(pl.col('sample').str.starts_with('A'))
        .then(pl.lit('A'))
        .otherwise(pl.lit('B'))
        .alias('condition')
    )
    
    df_stats = df_long.group_by('sample').agg([
        pl.col('log2_intensity').len().alias('n'),
        pl.col('log2_intensity').mean().alias('mean'),
        pl.col('log2_intensity').median().alias('median'),
        pl.col('log2_intensity').std().alias('std'),
        pl.col('log2_intensity').quantile(0.25).alias('q25'),
        pl.col('log2_intensity').quantile(0.75).alias('q75')
    ]).sort('sample')
    
    return {
        'long_data': df_long.to_dict(as_series=False),
        'stats': df_stats.to_dict(as_series=False)
    }

@st.cache_data
def compute_cv_data(df_dict: dict, id_col: str, numeric_cols: list) -> dict:
    """Cache CV calculations."""
    df_temp = pl.from_dict(df_dict)
    
    conditions = {}
    for col in numeric_cols:
        condition = col[0]
        if condition not in conditions:
            conditions[condition] = []
        conditions[condition].append(col)
    
    cv_data = []
    cv_threshold_data = []
    
    for condition, cols in conditions.items():
        df_cv = df_temp.select([id_col] + cols).with_columns([
            pl.concat_list(cols).list.mean().alias('mean'),
            pl.concat_list(cols).list.std().alias('std')
        ]).with_columns(
            (pl.col('std') / pl.col('mean') * 100).alias('cv')
        ).filter(
            pl.col('cv').is_finite() & (pl.col('cv') > 0)
        )
        
        # For violin plot
        for row in df_cv.select([id_col, 'cv']).iter_rows(named=True):
            cv_data.append({
                'id': row[id_col],
                'condition': condition,
                'cv': row['cv']
            })
        
        # For threshold plot
        total = df_cv.shape[0]
        cv_under_20 = df_cv.filter(pl.col('cv') < 20).shape[0]
        cv_under_10 = df_cv.filter(pl.col('cv') < 10).shape[0]
        
        cv_threshold_data.append({'condition': condition, 'threshold': 'Total', 'count': total})
        cv_threshold_data.append({'condition': condition, 'threshold': 'CV < 20%', 'count': cv_under_20})
        cv_threshold_data.append({'condition': condition, 'threshold': 'CV < 10%', 'count': cv_under_10})
    
    return {
        'cv_data': cv_data,
        'cv_threshold_data': cv_threshold_data,
        'conditions': conditions
    }

@st.cache_data
def compute_missing_data(df_dict: dict, id_col: str, numeric_cols: list) -> dict:
    """Cache missing value calculations."""
    df_temp = pl.from_dict(df_dict)
    
    conditions = {}
    for col in numeric_cols:
        condition = col[0]
        if condition not in conditions:
            conditions[condition] = []
        conditions[condition].append(col)
    
    missing_plot_data = []
    
    for condition, cols in conditions.items():
        df_missing = df_temp.select([id_col] + cols).with_columns([
            pl.sum_horizontal([
                (pl.col(c) <= 1.0) | (pl.col(c).is_null()) for c in cols
            ]).alias('n_missing')
        ])
        
        for n_miss in range(len(cols) + 1):
            count = df_missing.filter(pl.col('n_missing') == n_miss).shape[0]
            missing_plot_data.append({
                'condition': condition,
                'n_missing': f'{n_miss} missing',
                'count': count
            })
    
    return missing_plot_data

# ============================================================================
# PAGE SECTIONS & PLOTS
# ============================================================================

st.subheader("1️⃣ Data Summary")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Proteins", f"{len(df_raw):,}")

with col2:
    st.metric("Samples", len(numeric_cols))

with col3:
    missing_pct = (df_raw[numeric_cols].isna().sum().sum() / 
                   (len(df_raw) * len(numeric_cols)) * 100)
    st.metric("Missing %", f"{missing_pct:.1f}%")

with col4:
    mean_intensity = df_raw[numeric_cols].mean().mean()
    st.metric("Mean Intensity", f"{mean_intensity:.0f}")

with col5:
    valid_proteins = (df_raw[numeric_cols].notna().sum(axis=1) > 0).sum()
    st.metric("Valid Proteins", f"{valid_proteins:,}")

st.markdown("---")

# ============================================================================
# SECTION 1: Valid Proteins per Species
# ============================================================================

st.subheader("2️⃣ Valid Proteins per Species per Sample")
st.info("**Valid = intensity > 1.0** (excludes missing/NaN/zero)")

df_counts = pl.from_dict(compute_valid_counts(
    df_raw.to_dict(as_series=False), 
    id_col, 
    species_col, 
    numeric_cols
))

plot = (ggplot(df_counts.to_pandas(), aes(x='sample', y='count', fill=species_col)) +
        geom_bar(stat='identity') +
        geom_text(aes(y='label_pos', label='count'), 
                  size=8, color='white', fontweight='bold') +
        labs(title='Valid Protein Count by Species per Sample',
             x='Sample', y='Protein Count', fill='Species') +
        theme_minimal() +
        theme(axis_text_x=element_text(rotation=45, hjust=1),
              figure_size=(10, 5)))

fig = plot.draw()
st.pyplot(fig)
clear_plot_memory()

df_table = pl.from_dict(compute_valid_table(
    df_raw.to_dict(as_series=False), 
    id_col, 
    species_col, 
    numeric_cols
))

st.markdown("**Valid Proteins per Species per Sample:**")
st.dataframe(df_table.to_pandas(), use_container_width=True)

st.download_button(
    "📥 Download Table (CSV)",
    df_table.write_csv(),
    "valid_proteins_per_species.csv",
    "text/csv",
    key="download_valid_proteins"
)

st.markdown("---")

# ============================================================================
# SECTION 2: Log2 Intensity Distribution
# ============================================================================

st.subheader("3️⃣ Log2 Intensity Distribution by Sample")

# Compute log2 transformed data
df_raw_dict = df_raw.to_dict(as_series=False)
df_log2_dict = compute_log2(df_raw_dict, numeric_cols)
df_log2 = pd.DataFrame(df_log2_dict)

intensity_data = compute_intensity_stats(df_log2_dict, id_col, numeric_cols)
df_long = pl.from_dict(intensity_data['long_data'])

plot = (ggplot(df_long.to_pandas(), aes(x='sample', y='log2_intensity', fill='condition')) +
        geom_violin(alpha=0.7) +
        geom_boxplot(width=0.1, fill='white', outlier_alpha=0.3) +
        scale_fill_manual(values=['#66c2a5', '#fc8d62']) +
        labs(title='Log2 Intensity Distribution',
             x='Sample', y='Log2 Intensity', fill='Condition') +
        theme_minimal() +
        theme(axis_text_x=element_text(rotation=45, hjust=1),
              figure_size=(12, 6)))

fig = plot.draw()
st.pyplot(fig)
clear_plot_memory()

df_stats = pl.from_dict(intensity_data['stats'])

st.markdown("**Summary Statistics:**")
st.dataframe(df_stats.to_pandas(), use_container_width=True)

st.download_button(
    "📥 Download Statistics (CSV)",
    df_stats.write_csv(),
    "log2_intensity_statistics.csv",
    "text/csv",
    key="download_intensity_stats"
)

st.markdown("---")

# ============================================================================
# SECTION 3: CV Distribution
# ============================================================================

st.subheader("4️⃣ Coefficient of Variation (CV) by Condition")
st.info("**CV = (std / mean) × 100** for each protein across replicates. Lower CV = better reproducibility.")

cv_result = compute_cv_data(df_raw_dict, id_col, numeric_cols)
df_cv_long = pl.DataFrame(cv_result['cv_data'])

high_cv_counts = df_cv_long.filter(pl.col('cv') > 100).group_by('condition').agg(
    pl.len().alias('n_high_cv')
).sort('condition')

if high_cv_counts.shape[0] > 0:
    warning_text = "⚠️ **High CV (>100%) detected:**  "
    warning_parts = []
    for row in high_cv_counts.iter_rows(named=True):
        warning_parts.append(f"**{row['condition']}**: {row['n_high_cv']} proteins")
    st.warning(warning_text + " | ".join(warning_parts))

df_cv_plot = df_cv_long.with_columns(
    pl.col('cv').clip(upper_bound=100).alias('cv_capped')
)

plot = (ggplot(df_cv_plot.to_pandas(), aes(x='condition', y='cv_capped', fill='condition')) +
        geom_violin(alpha=0.7) +
        geom_boxplot(width=0.1, fill='white', outlier_alpha=0) +
        scale_fill_brewer(type='qual', palette='Set2') +
        scale_y_continuous(limits=[0, 100]) +
        labs(title='Coefficient of Variation Distribution by Condition (capped at 100%)',
             x='Condition', y='CV (%)') +
        theme_minimal() +
        theme(axis_text_x=element_text(size=12),
              figure_size=(10, 6),
              legend_position='none'))

fig = plot.draw()
st.pyplot(fig)
clear_plot_memory()

df_cv_stats = df_cv_long.group_by('condition').agg([
    pl.col('cv').len().alias('n_proteins'),
    pl.col('cv').mean().alias('mean_cv'),
    pl.col('cv').median().alias('median_cv'),
    pl.col('cv').std().alias('std_cv'),
    pl.col('cv').quantile(0.25).alias('q25'),
    pl.col('cv').quantile(0.75).alias('q75'),
    (pl.col('cv') > 100).sum().alias('n_cv_over_100')
]).sort('condition')

st.markdown("**CV Summary Statistics:**")
st.dataframe(df_cv_stats.to_pandas(), use_container_width=True)

st.download_button(
    "📥 Download CV Statistics (CSV)",
    df_cv_stats.write_csv(),
    "cv_statistics.csv",
    "text/csv",
    key="download_cv_stats"
)

st.markdown("---")

# ============================================================================
# SECTION 4: CV Thresholds
# ============================================================================

st.subheader("5️⃣ Protein Count by CV Threshold")
st.info("**Quality tiers:** Total (all valid), CV < 20% (good+excellent), CV < 10% (excellent)")

df_cv_threshold = pl.DataFrame(cv_result['cv_threshold_data'])

plot = (ggplot(df_cv_threshold.to_pandas(), aes(x='condition', y='count', fill='threshold')) +
        geom_bar(stat='identity', position='dodge') +
        geom_text(aes(label='count'), position=position_dodge(width=0.9),
                  va='bottom', size=9, fontweight='bold') +
        scale_fill_manual(
            values={
                'Total': '#95a5a6',
                'CV < 20%': '#f39c12',
                'CV < 10%': '#2ecc71'
            },
            breaks=['Total', 'CV < 20%', 'CV < 10%']
        ) +
        labs(title='Protein Count by CV Quality Threshold',
             x='Condition', y='Protein Count', fill='Threshold') +
        theme_minimal() +
        theme(axis_text_x=element_text(size=12),
              figure_size=(10, 6)))

fig = plot.draw()
st.pyplot(fig)
clear_plot_memory()

st.markdown("---")

# ============================================================================
# SECTION 5: Missing Data
# ============================================================================

st.subheader("6️⃣ Missing Data Pattern")

missing_data = compute_missing_data(df_raw_dict, id_col, numeric_cols)
df_missing = pl.DataFrame(missing_data)

plot = (ggplot(df_missing.to_pandas(), aes(x='n_missing', y='count', fill='condition')) +
        geom_bar(stat='identity', position='dodge') +
        geom_text(aes(label='count'), position=position_dodge(width=0.9),
                  va='bottom', size=8, fontweight='bold') +
        scale_fill_brewer(type='qual', palette='Set1') +
        labs(title='Protein Missing Data Distribution',
             x='Number of Missing Values', y='Protein Count', fill='Condition') +
        theme_minimal() +
        theme(axis_text_x=element_text(angle=45, hjust=1),
              figure_size=(10, 6)))

fig = plot.draw()
st.pyplot(fig)
clear_plot_memory()

st.markdown("---")

# ============================================================================
# NAVIGATION & EXPORT
# ============================================================================

st.subheader("7️⃣ Next Steps")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Go to Statistical EDA", use_container_width=True, key="btn_stat_eda"):
        st.switch_page("pages/3_Statistical_EDA.py")

with col2:
    if st.button("📈 Go to Quality Overview", use_container_width=True, key="btn_quality"):
        st.switch_page("pages/4_Quality_Overview.py")

with col3:
    if st.button("🧬 Go to Data Upload", use_container_width=True, key="btn_upload"):
        st.switch_page("pages/1_Data_Upload.py")

st.markdown("---")

# Footer
col1, col2 = st.columns([1, 1])

with col1:
    st.caption("💡 **Tip:** High CV values indicate inconsistent measurements across replicates")

with col2:
    st.caption("📖 **Next:** Explore statistical testing in the **Statistical EDA** page")

# Cleanup
gc.collect()
