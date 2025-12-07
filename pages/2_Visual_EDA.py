"""
pages/2_Visual_EDA.py
Exploratory data analysis - CORRECTED & OPTIMIZED VERSION

"""

import streamlit as st
import polars as pl
from plotnine import *
import numpy as np
import matplotlib.pyplot as plt
import gc

st.set_page_config(page_title="Visual EDA", page_icon="📊", layout="wide")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clear_plot_memory():
    """Close all matplotlib figures and collect garbage."""
    plt.close('all')
    gc.collect()

# ============================================================================
# SHARED CACHE FUNCTIONS
# ============================================================================

@st.cache_data
def compute_log2(df_dict: dict, cols: list, data_type: str) -> dict:
    """Cache log2 transformation."""
    df_temp = pl.from_dict(df_dict)
    df_log2 = df_temp.with_columns([
        pl.col(c).clip(lower_bound=1.0).log(2).alias(c) for c in cols
    ])
    return df_log2.to_dict(as_series=False)

@st.cache_data
def compute_valid_counts(df_dict: dict, id_col: str, species_col: str, numeric_cols: list) -> dict:
    """Cache valid protein/peptide counts per species per sample."""
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
    """Cache valid protein/peptide table."""
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
def compute_cv_data(df_dict: dict, id_col: str, numeric_cols: list, data_type: str) -> dict:
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
def compute_missing_data(df_dict: dict, id_col: str, numeric_cols: list, replicates: int) -> dict:
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

@st.cache_data
def compute_peptide_metrics(df_dict: dict, numeric_cols: list, id_col: str) -> dict:
    """Pre-calculate completeness and CV for all peptides - SINGLE DEFINITION."""
    df_temp = pl.from_dict(df_dict)
    
    # Group by condition
    conditions = {}
    for col in numeric_cols:
        condition = col[0]
        if condition not in conditions:
            conditions[condition] = []
        conditions[condition].append(col)
    
    # Calculate completeness per condition
    completeness_exprs = []
    for condition, cols in conditions.items():
        completeness_expr = (
            pl.sum_horizontal([
                (pl.col(c) > 1.0) & pl.col(c).is_finite() for c in cols
            ]) / len(cols) * 100
        ).alias(f'completeness_{condition}')
        completeness_exprs.append(completeness_expr)
    
    df_metrics = df_temp.with_columns(completeness_exprs)
    
    # Calculate CV per condition
    cv_exprs = []
    for condition, cols in conditions.items():
        mean_expr = pl.concat_list(cols).list.eval(
            pl.element().filter((pl.element() > 1.0) & pl.element().is_finite())
        ).list.mean()
        std_expr = pl.concat_list(cols).list.eval(
            pl.element().filter((pl.element() > 1.0) & pl.element().is_finite())
        ).list.std()
        cv_exprs.append((std_expr / mean_expr * 100).alias(f'cv_{condition}'))
    
    df_metrics = df_metrics.with_columns(cv_exprs)
    
    # Calculate max CV across conditions
    cv_cols = [f'cv_{cond}' for cond in conditions.keys()]
    df_metrics = df_metrics.with_columns([
        pl.max_horizontal(cv_cols).fill_nan(999).alias('max_cv')
    ])
    
    # Calculate min completeness across conditions
    completeness_cols = [f'completeness_{cond}' for cond in conditions.keys()]
    df_metrics = df_metrics.with_columns([
        pl.min_horizontal(completeness_cols).alias('min_completeness')
    ])
    
    return {
        'data': df_metrics.to_dict(as_series=False),
        'conditions': list(conditions.keys()),
        'completeness_cols': completeness_cols,
        'cv_cols': cv_cols
    }

# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def plot_section_1_overview(df, numeric_cols, species_col, data_type):
    """1. Dataset Overview"""
    st.subheader("1️⃣ Dataset Overview")
    st.info(f"📁 {df.shape[0]:,} {data_type}s × {len(numeric_cols)} samples")
    
    n_species = df[species_col].n_unique()
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Total {data_type.title()}s", f"{df.shape[0]:,}")
    c2.metric("Total Samples", len(numeric_cols))
    c3.metric("Species", n_species)


def plot_section_2_stacked_bar(df, id_col, species_col, numeric_cols, data_type, download_key):
    """2. Valid proteins/peptides per species per sample"""
    st.subheader(f"2️⃣ Valid {data_type.title()}s per Species per Sample")
    st.info("**Valid = intensity > 1.0** (excludes missing/NaN/zero)")
    
    df_counts = pl.from_dict(compute_valid_counts(df.to_dict(as_series=False), id_col, species_col, numeric_cols))
    
    plot = (ggplot(df_counts.to_pandas(), aes(x='sample', y='count', fill=species_col)) +
     geom_bar(stat='identity') +
     geom_text(aes(y='label_pos', label='count'), 
               size=8, color='white', fontweight='bold') +
     labs(title=f'Valid {data_type.title()} Count by Species per Sample',
          x='Sample', y=f'{data_type.title()} Count', fill='Species') +
     theme_minimal() +
     theme(axis_text_x=element_text(rotation=45, hjust=1),
           figure_size=(10, 5)))
    
    fig = ggplot.draw(plot)
    st.pyplot(fig)
    plt.close(fig)
    del fig, plot
    
    df_table = pl.from_dict(compute_valid_table(df.to_dict(as_series=False), id_col, species_col, numeric_cols))
    
    st.markdown(f"**Valid {data_type.title()}s per Species per Sample:**")
    st.dataframe(df_table.to_pandas(), width="stretch")
    
    st.download_button(
        "📥 Download Table (CSV)",
        df_table.write_csv(),
        f"valid_{data_type}s_per_species.csv",
        "text/csv",
        key=f"{download_key}_valid_table"
    )
    
    del df_counts, df_table
    gc.collect()

def plot_section_3_intensity_dist(df_log2, id_col, numeric_cols, download_key):
    """3. Log2 Intensity Distribution"""
    st.subheader("3️⃣ Log2 Intensity Distribution by Sample")
    
    intensity_data = compute_intensity_stats(df_log2.to_dict(as_series=False), id_col, numeric_cols)
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
    
    fig = ggplot.draw(plot)
    st.pyplot(fig)
    plt.close(fig)
    del fig, plot
    
    df_stats = pl.from_dict(intensity_data['stats'])
    
    st.markdown("**Summary Statistics:**")
    st.dataframe(df_stats.to_pandas(), width="stretch")
    
    st.download_button(
        "📥 Download Statistics (CSV)",
        df_stats.write_csv(),
        "log2_intensity_statistics.csv",
        "text/csv",
        key=f"{download_key}_intensity_stats"
    )
    
    del df_long, df_stats, intensity_data
    gc.collect()

def plot_section_4_cv_distribution(df, id_col, numeric_cols, data_type, download_key):
    """4. CV Distribution"""
    st.subheader("4️⃣ Coefficient of Variation (CV) by Condition")
    st.info("**CV = (std / mean) × 100** for each protein across replicates. Lower CV = better reproducibility.")
    
    cv_result = compute_cv_data(df.to_dict(as_series=False), id_col, numeric_cols, data_type)
    df_cv_long = pl.DataFrame(cv_result['cv_data'])
    
    high_cv_counts = df_cv_long.filter(pl.col('cv') > 100).group_by('condition').agg(
        pl.len().alias('n_high_cv')
    ).sort('condition')
    
    if high_cv_counts.shape[0] > 0:
        warning_text = "⚠️ **High CV (>100%) detected:**  "
        warning_parts = []
        for row in high_cv_counts.iter_rows(named=True):
            warning_parts.append(f"**{row['condition']}**: {row['n_high_cv']} {data_type}s")
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
    
    fig = ggplot.draw(plot)
    st.pyplot(fig)
    plt.close(fig)
    del fig, plot
    
    df_cv_stats = df_cv_long.group_by('condition').agg([
        pl.col('cv').len().alias(f'n_{data_type}s'),
        pl.col('cv').mean().alias('mean_cv'),
        pl.col('cv').median().alias('median_cv'),
        pl.col('cv').std().alias('std_cv'),
        pl.col('cv').quantile(0.25).alias('q25'),
        pl.col('cv').quantile(0.75).alias('q75'),
        (pl.col('cv') > 100).sum().alias('n_cv_over_100')
    ]).sort('condition')
    
    st.markdown("**CV Summary Statistics:**")
    st.dataframe(df_cv_stats.to_pandas(), width="stretch")
    
    st.download_button(
        "📥 Download CV Statistics (CSV)",
        df_cv_stats.write_csv(),
        "cv_statistics.csv",
        "text/csv",
        key=f"{download_key}_cv_stats"
    )
    
    del df_cv_long, df_cv_plot, df_cv_stats, high_cv_counts
    gc.collect()
    
    return cv_result

def plot_section_5_cv_thresholds(cv_result, data_type, download_key):
    """5. Protein/Peptide Count by CV Threshold"""
    st.subheader(f"5️⃣ {data_type.title()} Count by CV Threshold")
    st.info("**Quality tiers:** Total (all valid), CV < 20% (good+excellent), CV < 10% (excellent)")
    
    df_cv_plot = pl.DataFrame(cv_result['cv_threshold_data'])
    
    df_cv_plot_ordered = df_cv_plot.sort(['condition', 'threshold'], 
                                         descending=[False, False])
    
    plot = (ggplot(df_cv_plot_ordered.to_pandas(), aes(x='condition', y='count', fill='threshold')) +
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
     labs(title=f'{data_type.title()} Count by CV Quality Threshold',
          x='Condition', y=f'{data_type.title()} Count', fill='Threshold') +
     theme_minimal() +
     theme(axis_text_x=element_text(size=12),
           figure_size=(10, 6)))
    
    fig = ggplot.draw(plot)
    st.pyplot(fig)
    plt.close(fig)
    del fig, plot
    del df_cv_plot, df_cv_plot_ordered
    gc.collect()

def plot_section_6_missing_values(df, id_col, numeric_cols, replicates, data_type, download_key):
    """6. Missing Values per Protein/Peptide"""
    st.subheader(f"6️⃣ Missing Values per {data_type.title()} by Condition")
    st.info("**Missing = intensity ≤ 1.0** (includes NaN, zero, and 1.0). Shows how many replicates are missing per protein.")
    
    missing_plot_data = compute_missing_data(df.to_dict(as_series=False), id_col, numeric_cols, replicates)
    df_missing_plot = pl.DataFrame(missing_plot_data)
    
    plot = (ggplot(df_missing_plot.to_pandas(), aes(x='condition', y='count', fill='n_missing')) +
     geom_bar(stat='identity', position='dodge') +
     geom_text(aes(label='count'), position=position_dodge(width=0.9),
               va='bottom', size=8, fontweight='bold') +
     scale_fill_brewer(type='seq', palette='YlOrRd') +
     labs(title=f'{data_type.title()} Count by Number of Missing Values per Condition',
          x='Condition', y=f'{data_type.title()} Count', fill='Missing Values') +
     theme_minimal() +
     theme(axis_text_x=element_text(size=12),
           figure_size=(12, 6)))
    
    fig = ggplot.draw(plot)
    st.pyplot(fig)
    plt.close(fig)
    del fig, plot
    
    st.markdown("**Missing Values Summary:**")
    
    df_missing_summary = df_missing_plot.pivot(
        index='condition',
        on='n_missing',
        values='count'
    ).fill_null(0)
    
    col_order = [f'{i} missing' for i in range(replicates + 1)]
    df_missing_summary = df_missing_summary.select(
        ['condition'] + [c for c in col_order if c in df_missing_summary.columns]
    ).with_columns([
        pl.sum_horizontal([c for c in col_order if c in df_missing_summary.columns]).alias('Total'),
        (pl.col('0 missing') / pl.sum_horizontal([c for c in col_order if c in df_missing_summary.columns]) * 100).alias('% Complete')
    ])
    
    st.dataframe(df_missing_summary.to_pandas(), width="stretch")
    
    st.download_button(
        "📥 Download Missing Values Summary (CSV)",
        df_missing_summary.write_csv(),
        "missing_values_summary.csv",
        "text/csv",
        key=f"{download_key}_missing"
    )
    
    del df_missing_plot, df_missing_summary, missing_plot_data
    gc.collect()

# ============================================================================
# FILTER HELPERS
# ============================================================================

def apply_filters(df, numeric_cols, id_col, max_cv, drop_missing):
    """Apply CV and missing value filters to dataframe."""
    df_filtered = df.clone()
    
    # Group by condition for CV calculation
    conditions = {}
    for col in numeric_cols:
        condition = col[0]
        if condition not in conditions:
            conditions[condition] = []
        conditions[condition].append(col)
    
    # Calculate CV for each row and condition
    if max_cv < 100:
        keep_rows = pl.Series([True] * df_filtered.shape[0])
        
        for condition, cols in conditions.items():
            df_cv = df_filtered.select([id_col] + cols).with_columns([
                pl.concat_list(cols).list.mean().alias('mean'),
                pl.concat_list(cols).list.std().alias('std')
            ]).with_columns(
                (pl.col('std') / pl.col('mean') * 100).alias('cv')
            )
            
            # Mark rows with CV > threshold in ANY condition
            keep_rows = keep_rows & ((df_cv['cv'] <= max_cv) | (~df_cv['cv'].is_finite()))
        
        df_filtered = df_filtered.filter(keep_rows)
    
    # Drop rows with ANY missing values
    if drop_missing:
        has_valid = pl.lit(True)
        for col in numeric_cols:
            has_valid = has_valid & (pl.col(col) > 1.0) & (pl.col(col).is_finite())
        
        df_filtered = df_filtered.filter(has_valid)
    
    return df_filtered

# ============================================================================
# CHECK DATA AVAILABILITY
# ============================================================================

has_protein = 'df_protein' in st.session_state
has_peptide = 'df_peptide' in st.session_state

if not has_protein and not has_peptide:
    st.warning("⚠️ No data loaded. Please upload data first.")
    if st.button("← Go to Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

st.title("📊 Visual Exploratory Data Analysis")

# ============================================================================
# CREATE TABS
# ============================================================================

if has_protein and has_peptide:
    tab_protein, tab_peptide = st.tabs(["🧬 Protein-Level EDA", "🔬 Peptide-Level EDA"])
elif has_protein:
    tab_protein = st.container()
    tab_peptide = None
else:
    tab_protein = None
    tab_peptide = st.container()

# ============================================================================
# PROTEIN EDA
# ============================================================================

if has_protein and tab_protein:
    with tab_protein:
        
        df = st.session_state.df_protein
        numeric_cols = st.session_state.protein_cols
        id_col = st.session_state.protein_id_col
        species_col = st.session_state.protein_species_col
        replicates = st.session_state.protein_replicates
        
        st.header("🧬 Protein-Level Analysis")
        
        # ====================================================================
        # RUN ANALYSIS ON ORIGINAL DATA
        # ====================================================================
        
        df_log2 = pl.from_dict(compute_log2(df.to_dict(as_series=False), numeric_cols, 'protein'))
        
        plot_section_1_overview(df, numeric_cols, species_col, 'protein')
        st.markdown("---")
        
        plot_section_2_stacked_bar(df, id_col, species_col, numeric_cols, 'protein', 'protein')
        st.markdown("---")
        
        plot_section_3_intensity_dist(df_log2, id_col, numeric_cols, 'protein')
        st.markdown("---")
        
        cv_result = plot_section_4_cv_distribution(df, id_col, numeric_cols, 'protein', 'protein')
        st.markdown("---")
        
        plot_section_5_cv_thresholds(cv_result, 'protein', 'protein')
        st.markdown("---")
        
        plot_section_6_missing_values(df, id_col, numeric_cols, replicates, 'protein', 'protein')
        st.markdown("---")
        
        del df_log2, cv_result
        clear_plot_memory()
        
        # ====================================================================
        # FILTER CONTROLS (BOTTOM)
        # ====================================================================
        
        st.markdown("---")
        st.subheader("🔍 Apply Filters for Next Steps")
        st.info("Filter the dataset before proceeding to normalization. Filters will be applied to downstream analysis.")
        
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            max_cv = st.slider(
                "Max CV (%)",
                min_value=0,
                max_value=100,
                value=100,
                step=5,
                help="Remove proteins with CV > threshold in ANY condition",
                key="protein_max_cv"
            )
        
        with col_f2:
            drop_missing = st.checkbox(
                "Drop rows with ANY missing values",
                value=False,
                help="Remove proteins that have missing values (≤1.0) in ANY sample",
                key="protein_drop_missing"
            )
        
        # Apply filters
        df_filtered = apply_filters(df, numeric_cols, id_col, max_cv, drop_missing)
        
        n_removed = df.shape[0] - df_filtered.shape[0]
        pct_removed = n_removed / df.shape[0] * 100 if df.shape[0] > 0 else 0
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Original", f"{df.shape[0]:,}")
        with col_m2:
            st.metric("Removed", f"{n_removed:,} ({pct_removed:.1f}%)")
        with col_m3:
            st.metric("Remaining", f"{df_filtered.shape[0]:,}")
        
        # Save filtered data to session state
        if st.button("✅ Apply Filters & Continue", type="primary", width="stretch", key="protein_apply_filters"):
            st.session_state.df_protein_filtered = df_filtered
            st.session_state.protein_filters_applied = {
                'max_cv': max_cv,
                'drop_missing': drop_missing,
                'n_removed': n_removed
            }
            st.success(f"✅ Filters applied! {df_filtered.shape[0]:,} proteins ready for normalization.")
            st.rerun()
        
        # Show current filter status
        if 'protein_filters_applied' in st.session_state:
            st.info(f"""
            **Current filters:**
            - Max CV: {st.session_state.protein_filters_applied['max_cv']}%
            - Drop missing: {st.session_state.protein_filters_applied['drop_missing']}
            - Removed: {st.session_state.protein_filters_applied['n_removed']:,} proteins
            """)
        
        del df_filtered

# ============================================================================
# PEPTIDE EDA
# ============================================================================

if has_peptide and tab_peptide:
    with tab_peptide:
        
        df = st.session_state.df_peptide
        numeric_cols = st.session_state.peptide_cols
        id_col = st.session_state.peptide_id_col
        species_col = st.session_state.peptide_species_col
        replicates = st.session_state.peptide_replicates
        
        st.header("🔬 Peptide-Level Analysis")
        
        # ====================================================================
        # OVERVIEW
        # ====================================================================
        
        st.subheader("📊 Dataset Overview")
        
        n_proteins = df[id_col].n_unique()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Peptides", f"{df.shape[0]:,}")
        col2.metric("Unique Proteins", f"{n_proteins:,}")
        col3.metric("Avg Peptides/Protein", f"{df.shape[0] / n_proteins:.1f}")
        
        st.markdown("---")
        
        # ====================================================================
        # PRE-CALCULATE METRICS ONCE (CACHED)
        # ====================================================================
        
        with st.spinner("🔄 Computing peptide metrics..."):
            metrics_result = compute_peptide_metrics(df.to_dict(as_series=False), numeric_cols, id_col)
            df_with_metrics = pl.from_dict(metrics_result['data'])
            conditions_list = metrics_result['conditions']
        
        # ====================================================================
        # HIERARCHICAL FILTERS
        # ====================================================================
        
        st.subheader("🔍 Hierarchical Filters for Peptide → Protein Aggregation")
        st.info("Enable and configure filters as needed. Filters are applied in order: Completeness → CV → Min Peptides")
        
        # ====================================================================
        # FILTER 1: DATA COMPLETENESS PER CONDITION
        # ====================================================================
        
        st.markdown("### Filter 1: Data Completeness per Condition")
        
        col_f1a, col_f1b = st.columns([2, 2])
        
        with col_f1a:
            enable_completeness = st.checkbox(
                "Enable completeness filter",
                value=False,
                help="Filter peptides by data completeness within each condition",
                key="peptide_enable_completeness"
            )
        
        with col_f1b:
            min_completeness = st.number_input(
                "Minimum completeness per condition (%)",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                help="Keep peptides with at least this % of valid values in ALL conditions",
                key="peptide_min_completeness",
                disabled=not enable_completeness
            )
        
        if enable_completeness:
            df_filtered_1 = df_with_metrics.filter(pl.col('min_completeness') >= min_completeness)
            n_removed_1 = df_with_metrics.shape[0] - df_filtered_1.shape[0]
            st.success(f"✅ After Filter 1: **{df_filtered_1.shape[0]:,} peptides** (removed {n_removed_1:,})")
            
            # Show breakdown by condition
            with st.expander("📋 View completeness by condition"):
                comp_stats = []
                for cond in sorted(conditions_list):
                    col_name = f'completeness_{cond}'
                    n_pass = df_with_metrics.filter(pl.col(col_name) >= min_completeness).shape[0]
                    comp_stats.append({
                        'Condition': cond,
                        'Peptides passing': n_pass,
                        '% passing': round(n_pass / df_with_metrics.shape[0] * 100, 1)
                    })
                
                df_comp_stats = pl.DataFrame(comp_stats)
                st.dataframe(df_comp_stats.to_pandas(), hide_index=True)
        else:
            df_filtered_1 = df_with_metrics
            n_removed_1 = 0
            st.info("⏭️ Filter 1 disabled")
        
        # ====================================================================
        # FILTER 2: CV THRESHOLD
        # ====================================================================
        
        st.markdown("### Filter 2: Coefficient of Variation")
        
        col_f2a, col_f2b = st.columns([2, 2])
        
        with col_f2a:
            enable_cv_filter = st.checkbox(
                "Enable CV filter",
                value=False,
                help="Filter peptides by maximum CV across conditions",
                key="peptide_enable_cv"
            )
        
        with col_f2b:
            max_cv = st.number_input(
                "Maximum CV (%)",
                min_value=0,
                max_value=200,
                value=30,
                step=5,
                help="Keep peptides with CV ≤ threshold in ALL conditions",
                key="peptide_max_cv",
                disabled=not enable_cv_filter
            )
        
        if enable_cv_filter:
            df_filtered_2 = df_filtered_1.filter(pl.col('max_cv') <= max_cv)
            n_removed_2 = df_filtered_1.shape[0] - df_filtered_2.shape[0]
            st.success(f"✅ After Filter 2: **{df_filtered_2.shape[0]:,} peptides** (removed {n_removed_2:,})")
        else:
            df_filtered_2 = df_filtered_1
            n_removed_2 = 0
            st.info("⏭️ Filter 2 disabled")
        
        # ====================================================================
        # FILTER 3: MIN PEPTIDES PER PROTEIN
        # ====================================================================
        
        st.markdown("### Filter 3: Minimum Peptides per Protein")
        
        col_f3a, col_f3b = st.columns([2, 2])
        
        with col_f3a:
            enable_min_peptides = st.checkbox(
                "Enable min peptides filter",
                value=False,
                help="Keep only proteins with minimum number of peptides",
                key="peptide_enable_min_peptides"
            )
        
        with col_f3b:
            min_peptides = st.number_input(
                "Minimum peptides per protein",
                min_value=1,
                max_value=20,
                value=2,
                step=1,
                help="Keep only proteins with at least this many peptides",
                key="peptide_min_peptides",
                disabled=not enable_min_peptides
            )
        
        # Always define df_filtered_3 and n_proteins_final
        if enable_min_peptides:
            # Count peptides per protein
            peptides_per_protein = df_filtered_2.group_by(id_col).agg(
                pl.len().alias('n_peptides')
            )
            
            # Keep proteins with enough peptides
            proteins_to_keep = peptides_per_protein.filter(
                pl.col('n_peptides') >= min_peptides
            )[id_col].to_list()
            
            df_filtered_3 = df_filtered_2.filter(pl.col(id_col).is_in(proteins_to_keep))
            n_removed_3 = df_filtered_2.shape[0] - df_filtered_3.shape[0]
            n_proteins_final = len(proteins_to_keep)
            
            st.success(f"✅ After Filter 3: **{df_filtered_3.shape[0]:,} peptides** from **{n_proteins_final:,} proteins** (removed {n_removed_3:,} peptides)")
        else:
            df_filtered_3 = df_filtered_2
            n_removed_3 = 0
            n_proteins_final = df_filtered_3[id_col].n_unique()
            st.info(f"⏭️ Filter 3 disabled - {df_filtered_3.shape[0]:,} peptides from {n_proteins_final:,} proteins")
        
        # ====================================================================
        # FINAL SUMMARY
        # ====================================================================
        
        st.markdown("---")
        st.subheader("📊 Final Summary")
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            st.metric("Original Peptides", f"{df.shape[0]:,}")
        with col_s2:
            st.metric("Final Peptides", f"{df_filtered_3.shape[0]:,}")
        with col_s3:
            st.metric("Original Proteins", f"{n_proteins:,}")
        with col_s4:
            st.metric("Final Proteins", f"{n_proteins_final:,}")
        
        # Show removed counts
        total_removed = df.shape[0] - df_filtered_3.shape[0]
        pct_peptides_removed = total_removed / df.shape[0] * 100 if df.shape[0] > 0 else 0
        pct_proteins_removed = (n_proteins - n_proteins_final) / n_proteins * 100 if n_proteins > 0 else 0
        
        st.info(f"""
        **Removed:**
        - Peptides: {total_removed:,} ({pct_peptides_removed:.1f}%)
        - Proteins: {n_proteins - n_proteins_final:,} ({pct_proteins_removed:.1f}%)
        - Avg peptides/protein: {df_filtered_3.shape[0] / n_proteins_final:.1f} (was {df.shape[0] / n_proteins:.1f})
        """)
        
        # ====================================================================
        # PLOTS (LAZY LOADED IN EXPANDER)
        # ====================================================================
        
        with st.expander("📊 View Intensity Distribution Plots", expanded=False):
            st.markdown("### Log2 Intensity Distribution by Sample")
            
            with st.spinner("🎨 Generating plots..."):
                df_log2 = pl.from_dict(compute_log2(df_filtered_3.to_dict(as_series=False), numeric_cols, 'peptide'))
                
                intensity_data = compute_intensity_stats(df_log2.to_dict(as_series=False), id_col, numeric_cols)
                df_long = pl.from_dict(intensity_data['long_data'])
                
                plot = (ggplot(df_long.to_pandas(), aes(x='sample', y='log2_intensity', fill='condition')) +
                 geom_violin(alpha=0.7) +
                 geom_boxplot(width=0.1, fill='white', outlier_alpha=0.3) +
                 scale_fill_manual(values=['#66c2a5', '#fc8d62']) +
                 labs(title='Log2 Intensity Distribution (Filtered Data)',
                      x='Sample', y='Log2 Intensity', fill='Condition') +
                 theme_minimal() +
                 theme(axis_text_x=element_text(rotation=45, hjust=1),
                       figure_size=(12, 6)))
                
                fig = ggplot.draw(plot)
                st.pyplot(fig)
                plt.close(fig)
                del fig, plot
                
                # Summary statistics table
                df_stats = pl.from_dict(intensity_data['stats'])
                
                st.markdown("**Summary Statistics:**")
                st.dataframe(df_stats.to_pandas(), width="stretch")
                
                st.download_button(
                    "📥 Download Statistics (CSV)",
                    df_stats.write_csv(),
                    "log2_intensity_statistics_peptide_filtered.csv",
                    "text/csv",
                    key="peptide_download_intensity_stats"
                )
                
                del df_long, df_stats, intensity_data, df_log2
                gc.collect()
        
        # ====================================================================
        # SAVE FILTERED DATA
        # ====================================================================
        
        st.markdown("---")
        
        if st.button("✅ Apply Filters & Continue", type="primary", width="stretch", key="peptide_apply_filters"):
            # Columns to drop
            cols_to_drop = ['min_completeness', 'max_cv'] + metrics_result['completeness_cols'] + metrics_result['cv_cols']
            
            # Only drop columns that exist
            cols_to_drop = [c for c in cols_to_drop if c in df_filtered_3.columns]
            
            # Save filtered peptide data
            st.session_state.df_peptide_filtered = df_filtered_3.drop(cols_to_drop)
            st.session_state.peptide_filters_applied = {
                'enable_completeness': enable_completeness,
                'min_completeness': min_completeness if enable_completeness else None,
                'enable_cv_filter': enable_cv_filter,
                'max_cv': max_cv if enable_cv_filter else None,
                'enable_min_peptides': enable_min_peptides,
                'min_peptides': min_peptides if enable_min_peptides else None,
                'n_peptides_original': df.shape[0],
                'n_peptides_final': df_filtered_3.shape[0],
                'n_proteins_original': n_proteins,
                'n_proteins_final': n_proteins_final
            }
            st.success(f"✅ Filters applied! {df_filtered_3.shape[0]:,} peptides from {n_proteins_final:,} proteins ready for aggregation.")
            st.rerun()
        
        # Show current saved filter status
        if 'peptide_filters_applied' in st.session_state:
            st.markdown("---")
            st.success("✅ **Filters have been saved**")
            
            filt = st.session_state.peptide_filters_applied
            filter_text = "**Applied filters:**\n"
            
            if filt['enable_completeness']:
                filter_text += f"- Completeness ≥ {filt['min_completeness']}%\n"
            if filt['enable_cv_filter']:
                filter_text += f"- Max CV ≤ {filt['max_cv']}%\n"
            if filt['enable_min_peptides']:
                filter_text += f"- Min peptides/protein: {filt['min_peptides']}\n"
            
            if not any([filt['enable_completeness'], filt['enable_cv_filter'], filt['enable_min_peptides']]):
                filter_text += "- No filters enabled (using all data)\n"
            
            filter_text += f"\n**Result:** {filt['n_peptides_final']:,} peptides → {filt['n_proteins_final']:,} proteins"
            
            st.info(filter_text)
        
        del df_with_metrics, df_filtered_1, df_filtered_2, df_filtered_3, metrics_result
        clear_plot_memory()

# ============================================================================
# NAVIGATION (OUTSIDE TABS)
# ============================================================================

clear_plot_memory()

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("← Back to Upload", width="stretch"):
        st.switch_page("pages/1_Data_Upload.py")

with col2:
    # Check if filters have been applied
    can_continue = False
    if has_protein and has_peptide:
        can_continue = 'protein_filters_applied' in st.session_state and 'peptide_filters_applied' in st.session_state
    elif has_protein:
        can_continue = 'protein_filters_applied' in st.session_state
    elif has_peptide:
        can_continue = 'peptide_filters_applied' in st.session_state
    
    if st.button("Continue to Normalization →", type="primary", width="stretch", disabled=not can_continue):
        st.switch_page("pages/3_Normalization.py")
    
    if not can_continue:
        st.caption("⚠️ Apply filters in each tab before continuing")
