"""
pages/2_Visual_EDA.py
Exploratory data analysis for protein and/or peptide data - FULLY OPTIMIZED
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
    
    # Calculate species order (return as list, not categorical)
    species_order = df_counts.group_by(species_col).agg(
        pl.col('count').sum().alias('total')
    ).sort('total', descending=True)[species_col].to_list()
    
    # Sort by species order manually
    df_counts = df_counts.sort(['sample', species_col]).with_columns([
        (pl.col('count').cum_sum().over('sample') - pl.col('count') / 2).alias('label_pos')
    ])
    
    return {
        'counts': df_counts.to_dict(as_series=False),
        'species_order': species_order
    }

@st.cache_data
def compute_valid_table(df_dict: dict, id_col: str, species_col: str, numeric_cols: list) -> dict:
    """Cache valid protein/peptide table."""
    df_temp = pl.from_dict(df_dict)
    
    # Pivot table
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
    
    # Add totals
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
def compute_cv_data(df_dict: dict, id_col: str, numeric_cols: list, data_type: str) -> dict:
    """Cache CV calculations."""
    df_temp = pl.from_dict(df_dict)
    
    # Group by condition
    conditions = {}
    for col in numeric_cols:
        condition = col[0]
        if condition not in conditions:
            conditions[condition] = []
        conditions[condition].append(col)
    
    cv_data = []
    for condition, cols in conditions.items():
        df_cv = df_temp.select([id_col] + cols).with_columns([
            pl.concat_list(cols).list.mean().alias('mean'),
            pl.concat_list(cols).list.std().alias('std')
        ]).with_columns(
            (pl.col('std') / pl.col('mean') * 100).alias('cv')
        ).filter(
            pl.col('cv').is_finite() & (pl.col('cv') > 0)
        )
        
        for row in df_cv.select([id_col, 'cv']).iter_rows(named=True):
            cv_data.append({
                'id': row[id_col],
                'condition': condition,
                'cv': row['cv']
            })
    
    return {'cv_data': cv_data, 'conditions': conditions}

@st.cache_data
def compute_missing_data(df_dict: dict, id_col: str, numeric_cols: list, replicates: int) -> dict:
    """Cache missing value calculations."""
    df_temp = pl.from_dict(df_dict)
    
    # Group by condition
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
        
        # Load cached data
        df_log2 = pl.from_dict(compute_log2(df.to_dict(as_series=False), numeric_cols, 'protein'))
        
        # ====================================================================
        # 1. OVERVIEW
        # ====================================================================
        
        st.subheader("1️⃣ Dataset Overview")
        
        st.info(f"📁 {df.shape[0]:,} proteins × {len(numeric_cols)} samples")
        
        n_species = df[species_col].n_unique()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Proteins", f"{df.shape[0]:,}")
        c2.metric("Total Samples", len(numeric_cols))
        c3.metric("Species", n_species)
        c4.metric("Avg/Species", int(df.shape[0] / n_species))
        
        st.markdown("---")
        
        # ====================================================================
        # 2. STACKED BAR
        # ====================================================================
        
        st.subheader("2️⃣ Valid Proteins per Species per Sample")
        st.info("**Valid = intensity > 1.0** (excludes missing/NaN/zero)")
        
        # In both protein and peptide sections, after getting counts_data:
        
        counts_data = compute_valid_counts(df.to_dict(as_series=False), id_col, species_col, numeric_cols)
        df_counts = pl.from_dict(counts_data['counts'])
        species_order = counts_data['species_order']
        
        # Apply categorical ordering for plotting
        df_counts_plot = df_counts.with_columns([
            pl.col(species_col).cast(pl.Categorical(categories=species_order, ordering='physical'))
        ])
        
        plot = (ggplot(df_counts_plot.to_pandas(), aes(x='sample', y='count', fill=species_col)) +
         geom_bar(stat='identity') +
         geom_text(aes(y='label_pos', label='count'), 
                   size=8, color='white', fontweight='bold') +
         labs(title='Valid Protein Count by Species per Sample',  # or Peptide
              x='Sample', y='Protein Count', fill='Species') +
         theme_minimal() +
         theme(axis_text_x=element_text(rotation=45, hjust=1),
               figure_size=(10, 5)))
        
        fig = ggplot.draw(plot)
        st.pyplot(fig)
        plt.close(fig)
        del fig, plot, df_counts_plot  # Add df_counts_plot to cleanup

        
        # Table
        df_table = pl.from_dict(compute_valid_table(df.to_dict(as_series=False), id_col, species_col, numeric_cols))
        
        st.markdown("**Valid Proteins per Species per Sample:**")
        st.dataframe(df_table.to_pandas(), width='stretch')
        
        st.download_button(
            "📥 Download Table (CSV)",
            df_table.write_csv(),
            "valid_proteins_per_species.csv",
            "text/csv",
            key="protein_download_valid_table"
        )
        
        del df_counts, df_table, counts_data
        gc.collect()
        
        st.markdown("---")
        
        # ====================================================================
        # 3. VIOLIN/BOX PLOT
        # ====================================================================
        
        st.subheader("3️⃣ Log2 Intensity Distribution by Sample")
        
        df_long = df_log2.select([id_col] + numeric_cols).melt(
            id_vars=[id_col],
            value_vars=numeric_cols,
            variable_name='sample',
            value_name='log2_intensity'
        ).filter(
            pl.col('log2_intensity').is_finite()
        )
        
        df_long = df_long.with_columns(
            pl.when(pl.col('sample').str.starts_with('A'))
            .then(pl.lit('A'))
            .otherwise(pl.lit('B'))
            .alias('condition')
        )
        
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
        
        df_stats = df_long.group_by('sample').agg([
            pl.col('log2_intensity').len().alias('n'),
            pl.col('log2_intensity').mean().alias('mean'),
            pl.col('log2_intensity').median().alias('median'),
            pl.col('log2_intensity').std().alias('std'),
            pl.col('log2_intensity').quantile(0.25).alias('q25'),
            pl.col('log2_intensity').quantile(0.75).alias('q75')
        ]).sort('sample')
        
        st.markdown("**Summary Statistics:**")
        st.dataframe(df_stats.to_pandas(), width='stretch')
        
        st.download_button(
            "📥 Download Statistics (CSV)",
            df_stats.write_csv(),
            "log2_intensity_statistics.csv",
            "text/csv",
            key="protein_download_intensity_stats"
        )
        
        del df_long, df_stats
        gc.collect()
        
        st.markdown("---")
        
        # ====================================================================
        # 4. CV DISTRIBUTION
        # ====================================================================
        
        st.subheader("4️⃣ Coefficient of Variation (CV) by Condition")
        st.info("**CV = (std / mean) × 100** for each protein across replicates. Lower CV = better reproducibility.")
        
        cv_result = compute_cv_data(df.to_dict(as_series=False), id_col, numeric_cols, 'protein')
        df_cv_long = pl.DataFrame(cv_result['cv_data'])
        conditions = cv_result['conditions']
        
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
        
        fig = ggplot.draw(plot)
        st.pyplot(fig)
        plt.close(fig)
        del fig, plot
        
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
        st.dataframe(df_cv_stats.to_pandas(), width='stretch')
        
        st.download_button(
            "📥 Download CV Statistics (CSV)",
            df_cv_stats.write_csv(),
            "cv_statistics.csv",
            "text/csv",
            key="protein_download_cv_stats"
        )
        
        del df_cv_long, df_cv_plot, df_cv_stats, high_cv_counts
        gc.collect()
        
        st.markdown("---")
        
        # ====================================================================
        # 5. PROTEIN COUNT BY CV THRESHOLD
        # ====================================================================
        
        st.subheader("5️⃣ Protein Count by CV Threshold")
        st.info("**Quality tiers:** Total (all valid), CV < 20% (good+excellent), CV < 10% (excellent)")
        
        cv_plot_data = []
        
        for condition, cols in conditions.items():
            df_cv_cond = df.select([id_col] + cols).with_columns([
                pl.concat_list(cols).list.mean().alias('mean'),
                pl.concat_list(cols).list.std().alias('std')
            ]).with_columns(
                (pl.col('std') / pl.col('mean') * 100).alias('cv')
            ).filter(
                pl.col('cv').is_finite() & (pl.col('cv') > 0)
            )
            
            total = df_cv_cond.shape[0]
            cv_under_20 = df_cv_cond.filter(pl.col('cv') < 20).shape[0]
            cv_under_10 = df_cv_cond.filter(pl.col('cv') < 10).shape[0]
            
            cv_plot_data.append({'condition': condition, 'threshold': 'Total', 'count': total})
            cv_plot_data.append({'condition': condition, 'threshold': 'CV < 20%', 'count': cv_under_20})
            cv_plot_data.append({'condition': condition, 'threshold': 'CV < 10%', 'count': cv_under_10})
        
        df_cv_plot = pl.DataFrame(cv_plot_data)
        
        df_cv_plot = df_cv_plot.with_columns(
            pl.col('threshold').cast(pl.Categorical(ordering='physical'))
        )
        
        df_cv_plot_ordered = df_cv_plot.sort(['condition', 'threshold'], 
                                             descending=[True, True])
        
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
         labs(title='Protein Count by CV Quality Threshold',
              x='Condition', y='Protein Count', fill='Threshold') +
         theme_minimal() +
         theme(axis_text_x=element_text(size=12),
               figure_size=(10, 6)))
        
        fig = ggplot.draw(plot)
        st.pyplot(fig)
        plt.close(fig)
        del fig, plot
        
        st.markdown("**Protein Counts by CV Threshold:**")
        
        summary_data = []
        for cond in sorted(conditions.keys()):
            cond_data = df_cv_plot.filter(pl.col('condition') == cond)
            total = cond_data.filter(pl.col('threshold') == 'Total')['count'][0]
            cv_lt_20 = cond_data.filter(pl.col('threshold') == 'CV < 20%')['count'][0]
            cv_lt_10 = cond_data.filter(pl.col('threshold') == 'CV < 10%')['count'][0]
            
            summary_data.append({
                'Condition': cond,
                'Total': total,
                'CV < 20%': cv_lt_20,
                'CV < 10%': cv_lt_10,
                '% < 20%': round(cv_lt_20 / total * 100, 1) if total > 0 else 0,
                '% < 10%': round(cv_lt_10 / total * 100, 1) if total > 0 else 0
            })
        
        df_summary = pl.DataFrame(summary_data)
        
        st.dataframe(df_summary.to_pandas(), width='stretch')
        
        st.download_button(
            "📥 Download CV Threshold Summary (CSV)",
            df_summary.write_csv(),
            "cv_threshold_summary.csv",
            "text/csv",
            key="protein_download_cv_threshold"
        )
        
        del df_cv_plot, df_cv_plot_ordered, df_summary, cv_plot_data, summary_data
        gc.collect()
        
        st.markdown("---")
        
        # ====================================================================
        # 6. MISSING VALUES
        # ====================================================================
        
        st.subheader("6️⃣ Missing Values per Protein by Condition")
        st.info("**Missing = intensity ≤ 1.0** (includes NaN, zero, and 1.0). Shows how many replicates are missing per protein.")
        
        missing_plot_data = compute_missing_data(df.to_dict(as_series=False), id_col, numeric_cols, replicates)
        df_missing_plot = pl.DataFrame(missing_plot_data)
        
        plot = (ggplot(df_missing_plot.to_pandas(), aes(x='condition', y='count', fill='n_missing')) +
         geom_bar(stat='identity', position='dodge') +
         geom_text(aes(label='count'), position=position_dodge(width=0.9),
                   va='bottom', size=8, fontweight='bold') +
         scale_fill_brewer(type='seq', palette='YlOrRd') +
         labs(title='Protein Count by Number of Missing Values per Condition',
              x='Condition', y='Protein Count', fill='Missing Values') +
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
            columns='n_missing',
            values='count'
        ).fill_null(0)
        
        col_order = [f'{i} missing' for i in range(replicates + 1)]
        df_missing_summary = df_missing_summary.select(
            ['condition'] + [c for c in col_order if c in df_missing_summary.columns]
        ).with_columns([
            pl.sum_horizontal([c for c in col_order if c in df_missing_summary.columns]).alias('Total'),
            (pl.col('0 missing') / pl.sum_horizontal([c for c in col_order if c in df_missing_summary.columns]) * 100).alias('% Complete')
        ])
        
        st.dataframe(df_missing_summary.to_pandas(), width='stretch')
        
        st.download_button(
            "📥 Download Missing Values Summary (CSV)",
            df_missing_summary.write_csv(),
            "missing_values_summary.csv",
            "text/csv",
            key="protein_download_missing"
        )
        
        del df_missing_plot, df_missing_summary, missing_plot_data
        gc.collect()
        
        st.markdown("---")
        
        del df_log2
        clear_plot_memory()

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
        
        # Load cached data
        df_log2 = pl.from_dict(compute_log2(df.to_dict(as_series=False), numeric_cols, 'peptide'))
        
        # ====================================================================
        # 1. OVERVIEW
        # ====================================================================
        
        st.subheader("1️⃣ Dataset Overview")
        
        st.info(f"📁 {df.shape[0]:,} peptides × {len(numeric_cols)} samples")
        
        n_species = df[species_col].n_unique()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Peptides", f"{df.shape[0]:,}")
        c2.metric("Total Samples", len(numeric_cols))
        c3.metric("Species", n_species)
        c4.metric("Avg/Species", int(df.shape[0] / n_species))
        
        st.markdown("---")
        
        # ====================================================================
        # 2. STACKED BAR
        # ====================================================================
        
        st.subheader("2️⃣ Valid Peptides per Species per Sample")
        st.info("**Valid = intensity > 1.0** (excludes missing/NaN/zero)")
        
        # In both protein and peptide sections, after getting counts_data:

        counts_data = compute_valid_counts(df.to_dict(as_series=False), id_col, species_col, numeric_cols)
        df_counts = pl.from_dict(counts_data['counts'])
        species_order = counts_data['species_order']
        
        # Apply categorical ordering for plotting
        df_counts_plot = df_counts.with_columns([
            pl.col(species_col).cast(pl.Categorical(categories=species_order, ordering='physical'))
        ])
        
        plot = (ggplot(df_counts_plot.to_pandas(), aes(x='sample', y='count', fill=species_col)) +
         geom_bar(stat='identity') +
         geom_text(aes(y='label_pos', label='count'), 
                   size=8, color='white', fontweight='bold') +
         labs(title='Valid Protein Count by Species per Sample',  # or Peptide
              x='Sample', y='Protein Count', fill='Species') +
         theme_minimal() +
         theme(axis_text_x=element_text(rotation=45, hjust=1),
               figure_size=(10, 5)))
        
        fig = ggplot.draw(plot)
        st.pyplot(fig)
        plt.close(fig)
        del fig, plot, df_counts_plot  # Add df_counts_plot to cleanup

        
        # Table
        df_table = pl.from_dict(compute_valid_table(df.to_dict(as_series=False), id_col, species_col, numeric_cols))
        
        st.markdown("**Valid Peptides per Species per Sample:**")
        st.dataframe(df_table.to_pandas(), width='stretch')
        
        st.download_button(
            "📥 Download Table (CSV)",
            df_table.write_csv(),
            "valid_peptides_per_species.csv",
            "text/csv",
            key="peptide_download_valid_table"
        )
        
        del df_counts, df_table, counts_data
        gc.collect()
        
        st.markdown("---")
        
        # ====================================================================
        # 3. VIOLIN/BOX PLOT
        # ====================================================================
        
        st.subheader("3️⃣ Log2 Intensity Distribution by Sample")
        
        df_long = df_log2.select([id_col] + numeric_cols).melt(
            id_vars=[id_col],
            value_vars=numeric_cols,
            variable_name='sample',
            value_name='log2_intensity'
        ).filter(
            pl.col('log2_intensity').is_finite()
        )
        
        df_long = df_long.with_columns(
            pl.when(pl.col('sample').str.starts_with('A'))
            .then(pl.lit('A'))
            .otherwise(pl.lit('B'))
            .alias('condition')
        )
        
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
        
        df_stats = df_long.group_by('sample').agg([
            pl.col('log2_intensity').len().alias('n'),
            pl.col('log2_intensity').mean().alias('mean'),
            pl.col('log2_intensity').median().alias('median'),
            pl.col('log2_intensity').std().alias('std'),
            pl.col('log2_intensity').quantile(0.25).alias('q25'),
            pl.col('log2_intensity').quantile(0.75).alias('q75')
        ]).sort('sample')
        
        st.markdown("**Summary Statistics:**")
        st.dataframe(df_stats.to_pandas(), width='stretch')
        
        st.download_button(
            "📥 Download Statistics (CSV)",
            df_stats.write_csv(),
            "log2_intensity_statistics_peptide.csv",
            "text/csv",
            key="peptide_download_intensity_stats"
        )
        
        del df_long, df_stats
        gc.collect()
        
        st.markdown("---")
        
        # ====================================================================
        # 4. CV DISTRIBUTION
        # ====================================================================
        
        st.subheader("4️⃣ Coefficient of Variation (CV) by Condition")
        st.info("**CV = (std / mean) × 100** for each peptide across replicates. Lower CV = better reproducibility.")
        
        cv_result = compute_cv_data(df.to_dict(as_series=False), id_col, numeric_cols, 'peptide')
        df_cv_long = pl.DataFrame(cv_result['cv_data'])
        conditions = cv_result['conditions']
        
        high_cv_counts = df_cv_long.filter(pl.col('cv') > 100).group_by('condition').agg(
            pl.len().alias('n_high_cv')
        ).sort('condition')
        
        if high_cv_counts.shape[0] > 0:
            warning_text = "⚠️ **High CV (>100%) detected:**  "
            warning_parts = []
            for row in high_cv_counts.iter_rows(named=True):
                warning_parts.append(f"**{row['condition']}**: {row['n_high_cv']} peptides")
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
            pl.col('cv').len().alias('n_peptides'),
            pl.col('cv').mean().alias('mean_cv'),
            pl.col('cv').median().alias('median_cv'),
            pl.col('cv').std().alias('std_cv'),
            pl.col('cv').quantile(0.25).alias('q25'),
            pl.col('cv').quantile(0.75).alias('q75'),
            (pl.col('cv') > 100).sum().alias('n_cv_over_100')
        ]).sort('condition')
        
        st.markdown("**CV Summary Statistics:**")
        st.dataframe(df_cv_stats.to_pandas(), width='stretch')
        
        st.download_button(
            "📥 Download CV Statistics (CSV)",
            df_cv_stats.write_csv(),
            "cv_statistics_peptide.csv",
            "text/csv",
            key="peptide_download_cv_stats"
        )
        
        del df_cv_long, df_cv_plot, df_cv_stats, high_cv_counts
        gc.collect()
        
        st.markdown("---")
        
        # ====================================================================
        # 5. PEPTIDE COUNT BY CV THRESHOLD
        # ====================================================================
        
        st.subheader("5️⃣ Peptide Count by CV Threshold")
        st.info("**Quality tiers:** Total (all valid), CV < 20% (good+excellent), CV < 10% (excellent)")
        
        cv_plot_data = []
        
        for condition, cols in conditions.items():
            df_cv_cond = df.select([id_col] + cols).with_columns([
                pl.concat_list(cols).list.mean().alias('mean'),
                pl.concat_list(cols).list.std().alias('std')
            ]).with_columns(
                (pl.col('std') / pl.col('mean') * 100).alias('cv')
            ).filter(
                pl.col('cv').is_finite() & (pl.col('cv') > 0)
            )
            
            total = df_cv_cond.shape[0]
            cv_under_20 = df_cv_cond.filter(pl.col('cv') < 20).shape[0]
            cv_under_10 = df_cv_cond.filter(pl.col('cv') < 10).shape[0]
            
            cv_plot_data.append({'condition': condition, 'threshold': 'Total', 'count': total})
            cv_plot_data.append({'condition': condition, 'threshold': 'CV < 20%', 'count': cv_under_20})
            cv_plot_data.append({'condition': condition, 'threshold': 'CV < 10%', 'count': cv_under_10})
        
        df_cv_plot = pl.DataFrame(cv_plot_data)
        
        df_cv_plot = df_cv_plot.with_columns(
            pl.col('threshold').cast(pl.Categorical(ordering='physical'))
        )
        
        df_cv_plot_ordered = df_cv_plot.sort(['condition', 'threshold'], 
                                             descending=[True, True])
        
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
         labs(title='Peptide Count by CV Quality Threshold',
              x='Condition', y='Peptide Count', fill='Threshold') +
         theme_minimal() +
         theme(axis_text_x=element_text(size=12),
               figure_size=(10, 6)))
        
        fig = ggplot.draw(plot)
        st.pyplot(fig)
        plt.close(fig)
        del fig, plot
        
        st.markdown("**Peptide Counts by CV Threshold:**")
        
        summary_data = []
        for cond in sorted(conditions.keys()):
            cond_data = df_cv_plot.filter(pl.col('condition') == cond)
            total = cond_data.filter(pl.col('threshold') == 'Total')['count'][0]
            cv_lt_20 = cond_data.filter(pl.col('threshold') == 'CV < 20%')['count'][0]
            cv_lt_10 = cond_data.filter(pl.col('threshold') == 'CV < 10%')['count'][0]
            
            summary_data.append({
                'Condition': cond,
                'Total': total,
                'CV < 20%': cv_lt_20,
                'CV < 10%': cv_lt_10,
                '% < 20%': round(cv_lt_20 / total * 100, 1) if total > 0 else 0,
                '% < 10%': round(cv_lt_10 / total * 100, 1) if total > 0 else 0
            })
        
        df_summary = pl.DataFrame(summary_data)
        
        st.dataframe(df_summary.to_pandas(), width='stretch')
        
        st.download_button(
            "📥 Download CV Threshold Summary (CSV)",
            df_summary.write_csv(),
            "cv_threshold_summary_peptide.csv",
            "text/csv",
            key="peptide_download_cv_threshold"
        )
        
        del df_cv_plot, df_cv_plot_ordered, df_summary, cv_plot_data, summary_data
        gc.collect()
        
        st.markdown("---")
        
        # ====================================================================
        # 6. MISSING VALUES
        # ====================================================================
        
        st.subheader("6️⃣ Missing Values per Peptide by Condition")
        st.info("**Missing = intensity ≤ 1.0** (includes NaN, zero, and 1.0). Shows how many replicates are missing per peptide.")
        
        missing_plot_data = compute_missing_data(df.to_dict(as_series=False), id_col, numeric_cols, replicates)
        df_missing_plot = pl.DataFrame(missing_plot_data)
        
        plot = (ggplot(df_missing_plot.to_pandas(), aes(x='condition', y='count', fill='n_missing')) +
         geom_bar(stat='identity', position='dodge') +
         geom_text(aes(label='count'), position=position_dodge(width=0.9),
                   va='bottom', size=8, fontweight='bold') +
         scale_fill_brewer(type='seq', palette='YlOrRd') +
         labs(title='Peptide Count by Number of Missing Values per Condition',
              x='Condition', y='Peptide Count', fill='Missing Values') +
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
            columns='n_missing',
            values='count'
        ).fill_null(0)
        
        col_order = [f'{i} missing' for i in range(replicates + 1)]
        df_missing_summary = df_missing_summary.select(
            ['condition'] + [c for c in col_order if c in df_missing_summary.columns]
        ).with_columns([
            pl.sum_horizontal([c for c in col_order if c in df_missing_summary.columns]).alias('Total'),
            (pl.col('0 missing') / pl.sum_horizontal([c for c in col_order if c in df_missing_summary.columns]) * 100).alias('% Complete')
        ])
        
        st.dataframe(df_missing_summary.to_pandas(), width='stretch')
        
        st.download_button(
            "📥 Download Missing Values Summary (CSV)",
            df_missing_summary.write_csv(),
            "missing_values_summary_peptide.csv",
            "text/csv",
            key="peptide_download_missing"
        )
        
        del df_missing_plot, df_missing_summary, missing_plot_data
        gc.collect()
        
        st.markdown("---")
        
        del df_log2
        clear_plot_memory()

# ============================================================================
# NAVIGATION
# ============================================================================

clear_plot_memory()

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("← Back to Upload", width='stretch'):
        st.switch_page("pages/1_Data_Upload.py")

with col2:
    if st.button("Continue to Normalization →", type="primary", width='stretch'):
        st.switch_page("pages/3_Normalization.py")
