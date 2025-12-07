"""
pages/2_Visual_EDA.py
Exploratory data analysis for protein and/or peptide data - MEMORY OPTIMIZED ONLY
"""

import streamlit as st
import polars as pl
import polars.selectors as cs
import numpy as np
import pandas as pd
from plotnine import *
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

def sample_data_for_plot(df: pl.DataFrame, max_rows: int = 5000) -> pl.DataFrame:
    """Sample dataframe if it's too large for plotting."""
    if df.shape[0] > max_rows:
        return df.sample(n=max_rows, seed=42)
    return df

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
# CREATE TABS BASED ON AVAILABLE DATA
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
        
        st.header("🧬 Protein-Level Analysis")
        
        st.info(f"""
        **Dataset:** {df.shape[0]:,} proteins × {len(numeric_cols)} samples
        """)
        
        # Sample for plotting
        df_plot = sample_data_for_plot(df)
        
        # ====================================================================
        # 1. DISTRIBUTION OVERVIEW
        # ====================================================================
        
        st.subheader("1️⃣ Intensity Distributions")
        
        df_melted = df_plot.select([id_col] + numeric_cols).melt(
            id_vars=[id_col],
            value_vars=numeric_cols,
            variable_name='sample',
            value_name='intensity'
        ).filter(
            pl.col('intensity').is_finite() & (pl.col('intensity') > 0)
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            plot_hist = (ggplot(df_melted.to_pandas(), aes(x='intensity', fill='sample')) +
             geom_histogram(bins=50, alpha=0.6, position='identity') +
             labs(title='Raw Intensity Distribution', x='Intensity', y='Count') +
             theme_minimal() +
             theme(figure_size=(6, 4), legend_position='none'))
            
            fig = ggplot.draw(plot_hist)
            st.pyplot(fig)
            plt.close(fig)
            del fig, plot_hist
        
        with col2:
            df_log = df_melted.with_columns(
                pl.col('intensity').log(2).alias('log2_intensity')
            )
            
            plot_log = (ggplot(df_log.to_pandas(), aes(x='log2_intensity', fill='sample')) +
             geom_histogram(bins=50, alpha=0.6, position='identity') +
             labs(title='Log2 Intensity Distribution', x='Log2(Intensity)', y='Count') +
             theme_minimal() +
             theme(figure_size=(6, 4), legend_position='none'))
            
            fig = ggplot.draw(plot_log)
            st.pyplot(fig)
            plt.close(fig)
            del fig, plot_log
        
        del df_melted
        gc.collect()
        
        st.markdown("---")
        
        # ====================================================================
        # 2. BOXPLOTS
        # ====================================================================
        
        st.subheader("2️⃣ Sample Comparison")
        
        plot_box = (ggplot(df_log.to_pandas(), aes(x='sample', y='log2_intensity', fill='sample')) +
         geom_boxplot(outlier_alpha=0.2) +
         labs(title='Log2 Intensity by Sample', x='Sample', y='Log2(Intensity)') +
         theme_minimal() +
         theme(figure_size=(12, 5), axis_text_x=element_text(rotation=45, hjust=1), legend_position='none'))
        
        fig = ggplot.draw(plot_box)
        st.pyplot(fig)
        plt.close(fig)
        del fig, plot_box, df_log
        clear_plot_memory()
        
        st.markdown("---")
        
        # ====================================================================
        # 3. MISSING DATA HEATMAP
        # ====================================================================
        
        st.subheader("3️⃣ Missing Data Pattern")
        
        missing_data = []
        for col in numeric_cols:
            n_missing = df.filter((pl.col(col) == 1.0) | pl.col(col).is_null()).shape[0]
            pct_missing = n_missing / df.shape[0] * 100
            missing_data.append({
                'Sample': col,
                'Missing (%)': pct_missing,
                'Valid': df.shape[0] - n_missing
            })
        
        df_missing = pl.DataFrame(missing_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            plot_missing = (ggplot(df_missing.to_pandas(), aes(x='Sample', y='Missing (%)')) +
             geom_col(fill='#e74c3c', alpha=0.7) +
             geom_hline(yintercept=20, linetype='dashed', color='gray') +
             labs(title='Missing Data by Sample', x='Sample', y='Missing (%)') +
             theme_minimal() +
             theme(figure_size=(8, 4), axis_text_x=element_text(rotation=45, hjust=1)))
            
            fig = ggplot.draw(plot_missing)
            st.pyplot(fig)
            plt.close(fig)
            del fig, plot_missing
        
        with col2:
            st.dataframe(
                df_missing.to_pandas().style.format({'Missing (%)': '{:.1f}%'}),
                hide_index=True,
                width='stretch',
                height=300
            )
        
        del df_missing, missing_data
        clear_plot_memory()
        
        st.markdown("---")
        
        # ====================================================================
        # 4. SPECIES BREAKDOWN
        # ====================================================================
        
        if species_col:
            st.subheader("4️⃣ Species Composition")
            
            species_counts = df.group_by(species_col).agg(
                pl.len().alias('count')
            ).sort('count', descending=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                plot_species = (ggplot(species_counts.to_pandas(), aes(x='reorder(' + species_col + ', count)', y='count', fill=species_col)) +
                 geom_col() +
                 coord_flip() +
                 labs(title='Proteins by Species', x='Species', y='Count') +
                 theme_minimal() +
                 theme(figure_size=(6, 4), legend_position='none'))
                
                fig = ggplot.draw(plot_species)
                st.pyplot(fig)
                plt.close(fig)
                del fig, plot_species
            
            with col2:
                st.dataframe(
                    species_counts.to_pandas(),
                    hide_index=True,
                    width='stretch',
                    height=300
                )
            
            del species_counts
            clear_plot_memory()
            
            st.markdown("---")
        
        # ====================================================================
        # 5. CORRELATION HEATMAP
        # ====================================================================
        
        st.subheader("5️⃣ Sample Correlation")
        
        corr_df = df.select(numeric_cols).fill_null(1.0)
        corr_matrix = np.corrcoef(corr_df.to_numpy().T)
        
        corr_data = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                corr_data.append({
                    'Sample1': col1,
                    'Sample2': col2,
                    'Correlation': corr_matrix[i, j]
                })
        
        df_corr = pl.DataFrame(corr_data)
        
        plot_corr = (ggplot(df_corr.to_pandas(), aes(x='Sample1', y='Sample2', fill='Correlation')) +
         geom_tile() +
         scale_fill_gradient2(low='blue', mid='white', high='red', midpoint=0.9, limits=[0.5, 1]) +
         labs(title='Sample Correlation Matrix', x='', y='') +
         theme_minimal() +
         theme(figure_size=(10, 8), axis_text_x=element_text(rotation=45, hjust=1)))
        
        fig = ggplot.draw(plot_corr)
        st.pyplot(fig)
        plt.close(fig)
        
        del fig, plot_corr, df_corr, corr_data, corr_df, corr_matrix, df_plot
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
        sequence_col = st.session_state.get('peptide_sequence_col')
        
        st.header("🔬 Peptide-Level Analysis")
        
        st.info(f"""
        **Dataset:** {df.shape[0]:,} peptides × {len(numeric_cols)} samples
        """)
        
        df_plot = sample_data_for_plot(df)
        
        # ====================================================================
        # 1. DISTRIBUTION OVERVIEW
        # ====================================================================
        
        st.subheader("1️⃣ Intensity Distributions")
        
        df_melted = df_plot.select([id_col] + numeric_cols).melt(
            id_vars=[id_col],
            value_vars=numeric_cols,
            variable_name='sample',
            value_name='intensity'
        ).filter(
            pl.col('intensity').is_finite() & (pl.col('intensity') > 0)
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            plot_hist = (ggplot(df_melted.to_pandas(), aes(x='intensity', fill='sample')) +
             geom_histogram(bins=50, alpha=0.6, position='identity') +
             labs(title='Raw Intensity Distribution', x='Intensity', y='Count') +
             theme_minimal() +
             theme(figure_size=(6, 4), legend_position='none'))
            
            fig = ggplot.draw(plot_hist)
            st.pyplot(fig)
            plt.close(fig)
            del fig, plot_hist
        
        with col2:
            df_log = df_melted.with_columns(
                pl.col('intensity').log(2).alias('log2_intensity')
            )
            
            plot_log = (ggplot(df_log.to_pandas(), aes(x='log2_intensity', fill='sample')) +
             geom_histogram(bins=50, alpha=0.6, position='identity') +
             labs(title='Log2 Intensity Distribution', x='Log2(Intensity)', y='Count') +
             theme_minimal() +
             theme(figure_size=(6, 4), legend_position='none'))
            
            fig = ggplot.draw(plot_log)
            st.pyplot(fig)
            plt.close(fig)
            del fig, plot_log
        
        del df_melted
        gc.collect()
        
        st.markdown("---")
        
        # ====================================================================
        # 2. BOXPLOTS
        # ====================================================================
        
        st.subheader("2️⃣ Sample Comparison")
        
        plot_box = (ggplot(df_log.to_pandas(), aes(x='sample', y='log2_intensity', fill='sample')) +
         geom_boxplot(outlier_alpha=0.2) +
         labs(title='Log2 Intensity by Sample', x='Sample', y='Log2(Intensity)') +
         theme_minimal() +
         theme(figure_size=(12, 5), axis_text_x=element_text(rotation=45, hjust=1), legend_position='none'))
        
        fig = ggplot.draw(plot_box)
        st.pyplot(fig)
        plt.close(fig)
        del fig, plot_box, df_log
        clear_plot_memory()
        
        st.markdown("---")
        
        # ====================================================================
        # 3. PEPTIDE-SPECIFIC: Sequence Length Distribution
        # ====================================================================
        
        if sequence_col:
            st.subheader("3️⃣ Peptide Sequence Analysis")
            
            df_seq = df.with_columns(
                pl.col(sequence_col).str.len_chars().alias('seq_length')
            )
            
            df_seq_sample = sample_data_for_plot(df_seq)
            
            col1, col2 = st.columns(2)
            
            with col1:
                plot_length = (ggplot(df_seq_sample.to_pandas(), aes(x='seq_length')) +
                 geom_histogram(bins=30, fill='#3498db', alpha=0.7) +
                 labs(title='Peptide Length Distribution', x='Sequence Length (AA)', y='Count') +
                 theme_minimal() +
                 theme(figure_size=(6, 4)))
                
                fig = ggplot.draw(plot_length)
                st.pyplot(fig)
                plt.close(fig)
                del fig, plot_length, df_seq_sample
            
            with col2:
                peptides_per_protein = df.group_by(id_col).agg(
                    pl.len().alias('n_peptides')
                ).sort('n_peptides', descending=True)
                
                st.metric("Unique Proteins", f"{peptides_per_protein.shape[0]:,}")
                st.metric("Avg Peptides/Protein", f"{peptides_per_protein['n_peptides'].mean():.1f}")
                st.metric("Max Peptides/Protein", f"{peptides_per_protein['n_peptides'].max()}")
                
                with st.expander("Top 10 Proteins by Peptide Count"):
                    st.dataframe(
                        peptides_per_protein.head(10).to_pandas(),
                        hide_index=True,
                        width='stretch'
                    )
                
                del peptides_per_protein
            
            del df_seq
            clear_plot_memory()
            
            st.markdown("---")
        
        # ====================================================================
        # 4. MISSING DATA
        # ====================================================================
        
        st.subheader("4️⃣ Missing Data Pattern")
        
        missing_data = []
        for col in numeric_cols:
            n_missing = df.filter((pl.col(col) == 1.0) | pl.col(col).is_null()).shape[0]
            pct_missing = n_missing / df.shape[0] * 100
            missing_data.append({
                'Sample': col,
                'Missing (%)': pct_missing,
                'Valid': df.shape[0] - n_missing
            })
        
        df_missing = pl.DataFrame(missing_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            plot_missing = (ggplot(df_missing.to_pandas(), aes(x='Sample', y='Missing (%)')) +
             geom_col(fill='#e74c3c', alpha=0.7) +
             geom_hline(yintercept=20, linetype='dashed', color='gray') +
             labs(title='Missing Data by Sample', x='Sample', y='Missing (%)') +
             theme_minimal() +
             theme(figure_size=(8, 4), axis_text_x=element_text(rotation=45, hjust=1)))
            
            fig = ggplot.draw(plot_missing)
            st.pyplot(fig)
            plt.close(fig)
            del fig, plot_missing
        
        with col2:
            st.dataframe(
                df_missing.to_pandas().style.format({'Missing (%)': '{:.1f}%'}),
                hide_index=True,
                width='stretch',
                height=300
            )
        
        del df_missing, missing_data
        clear_plot_memory()
        
        st.markdown("---")
        
        # ====================================================================
        # 5. SPECIES BREAKDOWN
        # ====================================================================
        
        if species_col:
            st.subheader("5️⃣ Species Composition")
            
            species_counts = df.group_by(species_col).agg(
                pl.len().alias('count')
            ).sort('count', descending=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                plot_species = (ggplot(species_counts.to_pandas(), aes(x='reorder(' + species_col + ', count)', y='count', fill=species_col)) +
                 geom_col() +
                 coord_flip() +
                 labs(title='Peptides by Species', x='Species', y='Count') +
                 theme_minimal() +
                 theme(figure_size=(6, 4), legend_position='none'))
                
                fig = ggplot.draw(plot_species)
                st.pyplot(fig)
                plt.close(fig)
                del fig, plot_species
            
            with col2:
                st.dataframe(
                    species_counts.to_pandas(),
                    hide_index=True,
                    width='stretch',
                    height=300
                )
            
            del species_counts
            clear_plot_memory()
        
        del df_plot
        gc.collect()

# ============================================================================
# FINAL CLEANUP
# ============================================================================

clear_plot_memory()

# ============================================================================
# NAVIGATION
# ============================================================================

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("← Back to Upload", width='stretch'):
        st.switch_page("pages/1_Data_Upload.py")

with col2:
    if st.button("Continue to Normalization →", type="primary", width='stretch'):
        st.switch_page("pages/3_Normalization.py")
