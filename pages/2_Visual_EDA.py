"""
pages/2_Visual_EDA.py
Exploratory data analysis - SIMPLIFIED
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
        replicates = st.session_state.protein_replicates
        
        st.header("🧬 Protein-Level Analysis")
        
        st.info(f"""
        **Dataset:** {df.shape[0]:,} proteins × {len(numeric_cols)} samples
        """)
        
        # Sample for plotting
        df_plot = sample_data_for_plot(df)
        
        # ====================================================================
        # BOXPLOT BY SAMPLE - LOG2 INTENSITIES
        # ====================================================================
        
        st.subheader("Log2 Intensity Distribution by Sample")
        
        # Prepare data: melt and add condition
        df_melted = df_plot.select([id_col] + numeric_cols).melt(
            id_vars=[id_col],
            value_vars=numeric_cols,
            variable_name='sample',
            value_name='intensity'
        ).filter(
            pl.col('intensity').is_finite() & (pl.col('intensity') > 0)
        ).with_columns([
            pl.col('intensity').log(2).alias('log2_intensity'),
            pl.col('sample').str.slice(0, 1).alias('condition')  # Extract condition (A, B, etc.)
        ])
        
        # Create boxplot
        plot_box = (ggplot(df_melted.to_pandas(), aes(x='sample', y='log2_intensity', fill='condition')) +
         geom_boxplot(outlier_size=1, outlier_alpha=0.3) +
         labs(title='Log2 Intensity by Sample', x='Sample', y='Log2(Intensity)', fill='Condition') +
         theme_minimal() +
         theme(figure_size=(12, 8), axis_text_x=element_text(rotation=45, hjust=1)))
        
        fig = ggplot.draw(plot_box)
        st.pyplot(fig)
        plt.close(fig)
        
        # Cleanup
        del fig, plot_box, df_melted, df_plot
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
        
        st.info(f"""
        **Dataset:** {df.shape[0]:,} peptides × {len(numeric_cols)} samples
        """)
        
        # Sample for plotting
        df_plot = sample_data_for_plot(df)
        
        # ====================================================================
        # BOXPLOT BY SAMPLE - LOG2 INTENSITIES
        # ====================================================================
        
        st.subheader("Log2 Intensity Distribution by Sample")
        
        # Prepare data: melt and add condition
        df_melted = df_plot.select([id_col] + numeric_cols).melt(
            id_vars=[id_col],
            value_vars=numeric_cols,
            variable_name='sample',
            value_name='intensity'
        ).filter(
            pl.col('intensity').is_finite() & (pl.col('intensity') > 0)
        ).with_columns([
            pl.col('intensity').log(2).alias('log2_intensity'),
            pl.col('sample').str.slice(0, 1).alias('condition')  # Extract condition (A, B, etc.)
        ])
        
        # Create boxplot
        plot_box = (ggplot(df_melted.to_pandas(), aes(x='sample', y='log2_intensity', fill='condition')) +
         geom_boxplot(outlier_size=1, outlier_alpha=0.3) +
         labs(title='Log2 Intensity by Sample', x='Sample', y='Log2(Intensity)', fill='Condition') +
         theme_minimal() +
         theme(figure_size=(12, 8), axis_text_x=element_text(rotation=45, hjust=1)))
        
        fig = ggplot.draw(plot_box)
        st.pyplot(fig)
        plt.close(fig)
        
        # Cleanup
        del fig, plot_box, df_melted, df_plot
        clear_plot_memory()

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
