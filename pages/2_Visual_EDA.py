import streamlit as st
import pandas as pd
import numpy as np
from plotly import graph_objects as go
from helpers.core import ProteinData, get_theme, TransformCache
from helpers.viz import create_protein_count_stacked_bar, create_sample_boxplots

# ============================================================================
# LOAD DATA & CACHE
# ============================================================================

if "protein_data" not in st.session_state:
    st.warning("⚠️ Please upload data first")
    st.stop()

protein_data = st.session_state.protein_data

# Initialize transform cache in session (persists across reruns)
if "transform_cache" not in st.session_state:
    st.session_state.transform_cache = TransformCache()

cache = st.session_state.transform_cache

# Compute log2 if not cached
if not cache.has('log2'):
    df_log2 = protein_data.raw[protein_data.numeric_cols].copy()
    for col in protein_data.numeric_cols:
        df_log2[col] = np.log2(df_log2[col].clip(lower=0.1))
    cache.log2 = df_log2
else:
    df_log2 = cache.get('log2')

# Prepare viz data (1.0 → NaN) - lightweight, compute on demand
df_viz = protein_data.raw[protein_data.numeric_cols].copy()
for col in protein_data.numeric_cols:
    df_viz.loc[df_viz[col] == 1.0, col] = np.nan

# Get theme
theme = get_theme("light")

# ============================================================================
# PAGE CONTENT
# ============================================================================

st.title("📊 Visual EDA")

st.info(f"📁 **{protein_data.file_path}** | {protein_data.n_proteins:,} proteins × {protein_data.n_samples} samples")
st.markdown("---")

# ============================================================================
# SECTION 1: DATASET OVERVIEW
# ============================================================================

st.subheader("1️⃣ Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Proteins", f"{protein_data.n_proteins:,}")
c2.metric("Total Samples", protein_data.n_samples)
c3.metric("Species", len(set(protein_data.species_mapping.values())))
c4.metric("Avg/Species", int(protein_data.n_proteins / len(set(protein_data.species_mapping.values()))))

st.markdown("---")

# ============================================================================
# SECTION 2: VALID PROTEINS PER SPECIES
# ============================================================================

st.subheader("2️⃣ Valid Proteins per Species per Sample")

st.info("**Valid = intensity ≠ 1.00**. Each cell shows count of valid proteins.")

fig2, summary_df2 = create_protein_count_stacked_bar(
    df_viz,
    protein_data.numeric_cols,
    protein_data.species_mapping,
    theme
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("**Summary by Species:**")
st.dataframe(summary_df2, use_container_width=True, height=250)

st.download_button(
    label="📥 Download Summary (CSV)",
    data=summary_df2.to_csv(index=False),
    file_name="protein_counts_summary.csv",
    mime="text/csv"
)

st.markdown("---")

# ============================================================================
# SECTION 3: LOG2 INTENSITY DISTRIBUTIONS
# ============================================================================

st.subheader("3️⃣ Log2 Intensity Distribution by Sample")

st.info("**Single boxplot with 6 traces**: Condition A (A1-A3, green) vs Condition B (B1-B3, teal).")

fig3, df_stats3 = create_sample_boxplots(
    df_log2,
    protein_data.numeric_cols,
    theme
)

st.plotly_chart(fig3, use_container_width=True)

st.markdown("**Summary Statistics:**")
st.dataframe(df_stats3, use_container_width=True)

st.download_button(
    label="📥 Download Statistics (CSV)",
    data=df_stats3.to_csv(index=False),
    file_name="log2_intensity_statistics.csv",
    mime="text/csv"
)
