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

# Initialize transform cache in session
if "transform_cache" not in st.session_state:
    st.session_state.transform_cache = TransformCache()
elif not isinstance(st.session_state.transform_cache, TransformCache):
    # Handle legacy dict cache
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

# Prepare viz data (1.0 → NaN)
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

fig2, _ = create_protein_count_stacked_bar(
    df_viz,
    protein_data.numeric_cols,
    protein_data.species_mapping,
    theme
)

st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# BUILD DETAILED TABLE (Species × Samples + Total)
# ============================================================================

unique_counts_table = {}
unique_species = sorted(set(protein_data.species_mapping.values()))

for species in unique_species:
    unique_counts_table[species] = {}
    
    # Count unique proteins per species per sample
    for sample in protein_data.numeric_cols:
        valid_mask = (df_viz[sample].notna()) & (df_viz[sample] != 0.0)
        species_proteins = df_viz.index[valid_mask][
            df_viz.index[valid_mask].map(lambda x: protein_data.species_mapping.get(x) == species)
        ]
        unique_counts_table[species][sample] = len(species_proteins)
    
    # Total unique proteins for this species
    species_all_proteins = [
        pid for pid, sp in protein_data.species_mapping.items() if sp == species
    ]
    total_valid = sum(
        1 for pid in species_all_proteins 
        if (df_viz.loc[pid] != 0.0).any() and df_viz.loc[pid].notna().any()
    )
    unique_counts_table[species]['Total'] = total_valid

# Convert to DataFrame (sorted by total descending)
df_table = pd.DataFrame(unique_counts_table).T
df_table = df_table.sort_values('Total', ascending=False)

# Add row totals
df_table.loc['Total'] = df_table.sum()

# Display table
st.markdown("**Unique Proteins per Species per Sample:**")
st.dataframe(df_table, use_container_width=True)

# Download
st.download_button(
    label="📥 Download Table (CSV)",
    data=df_table.to_csv(),
    file_name="unique_proteins_per_species.csv",
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
