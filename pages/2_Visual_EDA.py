"""
pages/2_Visual_EDA.py

Visual Exploratory Data Analysis
- Data quality assessment
- Per-species protein counts
- Missing value distribution analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from helpers.core import ProteinData
from helpers.analysis import (
    count_valid_proteins_per_species_sample,
    count_missing_per_protein,
    count_proteins_by_species
)
from helpers.audit import log_event
from helpers.viz import create_protein_count_stacked_bar
from helpers.core import get_theme
# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Visual EDA", layout="wide")

# ============================================================================
# CHECK DATA LOADED
# ============================================================================

if "protein_data" not in st.session_state:
    st.warning("⚠️ No data loaded")
    if st.button("← Go to Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

protein_data: ProteinData = st.session_state.protein_data
df_raw = protein_data.raw
numeric_cols = protein_data.numeric_cols
species_mapping = protein_data.species_mapping


st.info(f"📁 **{protein_data.file_path}** | {protein_data.n_proteins:,} proteins × {protein_data.n_samples} samples")
st.markdown("---")

st.title("📊 Visual Exploratory Data Analysis")


# ============================================================================
# SECTION 2: VALID PROTEINS PER SPECIES PER REPLICATE
# ============================================================================

st.subheader("2️⃣ Valid Proteins per Species per Sample")

st.info("**Valid = intensity ≠ 1.00**. Unique protein counts per sample by species (sorted by total).")

# Prepare data for viz helper: convert missing values (1.0) to NaN
df_for_viz = protein_data.raw[protein_data.numeric_cols].copy()

for col in protein_data.numeric_cols:
    df_for_viz.loc[df_for_viz[col] == 1.0, col] = np.nan

# Create detailed table first to get totals for sorting
unique_counts_table = {}
unique_species = sorted(set(protein_data.species_mapping.values()))

for species in unique_species:
    unique_counts_table[species] = {}
    
    # Count unique proteins per species per sample
    for sample in protein_data.numeric_cols:
        valid_mask = (df_for_viz[sample].notna()) & (df_for_viz[sample] != 0.0)
        species_proteins = df_for_viz.index[valid_mask][
            df_for_viz.index[valid_mask].map(lambda x: protein_data.species_mapping.get(x) == species)
        ]
        unique_counts_table[species][sample] = len(species_proteins)
    
    # Total unique proteins for this species
    species_all_proteins = [
        pid for pid, sp in protein_data.species_mapping.items() if sp == species
    ]
    total_valid = sum(
        1 for pid in species_all_proteins 
        if (df_for_viz.loc[pid] != 0.0).any() and df_for_viz.loc[pid].notna().any()
    )
    unique_counts_table[species]['Total'] = total_valid

# Sort species by total (descending: most proteins first)
df_table = pd.DataFrame(unique_counts_table).T
df_table = df_table.sort_values('Total', ascending=False)
sorted_species = df_table.index.tolist()

# Add row totals
df_table.loc['Total'] = df_table.sum()

# Create stacked bar with sorted species (top to bottom ascending)
from helpers.core import get_theme, FONT_FAMILY
theme = get_theme("light")

fig = go.Figure()

colors = {
    'HUMAN': theme['color_human'],
    'YEAST': theme['color_yeast'],
    'ECOLI': theme['color_ecoli'],
    'MOUSE': '#9467bd',
    'UNKNOWN': '#999999'
}

# Add in reverse order (largest FIRST for bottom of stack)
for species in reversed(sorted_species):
    if species == 'Total':
        continue
    
    counts = [unique_counts_table[species][sample] for sample in protein_data.numeric_cols]
    total = unique_counts_table[species]['Total']
    
    fig.add_trace(go.Bar(
        name=f"{species} (Total: {total})",
        x=protein_data.numeric_cols,
        y=counts,
        marker_color=colors.get(species, '#cccccc'),
        hovertemplate=f"<b>{species}</b><br>Sample: %{{x}}<br>Proteins: %{{y}}<extra></extra>"
    ))

fig.update_layout(
    barmode='stack',
    title="Valid Proteins per Sample (Stacked)",
    xaxis_title="Sample",
    yaxis_title="Number of Proteins",
    plot_bgcolor=theme['bg_primary'],
    paper_bgcolor=theme['paper_bg'],
    font=dict(family=FONT_FAMILY, size=14, color=theme['text_primary']),
    showlegend=True,
    legend=dict(title="Species", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    height=500,
    hovermode='x unified'
)

fig.update_xaxes(showgrid=True, gridcolor=theme['grid'], tickangle=-45)
fig.update_yaxes(showgrid=True, gridcolor=theme['grid'])

# Display stacked bar chart
st.plotly_chart(fig, use_container_width=True)

# Display table (sorted by Total descending)
st.markdown("**Unique Proteins per Species per Sample:**")
st.dataframe(df_table, use_container_width=True)

# Download
col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="📥 Download Table (CSV)",
        data=df_table.to_csv(),
        file_name="unique_proteins_per_species.csv",
        mime="text/csv"
    )

with col2:
    st.info("✅ Table sorted: most proteins at top, least at bottom (ascending)")



# ============================================================================
# Navigation
# ============================================================================

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("← Back to Upload", use_container_width=True):
        st.switch_page("pages/1_Data_Upload.py")

with col2:
    st.info("**Next:** Statistical transformation & differential expression analysis (coming soon)")
