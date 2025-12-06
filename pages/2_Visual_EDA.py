"""
pages/2_Visual_EDA.py

Visual exploratory data analysis with cached computations
Leverages ProteinData dataclass and cached helper functions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from helpers.core import ProteinData, get_theme
from helpers.transforms import apply_transformation
from helpers.analysis import detect_conditions_from_columns, create_group_dict
from helpers.audit import log_event

# ============================================================================
# CACHED PLOT FUNCTIONS
# Compute-intensive visualizations cached to avoid recalculation
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="Computing protein counts...")
def compute_protein_counts_by_species(
    df: pd.DataFrame,
    numeric_cols: list,
    species_mapping: dict
) -> dict:
    """
    Count proteins per sample grouped by species.
    Cached to avoid recalculation on widget interactions.
    
    Returns:
        Dict[sample_name -> Dict[species -> count]]
    """
    protein_counts = {}
    
    for sample in numeric_cols:
        # Valid proteins: not NaN and not 0 after log2
        valid_proteins = df[df[sample].notna() & (df[sample] != 0.0)].index
        
        # Count by species
        species_counts = {}
        for protein_id in valid_proteins:
            species = species_mapping.get(protein_id, "UNKNOWN")
            if species:
                species_counts[species] = species_counts.get(species, 0) + 1
        
        protein_counts[sample] = species_counts
    
    return protein_counts

@st.cache_data(ttl=3600, show_spinner="Creating stacked bar plot...")
def create_protein_count_plot(
    protein_counts: dict,
    numeric_cols: list,
    theme: dict
) -> go.Figure:
    """
    Create stacked bar chart of protein counts by species.
    Uses precomputed counts for fast rendering.
    """
    # Get all unique species
    all_species = sorted(set(sp for counts in protein_counts.values() for sp in counts.keys()))
    
    fig = go.Figure()
    
    # Color mapping
    colors = {
        'HUMAN': theme['color_human'],
        'YEAST': theme['color_yeast'],
        'ECOLI': theme['color_ecoli'],
        'MOUSE': '#9467bd',
        'UNKNOWN': '#999999'
    }
    
    for species in all_species:
        counts = [protein_counts[sample].get(species, 0) for sample in numeric_cols]
        total_count = sum(counts)
        
        fig.add_trace(go.Bar(
            name=f"{species} (n={total_count})",
            x=numeric_cols,
            y=counts,
            marker_color=colors.get(species, '#cccccc'),
            hovertemplate=f"<b>{species}</b><br>Sample: %{{x}}<br>Proteins: %{{y}}<extra></extra>"
        ))
    
    fig.update_layout(
        barmode='stack',
        title="Protein Counts per Sample",
        xaxis_title="Sample",
        yaxis_title="Number of Proteins",
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['paper_bg'],
        font=dict(family="Arial", size=14, color=theme['text_primary']),
        showlegend=True,
        legend=dict(
            title="Species",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        height=500,
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=theme['grid'], tickangle=-45)
    fig.update_yaxes(showgrid=True, gridcolor=theme['grid'])
    
    return fig

@st.cache_data(ttl=3600, show_spinner="Creating box plots...")
def create_condition_boxplot(
    df_log2: pd.DataFrame,
    condition_dict: dict,
    conditions_to_plot: list,
    theme: dict
) -> go.Figure:
    """
    Create box plots grouped by condition (A and B only).
    Cached for fast rendering on page interactions.
    """
    fig = go.Figure()
    
    # Color mapping for conditions
    condition_colors = {
        conditions_to_plot[0]: theme['color_human'],
        conditions_to_plot[1]: theme['color_yeast']
    }
    
    for cond in conditions_to_plot:
        samples = condition_dict[cond]
        
        for sample in samples:
            # Get valid log2 values (not NaN, not 0)
            values = df_log2[sample].dropna()
            values = values[values != 0.0]
            
            fig.add_trace(go.Box(
                y=values,
                name=sample,
                marker_color=condition_colors[cond],
                legendgroup=cond,
                legendgrouptitle_text=f"Condition {cond}",
                boxmean='sd',  # Show mean and SD
                hovertemplate=f"<b>{sample}</b><br>Intensity: %{{y:.2f}}<extra></extra>"
            ))
    
    fig.update_layout(
        title="Log2 Intensity Distribution by Condition",
        xaxis_title="Sample",
        yaxis_title="Log2 Intensity",
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['paper_bg'],
        font=dict(family="Arial", size=14, color=theme['text_primary']),
        showlegend=True,
        height=600
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=theme['grid'], tickangle=-45)
    fig.update_yaxes(showgrid=True, gridcolor=theme['grid'])
    
    return fig

# ============================================================================
# CONTROL FUNCTIONS
# ============================================================================

def restart_pipeline():
    """Clear all session state and return to upload."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.switch_page("pages/1_Data_Upload.py")

def reset_eda():
    """Clear EDA cache only."""
    keys = ['eda_log2_data', 'eda_conditions']
    for k in keys:
        st.session_state.pop(k, None)
    st.cache_data.clear()
    st.rerun()

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="Visual EDA", layout="wide")

with st.sidebar:
    st.title("⚙️ Options")
    if st.button("🔄 Reset This Page", use_container_width=True):
        reset_eda()
    st.markdown("---")
    if st.button("🏠 Restart Pipeline", use_container_width=True):
        restart_pipeline()

# ============================================================================
# LOAD DATA FROM SESSION STATE
# ============================================================================

st.title("📊 Visual EDA")

if "protein_data" not in st.session_state:
    st.warning("⚠️ No data loaded")
    if st.button("← Go to Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

# Load ProteinData dataclass from session state
protein_data: ProteinData = st.session_state.protein_data

# Extract cached properties (no recalculation!)
df_raw = protein_data.raw
numeric_cols = protein_data.numeric_cols
species_mapping = protein_data.species_mapping
n_proteins = protein_data.n_proteins  # Property, not recalculated
n_samples = protein_data.n_samples    # Property

theme = get_theme(st.session_state.get("theme", "dark"))

st.info(f"📁 **{protein_data.file_path}** | {n_proteins:,} proteins × {n_samples} samples")

st.markdown("---")

# ============================================================================
# APPLY LOG2 TRANSFORMATION (CACHED)
# ============================================================================

# Check if already computed
if 'eda_log2_data' not in st.session_state:
    with st.spinner("Applying log2 transformation..."):
        # apply_transformation already has @st.cache_data!
        df_log2, trans_cols = apply_transformation(df_raw, numeric_cols, method="log2")
        st.session_state.eda_log2_data = df_log2
        st.session_state.eda_trans_cols = trans_cols
else:
    df_log2 = st.session_state.eda_log2_data
    trans_cols = st.session_state.eda_trans_cols

# ============================================================================
# DETECT CONDITIONS (CACHED IN SESSION)
# ============================================================================

if 'eda_conditions' not in st.session_state:
    conditions = detect_conditions_from_columns(numeric_cols)  # Fast, no caching needed
    condition_dict = create_group_dict(numeric_cols, conditions)
    st.session_state.eda_conditions = condition_dict
else:
    condition_dict = st.session_state.eda_conditions

# ============================================================================
# PLOT 1: PROTEIN COUNTS PER SAMPLE (STACKED BY SPECIES)
# ============================================================================

st.header("1️⃣ Protein Counts per Sample")

st.markdown("Number of quantified proteins in each sample, stacked by species.")

# Compute counts (cached)
protein_counts = compute_protein_counts_by_species(df_log2, numeric_cols, species_mapping)

# Create plot (cached)
fig1 = create_protein_count_plot(protein_counts, numeric_cols, theme)
st.plotly_chart(fig1, use_container_width=True)

# Summary table
st.markdown("**Summary by Species:**")

all_species = sorted(set(sp for counts in protein_counts.values() for sp in counts.keys()))
summary_data = []

for species in all_species:
    total = sum(protein_counts[sample].get(species, 0) for sample in numeric_cols)
    avg = total / len(numeric_cols)
    min_count = min(protein_counts[sample].get(species, 0) for sample in numeric_cols)
    max_count = max(protein_counts[sample].get(species, 0) for sample in numeric_cols)
    
    summary_data.append({
        'Species': species,
        'Total': total,
        'Avg/Sample': f"{avg:.1f}",
        'Min': min_count,
        'Max': max_count
    })

st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

st.markdown("---")

# ============================================================================
# PLOT 2: BOXPLOTS BY CONDITION (A vs B)
# ============================================================================

st.header("2️⃣ Log2 Intensity Distribution by Condition")

st.markdown("Box plots showing log2-transformed intensities for conditions A and B.")

if condition_dict and len(condition_dict) >= 2:
    
    # Get first two conditions
    conditions_to_plot = sorted(condition_dict.keys())[:2]
    
    # Create plot (cached)
    fig2 = create_condition_boxplot(df_log2, condition_dict, conditions_to_plot, theme)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Summary statistics table
    st.markdown("**Summary Statistics by Condition:**")
    
    summary_stats = []
    for cond in conditions_to_plot:
        samples = condition_dict[cond]
        
        # Concatenate all values for this condition
        vals = np.concatenate([
            df_log2[s].dropna().values[df_log2[s].dropna() != 0.0]
            for s in samples
        ])
        
        summary_stats.append({
            'Condition': cond,
            'N Samples': len(samples),
            'Mean': f"{np.mean(vals):.2f}",
            'Median': f"{np.median(vals):.2f}",
            'Std Dev': f"{np.std(vals):.2f}",
            'Min': f"{np.min(vals):.2f}",
            'Max': f"{np.max(vals):.2f}"
        })
    
    st.dataframe(pd.DataFrame(summary_stats), use_container_width=True, hide_index=True)
    
else:
    st.warning("⚠️ Need at least 2 conditions for comparison")

# ============================================================================
# LOG EVENT (ONE TIME)
# ============================================================================

if 'eda_logged' not in st.session_state:
    log_event(
        "Visual EDA",
        "Page viewed",
        {
            "n_proteins": n_proteins,
            "n_samples": n_samples,
            "n_species": len(all_species)
        }
    )
    st.session_state.eda_logged = True

# ============================================================================
# NAVIGATION
# ============================================================================

st.markdown("---")

c1, c2, c3, c4 = st.columns(4)

with c1:
    if st.button("← Upload", use_container_width=True):
        st.switch_page("pages/1_Data_Upload.py")

with c2:
    if st.button("Statistical EDA →", use_container_width=True, type="primary"):
        st.switch_page("pages/3_Statistical_EDA.py")

with c3:
    if st.button("🔄 Reset", use_container_width=True):
        reset_eda()

with c4:
    if st.button("🏠 Restart", use_container_width=True):
        restart_pipeline()

st.caption("✅ All computations cached | Fast reruns on widget interactions")
