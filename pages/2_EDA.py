import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Thermo Fisher Brand Colors ---
COLORS = {
    "red": "#E71316",
    "dark_red": "#A6192E",
    "gray": "#54585A",
    "light_gray": "#E2E3E4",
    "white": "#FFFFFF",
    "navy": "#262262",
    "green": "#B5BD00",
    "orange": "#EA7600",
    "dark_bg": "#1E1E1E",
    "dark_surface": "#2D2D2D",
    "dark_text": "#E2E3E4",
}

CHART_COLORS = ["#262262", "#A6192E", "#EA7600", "#F1B434", "#B5BD00", "#9BD3DD"]

st.set_page_config(
    page_title="EDA | Thermo Fisher Scientific",
    page_icon="📊",
    layout="wide",
)

# --- Adaptive CSS ---
st.markdown(f"""
<style>
    .header-banner {{
        background: linear-gradient(90deg, {COLORS['red']} 0%, {COLORS['dark_red']} 100%);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 30px;
    }}
    .header-banner h1 {{
        color: white;
        margin: 0;
        font-size: 28pt;
    }}
    .header-banner p {{
        color: white;
        margin: 5px 0 0 0;
        opacity: 0.9;
    }}
</style>
""", unsafe_allow_html=True)


def render_header(title: str, subtitle: str):
    st.markdown(f"""
    <div class="header-banner">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def parse_protein_group(pg_str: str) -> str:
    """Take first protein group if multiple separated by ;"""
    if pd.isna(pg_str):
        return "Unknown"
    return str(pg_str).split(";")[0].strip()


def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    """Get numeric column names."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def sort_columns_by_condition(cols: list[str]) -> list[str]:
    """Sort columns by condition (prefix before _) then replicate."""
    def sort_key(col):
        if "_" in col:
            parts = col.rsplit("_", 1)
            condition = parts[0]
            rep = int(parts[1]) if parts[1].isdigit() else 0
            return (condition, rep)
        return (col, 0)
    return sorted(cols, key=sort_key)


def create_missing_heatmap(df: pd.DataFrame, missing_mask: pd.DataFrame, 
                           index_col: str, numeric_cols: list[str]) -> go.Figure:
    """Create missing value heatmap."""
    # Prepare index labels
    if index_col and index_col in df.columns:
        labels = df[index_col].apply(parse_protein_group).tolist()
    else:
        labels = [f"Row {i}" for i in range(len(df))]
    
    # Sort columns by condition
    sorted_cols = sort_columns_by_condition(numeric_cols)
    mask_data = missing_mask[sorted_cols].astype(int)
    
    fig = go.Figure(data=go.Heatmap(
        z=mask_data.values,
        x=sorted_cols,
        y=labels,
        colorscale=[[0, COLORS["light_gray"]], [1, COLORS["red"]]],
        showscale=True,
        colorbar=dict(
            title="Missing",
            tickvals=[0, 1],
            ticktext=["Present", "Missing"]
        ),
        hovertemplate="Protein: %{y}<br>Sample: %{x}<br>Status: %{z}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Missing value distribution",
        xaxis_title="Samples",
        yaxis_title="Protein groups",
        height=max(400, len(labels) * 2),
        yaxis=dict(tickfont=dict(size=8)),
        xaxis=dict(tickangle=45),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color=COLORS["gray"])
    )
    
    return fig


def create_intensity_heatmap(df: pd.DataFrame, index_col: str, 
                              numeric_cols: list[str]) -> go.Figure:
    """Create log2 intensity heatmap."""
    import numpy as np
    
    # Prepare index labels
    if index_col and index_col in df.columns:
        labels = df[index_col].apply(parse_protein_group).tolist()
    else:
        labels = [f"Row {i}" for i in range(len(df))]
    
    # Sort columns by condition
    sorted_cols = sort_columns_by_condition(numeric_cols)
    intensity_data = df[sorted_cols].copy()
    
    # Log2 transform (data already has 1 for missing, so log2(1)=0)
    intensity_log2 = np.log2(intensity_data + 1)
    
    fig = go.Figure(data=go.Heatmap(
        z=intensity_log2.values,
        x=sorted_cols,
        y=labels,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="log2(intensity)"),
        hovertemplate="Protein: %{y}<br>Sample: %{x}<br>log2: %{z:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Intensity distribution (log2)",
        xaxis_title="Samples",
        yaxis_title="Protein groups",
        height=max(400, len(labels) * 2),
        yaxis=dict(tickfont=dict(size=8)),
        xaxis=dict(tickangle=45),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color=COLORS["gray"])
    )
    
    return fig


# --- Header ---
render_header("Exploratory data analysis", "Visualize intensity distributions and missing values")

# --- Check for cached data ---
protein_data = st.session_state.get("protein_data")
peptide_data = st.session_state.get("peptide_data")
protein_mask = st.session_state.get("protein_missing_mask")
peptide_mask = st.session_state.get("peptide_missing_mask")
protein_idx = st.session_state.get("protein_index_col")
peptide_idx = st.session_state.get("peptide_index_col")

if protein_data is None and peptide_data is None:
    st.warning("⚠ No data cached. Please upload data on the Data Upload page first.")
    st.stop()

# --- Tabs for Protein / Peptide ---
tab_protein, tab_peptide = st.tabs(["🧬 Protein data", "📋 Peptide data"])

with tab_protein:
    if protein_data is not None:
        numeric_cols = get_numeric_cols(protein_data)
        
        st.caption(f"**{len(protein_data):,} proteins** × **{len(numeric_cols)} samples**")
        
        # Subsample for large datasets
        max_rows = st.slider("Max proteins to display", 50, min(2000, len(protein_data)), 
                            min(500, len(protein_data)), key="protein_slider")
        display_df = protein_data.head(max_rows)
        display_mask = protein_mask.head(max_rows) if protein_mask is not None else None
        
        col1, col2 = st.columns(2)
        
        with col1:
            if display_mask is not None:
                fig_missing = create_missing_heatmap(display_df, display_mask, protein_idx, numeric_cols)
                st.plotly_chart(fig_missing, use_container_width=True)
                
                # Summary stats
                total_values = protein_mask[numeric_cols].size
                missing_count = protein_mask[numeric_cols].sum().sum()
                st.metric("Missing values", f"{missing_count:,} ({100*missing_count/total_values:.1f}%)")
            else:
                st.info("Missing value mask not available")
        
        with col2:
            fig_intensity = create_intensity_heatmap(display_df, protein_idx, numeric_cols)
            st.plotly_chart(fig_intensity, use_container_width=True)
            
            # Summary stats
            median_intensity = protein_data[numeric_cols].median().median()
            st.metric("Median intensity", f"{median_intensity:,.0f}")
    else:
        st.info("No protein data uploaded yet")

with tab_peptide:
    if peptide_data is not None:
        numeric_cols = get_numeric_cols(peptide_data)
        
        st.caption(f"**{len(peptide_data):,} peptides** × **{len(numeric_cols)} samples**")
        
        # Subsample for large datasets
        max_rows = st.slider("Max peptides to display", 50, min(2000, len(peptide_data)), 
                            min(500, len(peptide_data)), key="peptide_slider")
        display_df = peptide_data.head(max_rows)
        display_mask = peptide_mask.head(max_rows) if peptide_mask is not None else None
        
        col1, col2 = st.columns(2)
        
        with col1:
            if display_mask is not None:
                fig_missing = create_missing_heatmap(display_df, display_mask, peptide_idx, numeric_cols)
                st.plotly_chart(fig_missing, use_container_width=True)
                
                total_values = peptide_mask[numeric_cols].size
                missing_count = peptide_mask[numeric_cols].sum().sum()
                st.metric("Missing values", f"{missing_count:,} ({100*missing_count/total_values:.1f}%)")
            else:
                st.info("Missing value mask not available")
        
        with col2:
            fig_intensity = create_intensity_heatmap(display_df, peptide_idx, numeric_cols)
            st.plotly_chart(fig_intensity, use_container_width=True)
            
            median_intensity = peptide_data[numeric_cols].median().median()
            st.metric("Median intensity", f"{median_intensity:,.0f}")
    else:
        st.info("No peptide data uploaded yet")

# --- Footer ---
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {COLORS['gray']}; font-size: 12px; padding: 20px 0;">
    <p><strong>For research use only</strong></p>
    <p>© 2024 Thermo Fisher Scientific Inc. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
