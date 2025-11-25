# app.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go


from utils.data_generator import generate_mock_proteins
from utils.stats import (
    log2_transform,
    compute_cv,
    missing_fraction,
    quartiles,
    prepare_dataframe_from_proteins
)
from utils import stats as stats_utils
from components.qc_plots import (
    replicate_boxplots,
    cv_histogram,
    pca_scatter,
    missing_value_heatmap,
    rank_plot
)

# Simple in-file constants
REPLICATE_GROUPS: Dict[str, List[str]] = {
    'A': ['A1', 'A2', 'A3'],
    'B': ['B1', 'B2', 'B3']
}
REPLICATE_NAMES = REPLICATE_GROUPS['A'] + REPLICATE_GROUPS['B']

# Initialize session state
def init_state():
    if 'view' not in st.session_state:
        st.session_state.view = 'UPLOAD'  # UPLOAD / QC / GUIDE
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = pd.DataFrame()
    if 'replicate_names' not in st.session_state:
        st.session_state.replicate_names = REPLICATE_NAMES
    if 'qc_df' not in st.session_state:
        st.session_state.qc_df = pd.DataFrame()
    if 'log2_df' not in st.session_state:
        st.session_state.log2_df = pd.DataFrame()
init_state()

st.set_page_config(page_title="Proteomics QC Studio", layout="wide")

st.title("Proteomics QC Studio")
st.write("Guided workflow: Load -> Check Normality -> Transform (Log2) -> Visualize")

# Sidebar: Replicate configuration
st.sidebar.header("Replicate Configuration")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.session_state.view_mode = st.checkbox("Enable guided workflow", value=True)

# Upload section
st.header("Data Upload")
uploaded_file = st.file_uploader("Upload a CSV/TSV with replicate intensities", type=['csv','tsv'], key='upload_input')
def parse_uploaded(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    sep = ',' if file.name.endswith('.csv') else '\t'
    df = pd.read_csv(file, sep=sep)
    # Expect columns like: id/gene/description and replicate columns
    if 'id' in df.columns:
        df = df.set_index('id')
    elif 'Gene' in df.columns:
        df = df.set_index('Gene')
    # Ensure replicate columns exist
    for r in REPLICATE_NAMES:
        if r not in df.columns:
            df[r] = np.nan
    return df

if uploaded_file is not None:
    with st.spinner("Loading data..."):
        raw_df = parse_uploaded(uploaded_file)
        st.session_state.raw_data = raw_df
        st.success(f"Loaded {raw_df.shape[0]} proteins with replicates: {REPLICATE_NAMES}")

# Demo dataset button (optional)
if st.sidebar.button("Load Demo Dataset"):
    demo = generate_mock_proteins(60, REPLICATE_NAMES)
    demo_df = pd.DataFrame.from_records(demo).set_index('id')
    st.session_state.raw_data = demo_df
    st.success("Demo dataset loaded with replicates: " + ", ".join(REPLICATE_NAMES))

# Visualize raw replication counts table (basic)
if not st.session_state.raw_data.empty:
    st.subheader("Raw Data Snapshot")
    st.dataframe(st.session_state.raw_data[[*REPLICATE_NAMES]].head())

# Guided workflow toggle
if st.session_state.get('view_mode', True):
    st.markdown("---")
    st.subheader("Guided Analysis Workflow")

    # Step 1: Check Normality
    if st.button("Check Normality (Step 1 of 4)"):
        df = st.session_state.raw_data.copy()
        if df.empty:
            st.warning("No data loaded yet.")
        else:
            # Normality test per protein across replicates
            normality_results = {}
            for idx, row in df.iterrows():
                vals = row[REPLICATE_NAMES].dropna().values
                if len(vals) < 3:
                    normality_results[idx] = False
                    continue
                try:
                    vals_log = np.log2(vals)
                    w, p = stats.shapiro(vals_log)
                    normality_results[idx] = p > 0.05
                except Exception:
                    normality_results[idx] = False
            st.session_state.normality = normality_results
            st.success(f"Normality checked for {len(normality_results)} proteins. See results in sidebar.")

    # Step 2: Transform (Log2)
    if st.button("Transform (Log2) & Prepare (Step 2 of 4)"):
        df = st.session_state.raw_data.copy()
        if df.empty:
            st.warning("No data loaded yet.")
        else:
            df_log2 = df.copy()
            for r in REPLICATE_NAMES:
                df_log2[r] = df_log2[r].apply(lambda x: np.log2(x) if pd.notnull(x) and x > 0 else np.nan)
            st.session_state.log2_df = df_log2
            st.success("Log2 transformation applied to replicates.")

    # Step 3: Visualize QC
    if st.button("Show QC Visualizations (Step 3 of 4)"):
        df = st.session_state.log2_df if 'log2_df' in st.session_state and not st.session_state.log2_df.empty else st.session_state.raw_data
        if df is None or df.empty:
            st.warning("No data available for QC plots.")
        else:
            # Prepare a small QC dataframe: index proteins, replicate columns
            qc_df = df.copy()
            # Fill missing as NaN
            qc_df = qc_df[REPLICATE_NAMES]
            st.session_state.qc_df = qc_df

            # Replicate Boxplots
            st.subheader("Replicate Boxplots")
            fig_box = replicate_boxplots(qc_df, REPLICATE_NAMES, title="Replicate Boxplots")
            st.plotly_chart(fig_box, use_container_width=True)

            # CV Distribution
            st.subheader("CV Distribution")
            # compute CV per protein from original or log2? We'll use original intensities for CV
            cvs = []
            for idx, row in st.session_state.raw_data.iterrows():
                vals = [row[r] for r in REPLICATE_NAMES if pd.notnull(row[r])]
                if len(vals) > 1:
                    cvs.append(np.std(vals, ddof=1) / np.mean(vals))
            if len(cvs) > 0:
                fig_cv = px.histogram(x=cvs, nbins=20, title="CV Distribution across Proteins")
                st.plotly_chart(fig_cv, use_container_width=True)
            else:
                st.info("Not enough data to compute CV.")

            # PCA Scatter (on log2 data if available)
            st.subheader("PCA Scatter Plot")
            # Build a data matrix: proteins x replicates
            mat = qc_df.copy()
            # drop rows with all NaN in replicates
            mat = mat.dropna(how='all')
            # Some rows may still have NaN; fill with column means
            mat_filled = mat.fillna(mat.mean())
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                pcs = pca.fit_transform(mat_filled.values)
                pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'], index=mat_filled.index)
                pca_fig = px.scatter(pca_df, x='PC1', y='PC2', title="PCA of Proteins (log2-transformed)", hover_data=[pca_df.index])
                st.plotly_chart(pca_fig, use_container_width=True)
            except Exception as e:
                st.error(f"PCA failed: {e}")

            # Missing Value Heatmap
            st.subheader("Missing Value Heatmap")
            heatmap_fig = missing_value_heatmap(mat)
            st.plotly_chart(heatmap_fig, use_container_width=True)

            # Rank Plot
            st.subheader("Rank Plot (Dynamic Range)")
            try:
                rank_fig = rank_plot(mat_filled, by_col='MeanIntensity', replicate_names=REPLICATE_NAMES)
                # The above function expects a certain structure; adapt as simple line
                rank_df = pd.DataFrame({
                    'Protein': mat_filled.index,
                    'MeanIntensity': mat_filled[REPLICATE_NAMES].mean(axis=1)
                }).set_index('Protein')
                rank_fig = px.line(rank_df.reset_index(), x='Protein', y='MeanIntensity', title='Rank Plot by Mean Intensity')
                st.plotly_chart(rank_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Rank plot failed: {e}")

    # Step 4: Visualize final results (optional)
    if st.button("Visualize Summary (Step 4 of 4)"):
        df = st.session_state.log2_df if 'log2_df' in st.session_state and not st.session_state.log2_df.empty else st.session_state.raw_data
        if df is None or df.empty:
            st.warning("No data available for visualization.")
        else:
            st.subheader("Summary by Protein")
            df_summ = df[REPLICATE_NAMES].copy()
            df_summ['Mean'] = df_summ.mean(axis=1)
            df_summ['CV'] = df_summ.apply(lambda row: stats_utils.compute_cv(row.dropna().tolist()) if row.dropna().shape[0] > 1 else np.nan, axis=1)
            display_cols = ['Mean', 'CV']
            st.dataframe(df_summ[[*REPLICATE_NAMES, 'Mean', 'CV']].head())
else:
    st.info("Guided workflow is disabled. You can still explore data using the QC plots below.")

st.markdown("---")
st.header("Manual QC Explorer")
if not st.session_state.qc_df.empty:
    qc_df = st.session_state.qc_df
    st.subheader("Replicate Boxplots (Manual)")
    fig_box = replicate_boxplots(qc_df, REPLICATE_NAMES, title="Replicate Boxplots (Manual)")
    st.plotly_chart(fig_box, use_container_width=True)

# End of app.py
