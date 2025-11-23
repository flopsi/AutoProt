import streamlit as st
import pandas as pd
from pathlib import Path

from components.header import render_header
from components.charts import create_species_bar_chart
from components.condition_selector import render_condition_selector
from utils.file_handlers import load_data_file
from utils.species_detector import extract_species_map, extract_protein_groups
from models.proteomics_data import ProteomicsDataset

render_header()
st.title("Peptide Data Upload")

# Check for protein data first
if not st.session_state.get('protein_uploaded'):
    st.warning("⚠️ Please upload protein data first before uploading peptide data.")
    if st.button("Go to Protein Upload"):
        st.switch_page("pages/1_📊_Protein_Upload.py")
    st.stop()

protein_data = st.session_state.protein_data
st.info(f"📊 Protein data loaded: {protein_data.n_proteins:,} proteins")

if 'peptide_upload_stage' not in st.session_state:
    st.session_state.peptide_upload_stage = 'upload'

# Upload stage
if st.session_state.peptide_upload_stage == 'upload':
    st.markdown("## Upload Peptide Data")

    upload_method = st.radio(
        "Select upload method:",
        options=["Upload File", "Load from URL"],
        horizontal=True
    )

    df = None
    filename = None

    if upload_method == "Upload File":
        uploaded_file = st.file_uploader(
            "[ICON: upload] Drag and drop or browse",
            type=['csv', 'tsv', 'txt', 'xlsx', 'xls']
        )
        if uploaded_file is not None:
            with st.spinner("Loading file..."):
                df = load_data_file(uploaded_file)
                filename = uploaded_file.name

    elif upload_method == "Load from URL":
        url = st.text_input("Enter URL:")
        if url and st.button("Load"):
            with st.spinner("Downloading..."):
                try:
                    df = pd.read_csv(url)
                    filename = url.split('/')[-1]
                except Exception as e:
                    st.error(f"Error: {e}")

    if df is not None:
        st.success(f"✓ File loaded: **{filename}**")
        st.dataframe(df.head(5), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

        if st.button("Proceed to Aggregation Column Selection", type="primary"):
            st.session_state.peptide_df = df
            st.session_state.peptide_filename = filename
            st.session_state.peptide_upload_stage = 'aggregate'
            st.rerun()

elif st.session_state.peptide_upload_stage == 'aggregate':
    df = st.session_state.peptide_df
    filename = st.session_state.peptide_filename

    st.markdown("## Peptide-to-Protein Mapping")
    st.info(f"File: **{filename}** ({df.shape[0]:,} peptides)")

    # Auto-detect aggregation column
    common_names = [
        'Protein.Group', 'ProteinGroup', 'Protein_Group', 'Protein.Accession', 'ProteinAccession'
    ]
    auto_agg_col = None
    for col in df.columns:
        if col in common_names:
            auto_agg_col = col
            break
    if auto_agg_col is None:
        for col in df.columns:
            if col in protein_data.metadata.columns:
                auto_agg_col = col
                break

    object_columns = df.select_dtypes(include='object').columns.tolist()
    if auto_agg_col:
        default_index = object_columns.index(auto_agg_col)
        st.success(f"✓ Auto-detected: **{auto_agg_col}**")
    else:
        default_index = 0
        st.warning("Please select aggregation column manually")

    selected_agg_col = st.selectbox(
        "Select column for peptide-to-protein aggregation:",
        options=object_columns,
        index=default_index,
        help="Column that links peptides to proteins"
    )

    if selected_agg_col in protein_data.metadata.columns:
        st.success(f"✓ Column **{selected_agg_col}** found in protein data")
    else:
        st.warning(f"⚠️ Column **{selected_agg_col}** not found in protein data. Verify compatibility.")

    has_peptide_seq = st.checkbox(
        "Data contains peptide sequence column", value=True,
        help="Check if your data includes actual peptide sequences"
    )

    if st.button("Proceed to Summary", type="primary"):
        st.session_state.selected_agg_col = selected_agg_col
        st.session_state.has_peptide_seq = has_peptide_seq
        st.session_state.peptide_upload_stage = 'summary'
        st.rerun()

elif st.session_state.peptide_upload_stage == 'summary':
    df = st.session_state.peptide_df
    filename = st.session_state.peptide_filename
    selected_agg_col = st.session_state.selected_agg_col

    with st.spinner("Processing peptide data..."):
        condition_mapping = protein_data.condition_mapping
        numeric_columns = [col for col in df.columns if col in condition_mapping.keys()]
        metadata_cols = [col for col in df.columns if col not in numeric_columns]
        metadata_df = df[metadata_cols].copy()
        quant_df = df[numeric_columns].copy()
        for col in numeric_columns:
            quant_df[col] = pd.to_numeric(quant_df[col], errors='coerce')

        species_map = extract_species_map(df, selected_agg_col)
        protein_groups = extract_protein_groups(df, selected_agg_col)

        peptide_data = ProteomicsDataset(
            raw_df=df,
            metadata=metadata_df,
            quant_data=quant_df,
            metadata_columns=metadata_cols,
            quant_columns=numeric_columns,
            species_column=selected_agg_col,
            species_map=species_map,
            condition_mapping=condition_mapping,
            aggregation_column=selected_agg_col,
            protein_groups=protein_groups
        )

        st.session_state.peptide_data = peptide_data
        st.session_state.peptide_uploaded = True

    st.markdown("## Upload Summary")
    st.success("✓ Peptide data processed successfully!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File", filename)
    with col2:
        st.metric("Peptides", f"{peptide_data.n_rows:,}")
    with col3:
        st.metric("Protein Groups", f"{peptide_data.n_proteins:,}")

    st.markdown("### Peptide Counts by Species")
    chart_col1, chart_col2
