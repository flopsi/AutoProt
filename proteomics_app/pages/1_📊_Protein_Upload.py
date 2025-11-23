import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

from components.header import render_header
from components.charts import create_species_bar_chart
from components.condition_selector import render_condition_selector
from utils.file_handlers import load_data_file
from utils.species_detector import auto_detect_species_column, extract_species_map, extract_protein_groups
from utils.condition_detector import auto_detect_conditions
from models.proteomics_data import ProteomicsDataset
from config.workflows import WorkflowType

render_header()
st.title("Protein Data Upload")

if 'upload_stage' not in st.session_state:
    st.session_state.upload_stage = 'upload'

if st.session_state.upload_stage == 'upload':
    st.markdown("## Upload Protein Data")
    upload_method = st.radio("Select method:", ["Upload File", "Use Demo Data"], horizontal=True)
    
    df = None
    filename = None
    
    if upload_method == "Upload File":
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'tsv', 'xlsx', 'xls'])
        if uploaded_file:
            df = load_data_file(uploaded_file)
            filename = uploaded_file.name
    
    elif upload_method == "Use Demo Data":
        demo_path = Path("demo_data") / "test3_pg_matrix.csv"
        if demo_path.exists():
            df = load_data_file(str(demo_path))
            filename = "test3_pg_matrix.csv (Demo)"
    
    if df is not None:
        st.success(f"✓ Loaded: {filename}")
        st.dataframe(df.head(5))
        if st.button("Proceed", type="primary"):
            st.session_state.protein_df = df
            st.session_state.protein_filename = filename
            st.session_state.upload_stage = 'annotate'
            st.rerun()

elif st.session_state.upload_stage == 'annotate':
    df = st.session_state.protein_df
    st.markdown("## Column Annotations")
    
    auto_col = auto_detect_species_column(df)
    object_cols = df.select_dtypes(include='object').columns.tolist()
    selected_species_col = st.selectbox("Species column:", object_cols, 
                                        index=object_cols.index(auto_col) if auto_col else 0)
    
    st.markdown("### Workflow")
    selected_workflow = st.radio("Select workflow:", [WorkflowType.LFQ_BENCH.value])
    
    if st.button("Proceed to Conditions", type="primary"):
        st.session_state.selected_species_col = selected_species_col
        st.session_state.upload_stage = 'conditions'
        st.rerun()

elif st.session_state.upload_stage == 'conditions':
    df = st.session_state.protein_df
    selected_species_col = st.session_state.selected_species_col
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    auto_mapping = auto_detect_conditions(numeric_cols)
    condition_mapping = render_condition_selector(numeric_cols, auto_mapping)
    
    if st.button("Process Data", type="primary"):
        metadata_cols = [c for c in df.columns if c not in numeric_cols]
        protein_data = ProteomicsDataset(
            raw_df=df,
            metadata=df[metadata_cols],
            quant_data=df[numeric_cols],
            metadata_columns=metadata_cols,
            quant_columns=numeric_cols,
            species_column=selected_species_col,
            species_map=extract_species_map(df, selected_species_col),
            condition_mapping=condition_mapping,
            protein_groups=extract_protein_groups(df, selected_species_col)
        )
        st.session_state.protein_data = protein_data
        st.session_state.protein_uploaded = True
        st.session_state.upload_stage = 'summary'
        st.rerun()

elif st.session_state.upload_stage == 'summary':
    protein_data = st.session_state.protein_data
    st.success("✓ Data processed!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Proteins", f"{protein_data.n_proteins:,}")
    with col2:
        n_a = len([c for c in protein_data.condition_mapping.values() if c.startswith('A')])
        st.metric("Condition A", n_a)
    with col3:
        n_b = len([c for c in protein_data.condition_mapping.values() if c.startswith('B')])
        st.metric("Condition B", n_b)
    
    st.markdown("### Species Breakdown")
    chart_col1, chart_col2, chart_col3 = st.columns(3)
    
    total_counts = protein_data.get_species_counts()
    
    with chart_col1:
        fig1 = create_species_bar_chart(total_counts, "Total")
        st.plotly_chart(fig1, use_container_width=True)
    
    with chart_col2:
        a_data = protein_data.get_condition_data('A')
        a_idx = a_data.dropna(how='all').index
        a_counts = {sp: sum(1 for i in a_idx if protein_data.species_map.get(i) == sp)
                   for sp in ['human', 'ecoli', 'yeast']}
        fig2 = create_species_bar_chart(a_counts, "Condition A")
        st.plotly_chart(fig2, use_container_width=True)
    
    with chart_col3:
        b_data = protein_data.get_condition_data('B')
        b_idx = b_data.dropna(how='all').index
        b_counts = {sp: sum(1 for i in b_idx if protein_data.species_map.get(i) == sp)
                   for sp in ['human', 'ecoli', 'yeast']}
        fig3 = create_species_bar_chart(b_counts, "Condition B")
        st.plotly_chart(fig3, use_container_width=True)
    
    if st.button("Go to Data Quality", type="primary"):
        st.switch_page("pages/3_✓_Data_Quality.py")

if st.button("🔄 Start Over"):
    for key in ['upload_stage', 'protein_df']:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.upload_stage = 'upload'
    st.rerun()