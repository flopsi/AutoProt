import streamlit as st
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

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

# Initialize session state keys
if 'upload_stage' not in st.session_state:
    st.session_state.upload_stage = 'upload'
if 'protein_df' not in st.session_state:
    st.session_state.protein_df = None
if 'auto_species_col' not in st.session_state:
    st.session_state.auto_species_col = None
if 'auto_condition_mapping' not in st.session_state:
    st.session_state.auto_condition_mapping = None

# ========== STAGE 1: FILE UPLOAD ==========
if st.session_state.upload_stage == 'upload':
    st.markdown("## Upload Protein Data")
    
    upload_method = st.radio(
        "Select upload method:",
        options=["Upload File", "Load from URL", "Use Demo Data"],
        horizontal=True
    )
    
    df = None
    filename = None
    
    if upload_method == "Upload File":
        uploaded_file = st.file_uploader(
            "[ICON: upload] Drag and drop or browse",
            type=['csv', 'tsv', 'txt', 'xlsx', 'xls'],
            help="Supported formats: CSV, TSV, Excel"
        )
        if uploaded_file is not None:
            with st.spinner("Loading file..."):
                df = load_data_file(uploaded_file)
                filename = uploaded_file.name

    elif upload_method == "Load from URL":
        url = st.text_input("Enter URL to public data file:")
        if url and st.button("Load from URL"):
            with st.spinner("Downloading..."):
                try:
                    df = pd.read_csv(url)
                    filename = url.split('/')[-1]
                    st.success(f"Loaded {filename}")
                except Exception as e:
                    st.error(f"Error loading URL: {e}")

    elif upload_method == "Use Demo Data":
        demo_path = Path("demo_data") / "test3_pg_matrix.csv"
        if demo_path.exists():
            with st.spinner("Loading demo data..."):
                df = load_data_file(str(demo_path))
                filename = "test3_pg_matrix.csv (Demo)"
        else:
            st.warning(f"Demo file not found at: {demo_path}")
            st.info("Please ensure test3_pg_matrix.csv is in demo_data/ folder")

    # If file loaded, show preview and proceed
    if df is not None:
        st.success(f"✓ File loaded: **{filename}**")
        st.markdown("### Data Preview")
        st.dataframe(df.head(5), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        if st.button("Proceed to Column Annotation", type="primary"):
            st.session_state.protein_df = df
            st.session_state.protein_filename = filename
            st.session_state.upload_stage = 'annotate'
            st.rerun()

# ========== STAGE 2: COLUMN ANNOTATION ==========
elif st.session_state.upload_stage == 'annotate':
    df = st.session_state.protein_df
    filename = st.session_state.protein_filename

    st.markdown(f"## Column Annotations")
    st.info(f"File: **{filename}** ({df.shape[0]:,} rows × {df.shape[1]} columns)")

    # Auto-detect species column
    if st.session_state.auto_species_col is None:
        st.session_state.auto_species_col = auto_detect_species_column(df)

    # Species column selection
    st.markdown("### Species Column")
    object_columns = df.select_dtypes(include='object').columns.tolist()

    if st.session_state.auto_species_col:
        default_index = object_columns.index(st.session_state.auto_species_col)
        st.success(f"✓ Auto-detected: **{st.session_state.auto_species_col}**")
    else:
        default_index = 0
        st.warning("Could not auto-detect species column. Please select manually.")

    selected_species_col = st.selectbox(
        "Select column containing species information:",
        options=object_columns,
        index=default_index,
        help="Look for columns with _HUMAN, _ECOLI, or _YEAST suffixes"
    )

    sample_values = df[selected_species_col].head(5).tolist()
    st.caption(f"Sample values: {sample_values}")

    st.markdown("### Analysis Workflow")
    selected_workflow = st.radio(
        "Select analysis workflow:",
        options=[w.value for w in WorkflowType],
        help="Choose the analytical approach for your data"
    )

    if selected_workflow == WorkflowType.LFQ_BENCH.value:
        st.info(WorkflowType.LFQ_BENCH.description)

    if st.button("Proceed to Condition Assignment", type="primary"):
        st.session_state.selected_species_col = selected_species_col
        st.session_state.selected_workflow = selected_workflow
        st.session_state.upload_stage = 'conditions'
        st.rerun()

# ========== STAGE 3: CONDITION ASSIGNMENT ==========
elif st.session_state.upload_stage == 'conditions':
    df = st.session_state.protein_df
    filename = st.session_state.protein_filename
    selected_species_col = st.session_state.selected_species_col

    st.markdown(f"## Condition Assignment (A vs B)")
    st.info(f"File: **{filename}**")

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # If no numeric columns (they're stored as strings), convert
    if len(numeric_columns) == 0:
        for col in df.columns:
            if col != selected_species_col and df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    numeric_columns.append(col)
                except:
                    pass

    st.caption(f"Found {len(numeric_columns)} quantitative columns")

    if st.session_state.auto_condition_mapping is None:
        st.session_state.auto_condition_mapping = auto_detect_conditions(numeric_columns)

    # Render condition selector
    condition_mapping = render_condition_selector(
        numeric_columns,
        st.session_state.auto_condition_mapping
    )

    if st.button("Confirm Assignments & Process Data", type="primary"):
        with st.spinner("Processing data..."):
            metadata_cols = [col for col in df.columns if col not in numeric_columns]
            quant_cols = numeric_columns

            metadata_df = df[metadata_cols].copy()
            quant_df = df[quant_cols].copy()

            species_map = extract_species_map(df, selected_species_col)
            protein_groups = extract_protein_groups(df, selected_species_col)

            protein_data = ProteomicsDataset(
                raw_df=df,
                metadata=metadata_df,
                quant_data=quant_df,
                metadata_columns=metadata_cols,
                quant_columns=quant_cols,
                species_column=selected_species_col,
                species_map=species_map,
                condition_mapping=condition_mapping,
                protein_groups=protein_groups
            )

            st.session_state.protein_data = protein_data
            st.session_state.protein_uploaded = True
            st.session_state.upload_stage = 'summary'
            st.rerun()

# ========== STAGE 4: SUMMARY & CHARTS ==========
elif st.session_state.upload_stage == 'summary':
    protein_data = st.session_state.protein_data
    filename = st.session_state.protein_filename

    st.markdown("## Upload Summary")
    st.success("✓ Data processed successfully!")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("File", filename.split('(')[0].strip())
    with col2:
        st.metric("Proteins", f"{protein_data.n_proteins:,}")
    with col3:
        n_a = len([c for c in protein_data.condition_mapping.values() if c.startswith('A')])
        n_b = len([c for c in protein_data.condition_mapping.values() if c.startswith('B')])
        st.metric("Samples", f"{n_a + n_b} ({n_a}A + {n_b}B)")
    with col4:
        species_counts = protein_data.get_species_counts()
        st.metric("Species", len([s for s in species_counts.values() if s > 0]))

    st.markdown("### Protein Counts by Species")
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    total_species_counts = protein_data.get_species_counts()

    with chart_col1:
        fig1 = create_species_bar_chart(total_species_counts, "Total Proteins by Species")
        st.plotly_chart(fig1, use_container_width=True)

    with chart_col2:
        a_data = protein_data.get_condition_data('A')
        a_detected_indices = a_data.dropna(how='all').index
        species_a_counts = {sp: sum(1 for idx in a_detected_indices if protein_data.species_map.get(idx) == sp)
                           for sp in ['human', 'ecoli', 'yeast']}
        fig2 = create_species_bar_chart(species_a_counts, "Condition A Proteins")
        st.plotly_chart(fig2, use_container_width=True)

    with chart_col3:
        b_data = protein_data.get_condition_data('B')
        b_detected_indices = b_data.dropna(how='all').index
        species_b_counts = {sp: sum(1 for idx in b_detected_indices if protein_data.species_map.get(idx) == sp)
                           for sp in ['human', 'ecoli', 'yeast']}
        fig3 = create_species_bar_chart(species_b_counts, "Condition B Proteins")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("### Next Steps")

    nav_col1, nav_col2 = st.columns(2)

    with nav_col1:
        if st.button("📊 Upload Peptide Data", type="primary", use_container_width=True):
            st.switch_page("pages/2_🔬_Peptide_Upload.py")

    with nav_col2:
        if st.button("✓ Go to Data Quality", type="secondary", use_container_width=True):
            st.switch_page("pages/3_✓_Data_Quality.py")

    if 'auto_navigate_timer' not in st.session_state:
        st.session_state.auto_navigate_timer = datetime.now()

    elapsed = (datetime.now() - st.session_state.auto_navigate_timer).seconds
    remaining = max(0, 20 - elapsed)

    if remaining > 0:
        st.info(f"⏱️ Auto-navigating to Data Quality in {remaining} seconds...")
        time.sleep(1)
        st.rerun()
    else:
        st.switch_page("pages/3_✓_Data_Quality.py")

# Reset button (always visible)
st.markdown("---")
if st.button("🔄 Start Over", use_container_width=True):
    for key in ['upload_stage', 'protein_df', 'auto_species_col', 'auto_condition_mapping',
                'selected_species_col', 'selected_workflow', 'auto_navigate_timer']:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.upload_stage = 'upload'
    st.rerun()
