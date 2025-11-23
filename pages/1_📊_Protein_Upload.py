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
            "Drag and drop or browse",
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

# ========== STAGE 2: COLUMN ANNOTATION WITH CHECKBOXES ==========
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

    sample_values = df[selected_species_col].head(3).tolist()
    st.caption(f"Sample values: {sample_values}")

    # Require explicit confirmation
    species_confirmed = st.checkbox(
        f"✓ Confirm that **{selected_species_col}** is the correct species column",
        value=False,
        help="Check this box to confirm the species column selection"
    )

    if not species_confirmed:
        st.warning("⚠️ Please confirm the species column before proceeding")

    # ========== CHECKBOX TABLE FOR COLUMN SELECTION ==========
    st.markdown("### Select Quantitative Sample Columns")
    st.info("✓ Check the boxes for columns containing sample intensity values. Leave metadata columns unchecked.")
    
    # Create DataFrame for checkbox selection
    all_columns = df.columns.tolist()
    
    # Auto-suggest: numeric types or convertible columns
    suggested_quant = []
    for col in all_columns:
        if col == selected_species_col:
            continue  # Skip species column
        try:
            pd.to_numeric(df[col], errors='raise')
            suggested_quant.append(col)
        except:
            pass
    
    # Create selection table
    column_selection = []
    for col in all_columns:
        is_species = (col == selected_species_col)
        is_suggested = (col in suggested_quant)
        column_selection.append({
            'Column Name': col,
            'Select as Sample': is_suggested and not is_species,
            'Data Type': str(df[col].dtype),
            'Sample Values': str(df[col].head(2).tolist()[:2])
        })
    
    selection_df = pd.DataFrame(column_selection)
    
    # Use data_editor with checkbox column
    edited_selection = st.data_editor(
        selection_df,
        column_config={
            'Column Name': st.column_config.TextColumn('Column Name', disabled=True, width='medium'),
            'Select as Sample': st.column_config.CheckboxColumn('Select as Sample', default=False, width='small'),
            'Data Type': st.column_config.TextColumn('Data Type', disabled=True, width='small'),
            'Sample Values': st.column_config.TextColumn('Sample Values', disabled=True, width='large')
        },
        hide_index=True,
        use_container_width=True,
        key='column_selector'
    )
    
    # Extract selected columns
    selected_quant_cols = edited_selection[edited_selection['Select as Sample'] == True]['Column Name'].tolist()
    
    if len(selected_quant_cols) == 0:
        st.warning("⚠️ No sample columns selected. Please check at least one column.")
    else:
        st.success(f"✓ Selected {len(selected_quant_cols)} sample columns")
        with st.expander("View selected columns"):
            st.write(selected_quant_cols)

    st.markdown("### Analysis Workflow")
    selected_workflow = st.radio(
        "Select analysis workflow:",
        options=[w.value for w in WorkflowType],
        help="Choose the analytical approach for your data"
    )

    if selected_workflow == WorkflowType.LFQ_BENCH.value:
        st.info(WorkflowType.LFQ_BENCH.description)

    if st.button("Proceed to Condition Assignment", type="primary", disabled=(len(selected_quant_cols) == 0 or not species_confirmed)):
        st.session_state.selected_species_col = selected_species_col
        st.session_state.selected_workflow = selected_workflow
        st.session_state.selected_quant_cols = selected_quant_cols
        st.session_state.upload_stage = 'conditions'
        st.rerun()

# ========== STAGE 3: CONDITION ASSIGNMENT ==========
elif st.session_state.upload_stage == 'conditions':
    df = st.session_state.protein_df
    filename = st.session_state.protein_filename
    selected_species_col = st.session_state.selected_species_col
    selected_quant_cols = st.session_state.selected_quant_cols

    st.markdown(f"## Condition Assignment (A vs B)")
    st.info(f"File: **{filename}** | Selected {len(selected_quant_cols)} sample columns")

    # Convert selected columns to numeric if needed
    for col in selected_quant_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if st.session_state.auto_condition_mapping is None:
        st.session_state.auto_condition_mapping = auto_detect_conditions(selected_quant_cols)

    # Render condition selector
    condition_mapping = render_condition_selector(
        selected_quant_cols,
        st.session_state.auto_condition_mapping
    )

    if st.button("Confirm Assignments & Process Data", type="primary"):
        with st.spinner("Processing data..."):
            metadata_cols = [col for col in df.columns if col not in selected_quant_cols]
            quant_cols = selected_quant_cols

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

# ========== STAGE 4: SUMMARY & THREE-COLUMN CHARTS ==========
elif st.session_state.upload_stage == 'summary':
    protein_data = st.session_state.protein_data
    filename = st.session_state.protein_filename

    st.markdown("## Upload Summary")
    st.success("✓ Data processed successfully!")

  
    st.markdown("---")
    st.markdown("### Next Steps")

    nav_col1, nav_col2, nav_col3 = st.columns(3)

    with nav_col1:
        if st.button("📊 Upload Peptide Data", type="primary", use_container_width=True):
            st.switch_page("pages/2_🔬_Peptide_Upload.py")

    with nav_col2:
        if st.button("✓ Go to Data Quality", type="secondary", use_container_width=True):
            st.switch_page("pages/3_✓_Data_Quality.py")


    with nav_col3:
        if st.button("🔄 Start Over", use_container_width=True):
            for key in ['upload_stage', 'protein_df', 'auto_species_col', 'auto_condition_mapping',
                'selected_species_col', 'selected_workflow', 'selected_quant_cols']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.upload_stage = 'upload'
            st.rerun()
