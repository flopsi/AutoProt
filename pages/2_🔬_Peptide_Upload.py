import streamlit as st
import pandas as pd
from pathlib import Path

from components.header import render_header
from components.condition_selector import render_condition_selector
from utils.file_handlers import load_data_file
from utils.species_detector import auto_detect_species_column, extract_species_map, extract_protein_groups
from utils.condition_detector import auto_detect_conditions
from models.proteomics_data import ProteomicsDataset
from config.workflows import WorkflowType

render_header()
st.title("Peptide Data Upload")

# Check for protein data first
if not st.session_state.get('protein_uploaded'):
    st.warning("⚠️ Please upload protein data first before uploading peptide data.")
    if st.button("Go to Protein Upload", type="primary"):
        st.switch_page("pages/1_📊_Protein_Upload.py")
    st.stop()

protein_data = st.session_state.protein_data
st.info(f"📊 Protein data loaded: {protein_data.n_proteins:,} proteins")

# Initialize session state
if 'peptide_upload_stage' not in st.session_state:
    st.session_state.peptide_upload_stage = 'upload'
if 'peptide_df' not in st.session_state:
    st.session_state.peptide_df = None
if 'auto_peptide_species_col' not in st.session_state:
    st.session_state.auto_peptide_species_col = None

# ========== STAGE 1: FILE UPLOAD ==========
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

    if df is not None:
        st.success(f"✓ File loaded: **{filename}**")
        st.markdown("### Data Preview")
        st.dataframe(df.head(5), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        if st.button("Proceed to Column Annotation", type="primary"):
            st.session_state.peptide_df = df
            st.session_state.peptide_filename = filename
            st.session_state.peptide_upload_stage = 'annotate'
            st.rerun()

# ========== STAGE 2: COLUMN ANNOTATION ==========
elif st.session_state.peptide_upload_stage == 'annotate':
    df = st.session_state.peptide_df
    filename = st.session_state.peptide_filename

    st.markdown(f"## Column Annotations")
    st.info(f"File: **{filename}** ({df.shape[0]:,} rows × {df.shape[1]} columns)")

    # Auto-detect protein group column (aggregation)
    if st.session_state.auto_peptide_species_col is None:
        st.session_state.auto_peptide_species_col = auto_detect_species_column(df)

    # Protein group column selection (for aggregation)
    st.markdown("### Protein Group Column")
    object_columns = df.select_dtypes(include='object').columns.tolist()

    if st.session_state.auto_peptide_species_col:
        default_index = object_columns.index(st.session_state.auto_peptide_species_col)
        st.success(f"✓ Auto-detected: **{st.session_state.auto_peptide_species_col}**")
    else:
        default_index = 0
        st.warning("Could not auto-detect protein group column. Please select manually.")
    
    selected_protein_group_col = st.selectbox(
        "Select column for peptide-to-protein aggregation:",
        options=object_columns,
        index=default_index,
        help="Column that links peptides to proteins (e.g., Protein.Group, PG.ProteinAccessions)"
    )

    sample_values = df[selected_protein_group_col].head(3).tolist()
    st.caption(f"Sample values: {sample_values}")

    # Confirm protein group column
    protein_group_confirmed = st.checkbox(
        f"✓ Confirm that **{selected_protein_group_col}** is the correct protein group column",
        value=False,
        help="Check this box to confirm the protein group column selection"
    )

    if not protein_group_confirmed:
        st.warning("⚠️ Please confirm the protein group column before proceeding")

    # ========== PEPTIDE SEQUENCE COLUMN SELECTION ==========
    st.markdown("### Peptide Sequence Column (Optional)")
    
    has_sequence = st.checkbox(
        "Data contains peptide sequence column",
        value=True,
        help="Check if your data includes actual peptide sequences"
    )

    selected_seq_col = None
    if has_sequence:
        sequence_col_candidates = [col for col in object_columns if col != selected_protein_group_col]
        selected_seq_col = st.selectbox(
            "Select peptide sequence column:",
            options=sequence_col_candidates,
            help="Column containing peptide sequences"
        )
        sample_seq = df[selected_seq_col].head(3).tolist()
        st.caption(f"Sample sequences: {sample_seq}")

    # ========== CHECKBOX TABLE FOR SAMPLE COLUMN SELECTION ==========
    st.markdown("### Select Quantitative Sample Columns")
    st.info("✓ Check the boxes for columns containing peptide intensity values. Leave metadata columns unchecked.")
    
    all_columns = df.columns.tolist()
    
    # Auto-suggest numeric columns
    suggested_quant = []
    for col in all_columns:
        if col in [selected_protein_group_col, selected_seq_col]:
            continue
        try:
            pd.to_numeric(df[col], errors='raise')
            suggested_quant.append(col)
        except:
            pass
    
    # Create selection table
    column_selection = []
    for col in all_columns:
        is_metadata = (col in [selected_protein_group_col, selected_seq_col])
        is_suggested = (col in suggested_quant)
        column_selection.append({
            'Column Name': col,
            'Select as Sample': is_suggested and not is_metadata,
            'Data Type': str(df[col].dtype),
            'Sample Values': str(df[col].head(2).tolist()[:2])
        })
    
    selection_df = pd.DataFrame(column_selection)
    
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
        key='peptide_column_selector'
    )
    
    selected_quant_cols = edited_selection[edited_selection['Select as Sample'] == True]['Column Name'].tolist()
    
    if len(selected_quant_cols) == 0:
        st.warning("⚠️ No sample columns selected. Please check at least one column.")
    else:
        st.success(f"✓ Selected {len(selected_quant_cols)} sample columns")
        with st.expander("View selected columns"):
            st.write(selected_quant_cols)

    if st.button("Proceed to Condition Assignment", type="primary", 
                 disabled=(len(selected_quant_cols) == 0 or not protein_group_confirmed)):
        st.session_state.selected_protein_group_col = selected_protein_group_col
        st.session_state.selected_seq_col = selected_seq_col if has_sequence else None
        st.session_state.selected_peptide_quant_cols = selected_quant_cols
        st.session_state.peptide_upload_stage = 'conditions'
        st.rerun()

# ========== STAGE 3: CONDITION ASSIGNMENT ==========
elif st.session_state.peptide_upload_stage == 'conditions':
    df = st.session_state.peptide_df
    filename = st.session_state.peptide_filename
    selected_protein_group_col = st.session_state.selected_protein_group_col
    selected_quant_cols = st.session_state.selected_peptide_quant_cols

    st.markdown(f"## Condition Assignment (A vs B)")
    st.info(f"File: **{filename}** | Selected {len(selected_quant_cols)} sample columns")

    # Convert selected columns to numeric if needed
    for col in selected_quant_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Use same condition mapping as protein data
    condition_mapping = protein_data.condition_mapping
    
    st.markdown("### Inherited from Protein Data")
    st.success(f"✓ Using condition mapping from protein data ({len(condition_mapping)} samples)")
    
    # Display mapping
    mapping_df = pd.DataFrame({
        'Sample Column': list(condition_mapping.keys()),
        'Condition': list(condition_mapping.values())
    })
    st.dataframe(mapping_df, hide_index=True, use_container_width=True)

    if st.button("Confirm & Process Data", type="primary"):
        with st.spinner("Processing peptide data..."):
            metadata_cols = [col for col in df.columns if col not in selected_quant_cols]
            quant_cols = selected_quant_cols

            metadata_df = df[metadata_cols].copy()
            quant_df = df[quant_cols].copy()

            species_map = extract_species_map(df, selected_protein_group_col)
            protein_groups = extract_protein_groups(df, selected_protein_group_col)

            peptide_data = ProteomicsDataset(
                raw_df=df,
                metadata=metadata_df,
                quant_data=quant_df,
                metadata_columns=metadata_cols,
                quant_columns=quant_cols,
                species_column=selected_protein_group_col,
                species_map=species_map,
                condition_mapping=condition_mapping,
                aggregation_column=selected_protein_group_col,
                protein_groups=protein_groups
            )

            st.session_state.peptide_data = peptide_data
            st.session_state.peptide_uploaded = True
            st.session_state.peptide_upload_stage = 'summary'
            st.rerun()

# ========== STAGE 4: SUMMARY ==========
elif st.session_state.peptide_upload_stage == 'summary':
    peptide_data = st.session_state.peptide_data
    filename = st.session_state.peptide_filename

    st.markdown("## Upload Summary")
    st.success("✓ Peptide data processed successfully!")

    st.markdown("---")
    st.markdown("### Next Steps")

    if st.button("✓ Go to Data Quality", type="primary", use_container_width=True):
        st.switch_page("pages/3_✓_Data_Quality.py")

# Reset button (always visible)
st.markdown("---")
if st.button("🔄 Start Over", use_container_width=True):
    for key in ['peptide_upload_stage', 'peptide_df', 'auto_peptide_species_col',
                'selected_protein_group_col', 'selected_seq_col', 'selected_peptide_quant_cols']:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.peptide_upload_stage = 'upload'
    st.rerun()
