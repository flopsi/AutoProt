"""
Thermo Fisher Proteomics App - Module 1: Data Import
FIXED: Duplicate column handling, trimming, deprecation warnings
"""
import streamlit as st
import pandas as pd
from config import (
    get_numeric_columns,
    get_metadata_columns,
    get_default_species_mapping_cols,
    get_default_group_col,
    get_default_peptide_id_col,
    trim_column_names,
    THERMO_COLORS
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Proteomics Analysis | Thermo Fisher",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# THERMO FISHER CSS & BRANDING (THEME-AWARE)
# ============================================================================
st.markdown(f"""
<style>
/* Thermo Fisher Brand Colors */
:root {{
  --primary-red: {THERMO_COLORS['PRIMARY_RED']};
  --primary-gray: {THERMO_COLORS['PRIMARY_GRAY']};
  --light-gray: {THERMO_COLORS['LIGHT_GRAY']};
  --navy: {THERMO_COLORS['NAVY']};
  --dark-red: {THERMO_COLORS['DARK_RED']};
  --orange: {THERMO_COLORS['ORANGE']};
  --yellow: {THERMO_COLORS['YELLOW']};
  --green: {THERMO_COLORS['GREEN']};
  --sky: {THERMO_COLORS['SKY']};
}}

/* Theme-aware variables */
[data-theme="light"] {{
  --bg-primary: #f8f9fa;
  --bg-secondary: #ffffff;
  --text-primary: #54585A;
  --text-secondary: #6c757d;
  --border-color: #E2E3E4;
}}

[data-theme="dark"] {{
  --bg-primary: #0e1117;
  --bg-secondary: #262730;
  --text-primary: #fafafa;
  --text-secondary: #a3a8b8;
  --border-color: #3d4149;
}}

/* Auto-detect system theme */
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg-primary: #0e1117;
    --bg-secondary: #262730;
    --text-primary: #fafafa;
    --text-secondary: #a3a8b8;
    --border-color: #3d4149;
  }}
}}

@media (prefers-color-scheme: light) {{
  :root {{
    --bg-primary: #f8f9fa;
    --bg-secondary: #ffffff;
    --text-primary: #54585A;
    --text-secondary: #6c757d;
    --border-color: #E2E3E4;
  }}
}}

/* Global styles */
body {{
  font-family: Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background-color: var(--bg-primary);
  color: var(--text-primary);
}}

[data-testid="stAppViewContainer"] {{
  background-color: var(--bg-primary);
}}

/* Header Banner */
.header-banner {{
  background: linear-gradient(90deg, var(--primary-red) 0%, var(--dark-red) 100%);
  padding: 30px 40px;
  border-radius: 0;
  margin-bottom: 30px;
  color: white;
}}

.header-banner h1 {{
  margin: 0;
  font-size: 28pt;
  color: white;
  font-weight: 600;
}}

.header-banner p {{
  margin: 5px 0 0 0;
  font-size: 14px;
  opacity: 0.95;
}}

/* Step indicators */
.step-indicator {{
  display: inline-block;
  width: 32px;
  height: 32px;
  background: var(--primary-red);
  color: white;
  border-radius: 50%;
  text-align: center;
  line-height: 32px;
  font-weight: 600;
  margin-right: 10px;
}}

.step-header {{
  display: flex;
  align-items: center;
  margin: 25px 0 15px 0;
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
}}

/* Buttons */
.stButton button {{
  background-color: var(--primary-red);
  color: white;
  border: none;
  padding: 10px 24px;
  border-radius: 6px;
  font-weight: 500;
  transition: all 0.3s;
}}

.stButton button:hover {{
  background-color: var(--dark-red);
}}

/* Status Messages */
.status-success {{
  background-color: rgba(181, 189, 0, 0.15);
  border-left: 4px solid var(--green);
  padding: 15px;
  border-radius: 4px;
  margin: 15px 0;
  color: var(--text-primary);
}}

.status-info {{
  background-color: rgba(38, 34, 98, 0.15);
  border-left: 4px solid var(--navy);
  padding: 15px;
  border-radius: 4px;
  margin: 15px 0;
  color: var(--text-primary);
}}

.status-warning {{
  background-color: rgba(234, 118, 0, 0.15);
  border-left: 4px solid var(--orange);
  padding: 15px;
  border-radius: 4px;
  margin: 15px 0;
  color: var(--text-primary);
}}

/* Upload sections */
.upload-section {{
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 25px;
}}

/* Metrics */
.stMetric {{
  background-color: var(--bg-secondary);
  padding: 15px;
  border-radius: 8px;
  border: 1px solid var(--border-color);
}}

/* Footer */
.footer {{
  text-align: center;
  padding: 30px 0;
  color: var(--text-secondary);
  font-size: 12px;
  border-top: 1px solid var(--border-color);
  margin-top: 60px;
}}

.footer strong {{
  display: block;
  margin-bottom: 10px;
}}

/* Dividers */
hr {{
  border: none;
  border-top: 1px solid var(--border-color);
  margin: 30px 0;
}}

/* Data editor styling */
.stDataFrame {{
  background-color: var(--bg-secondary);
}}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="header-banner">
  <h1>DIA Proteomics Analysis Framework</h1>
  <p>Module 1: Data Import & Validation | Thermo Fisher Scientific</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'protein_upload' not in st.session_state:
    st.session_state.protein_upload = None
if 'peptide_upload' not in st.session_state:
    st.session_state.peptide_upload = None

# For tracking column selection state
if 'protein_preview_df' not in st.session_state:
    st.session_state.protein_preview_df = None
if 'peptide_preview_df' not in st.session_state:
    st.session_state.peptide_preview_df = None

# ============================================================================
# REUSABLE UPLOAD BLOCK FUNCTION (FIXED)
# ============================================================================
def upload_annotation_block(kind, id_keys, col):
    """
    Step-by-step upload with column selection and annotation
    FIXED: Handles duplicate trimmed names, stores data separately
    
    Args:
        kind: "protein" or "peptide"
        id_keys: Dict with widget key IDs
        col: Streamlit column to render in
    """
    with col:
        # STEP 1: File Upload
        st.markdown(f"""
        <div class="step-header">
          <span class="step-indicator">1</span>
          <span>Upload {kind.capitalize()}-Level File</span>
        </div>
        """, unsafe_allow_html=True)
        
        user_file = st.file_uploader(
            f"Choose {kind}-level file",
            key=f"upl_{kind}",
            type=['csv', 'tsv', 'txt']
        )
        
        session_key = f"{kind}_upload"
        preview_key = f"{kind}_preview_df"
        
        if not user_file:
            st.markdown(f"""
            <div class="status-info">
              <strong>ℹ️ Info</strong><br>
              Upload a {kind}-level quantification file to begin analysis.
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Read just the header and first 100 rows for preview
        sep = '\t' if user_file.name.endswith(('.tsv', '.txt')) else ','
        preview_df = pd.read_csv(user_file, sep=sep, nrows=100)
        
        st.success(f"✓ File detected: **{user_file.name}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Preview Rows", f"{len(preview_df):,}")
        with col2:
            st.metric("Total Columns", len(preview_df.columns))
        
        # Detect columns
        num_cols = get_numeric_columns(preview_df)
        meta_cols = get_metadata_columns(preview_df, num_cols)
        
        # Auto-trim ALL column names (with duplicate handling)
        trimmed_quant = trim_column_names(num_cols)
        trimmed_meta = trim_column_names(meta_cols)
        
        st.markdown("---")
        
        # STEP 2: Column Selection and Annotation
        st.markdown(f"""
        <div class="step-header">
          <span class="step-indicator">2</span>
          <span>Select and Annotate Columns</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption(f"📊 Detected **{len(num_cols)}** quantitative and **{len(meta_cols)}** metadata columns")
        
        # Get defaults for auto-selection
        default_species_cols = get_default_species_mapping_cols(preview_df)
        default_group_cols = get_default_group_col(preview_df, meta_cols)
        default_peptide_cols = get_default_peptide_id_col(preview_df, meta_cols) if kind == "peptide" else []
        
        # Build annotation dataframe
        if kind == "protein":
            # Protein: Include | Trimmed Name | Protein Group | Species Mapping | Control | Treatment
            annotation_data = []
            
            # Metadata columns first
            for col in meta_cols:
                is_protein_group = col in (default_group_cols or [])
                is_species = col in (default_species_cols or [])
                annotation_data.append({
                    'Include': True,
                    'Trimmed Name': trimmed_meta[col],
                    'Original Name': col,
                    'Protein Group': is_protein_group,
                    'Species Mapping': is_species,
                    'Control': False,
                    'Treatment': False
                })
            
            # Quantitative columns
            for idx, col in enumerate(num_cols):
                # Auto-split: first half control, second half treatment
                is_control = idx < len(num_cols) // 2
                is_treatment = not is_control
                annotation_data.append({
                    'Include': True,
                    'Trimmed Name': trimmed_quant[col],
                    'Original Name': col,
                    'Protein Group': False,
                    'Species Mapping': False,
                    'Control': is_control,
                    'Treatment': is_treatment
                })
            
            col_order = ['Include', 'Trimmed Name', 'Protein Group', 'Species Mapping', 'Control', 'Treatment']
            
        else:  # peptide
            # Peptide: Include | Trimmed Name | Protein Group | Species Mapping | Peptide | Control | Treatment
            annotation_data = []
            
            # Metadata columns first
            for col in meta_cols:
                is_protein_group = col in (default_group_cols or [])
                is_species = col in (default_species_cols or [])
                is_peptide = col in (default_peptide_cols or [])
                annotation_data.append({
                    'Include': True,
                    'Trimmed Name': trimmed_meta[col],
                    'Original Name': col,
                    'Protein Group': is_protein_group,
                    'Species Mapping': is_species,
                    'Peptide': is_peptide,
                    'Control': False,
                    'Treatment': False
                })
            
            # Quantitative columns
            for idx, col in enumerate(num_cols):
                is_control = idx < len(num_cols) // 2
                is_treatment = not is_control
                annotation_data.append({
                    'Include': True,
                    'Trimmed Name': trimmed_quant[col],
                    'Original Name': col,
                    'Protein Group': False,
                    'Species Mapping': False,
                    'Peptide': False,
                    'Control': is_control,
                    'Treatment': is_treatment
                })
            
            col_order = ['Include', 'Trimmed Name', 'Protein Group', 'Species Mapping', 'Peptide', 'Control', 'Treatment']
        
        annotation_df = pd.DataFrame(annotation_data)[col_order + ['Original Name']]
        
        st.markdown("**Select columns and assign roles:**")
        st.caption("ℹ️ Uncheck 'Include' to drop a column. Check boxes to assign roles. Control/Treatment are mutually exclusive for quantitative columns.")
        
        # Configure column display
        if kind == "protein":
            column_config = {
                'Include': st.column_config.CheckboxColumn('Include', help='Uncheck to exclude', default=True),
                'Trimmed Name': st.column_config.TextColumn('Trimmed Name', help='Cleaned name', width='medium'),
                'Protein Group': st.column_config.CheckboxColumn('Protein Group', help='Use for grouping', default=False),
                'Species Mapping': st.column_config.CheckboxColumn('Species Mapping', help='Use for species', default=False),
                'Control': st.column_config.CheckboxColumn('Control', help='Control sample', default=False),
                'Treatment': st.column_config.CheckboxColumn('Treatment', help='Treatment sample', default=False),
                'Original Name': st.column_config.TextColumn('Original Name', help='Original column name', width='small')
            }
        else:
            column_config = {
                'Include': st.column_config.CheckboxColumn('Include', help='Uncheck to exclude', default=True),
                'Trimmed Name': st.column_config.TextColumn('Trimmed Name', help='Cleaned name', width='medium'),
                'Protein Group': st.column_config.CheckboxColumn('Protein Group', help='Use for grouping', default=False),
                'Species Mapping': st.column_config.CheckboxColumn('Species Mapping', help='Use for species', default=False),
                'Peptide': st.column_config.CheckboxColumn('Peptide', help='Peptide ID', default=False),
                'Control': st.column_config.CheckboxColumn('Control', help='Control sample', default=False),
                'Treatment': st.column_config.CheckboxColumn('Treatment', help='Treatment sample', default=False),
                'Original Name': st.column_config.TextColumn('Original Name', help='Original column name', width='small')
            }
        
        edited_df = st.data_editor(
            annotation_df,
            hide_index=True,
            width='stretch',  # FIXED: Updated from use_container_width
            column_config=column_config,
            key=f"col_annotate_{kind}"
        )
        
        # Validate selections
        included_rows = edited_df[edited_df['Include']]
        
        if len(included_rows) == 0:
            st.error("❌ You must include at least one column")
            return
        
        # Extract selections
        protein_group_cols = included_rows[included_rows['Protein Group']]['Original Name'].tolist()
        species_cols = included_rows[included_rows['Species Mapping']]['Original Name'].tolist()
        
        if kind == "peptide":
            peptide_cols = included_rows[included_rows['Peptide']]['Original Name'].tolist()
        
        control_cols = included_rows[included_rows['Control']]['Original Name'].tolist()
        treatment_cols = included_rows[included_rows['Treatment']]['Original Name'].tolist()
        
        # Validation messages
        warnings = []
        if len(protein_group_cols) == 0:
            warnings.append("⚠️ No Protein Group column selected")
        if len(species_cols) == 0:
            warnings.append("⚠️ No Species Mapping column selected")
        if kind == "peptide" and len(peptide_cols) == 0:
            warnings.append("⚠️ No Peptide column selected")
        if len(control_cols) == 0 and len(treatment_cols) == 0:
            warnings.append("⚠️ No Control or Treatment samples selected")
        
        for warn in warnings:
            st.warning(warn)
        
        # Show summary
        dropped_count = len(annotation_df) - len(included_rows)
        if dropped_count > 0:
            st.info(f"ℹ️ {dropped_count} column(s) will be excluded")
        
        st.success(f"✓ {len(included_rows)} columns selected | {len(control_cols)} Control | {len(treatment_cols)} Treatment")
        
        # Button to load full dataset
        if st.button(f"Load Full Dataset ({kind})", key=f"load_full_{kind}"):
            if len(warnings) > 0:
                st.error("❌ Please fix warnings before loading")
                return
            
            # Read FULL file with only selected columns
            user_file.seek(0)
            cols_to_read = included_rows['Original Name'].tolist()
            full_df = pd.read_csv(user_file, sep=sep, usecols=cols_to_read)
            
            # Apply trimmed names to dataframe
            rename_map = dict(zip(included_rows['Original Name'], included_rows['Trimmed Name']))
            full_df = full_df.rename(columns=rename_map)
            
            # CRITICAL FIX: Handle duplicate columns after renaming
            if not full_df.columns.is_unique:
                st.warning("⚠️ Some columns had identical names after trimming. Adding numeric suffixes to make them unique.")
                cols = pd.Series(full_df.columns)
                for dup in cols[cols.duplicated()].unique():
                    indices = cols[cols == dup].index.values.tolist()
                    cols.iloc[indices] = [f"{dup}_{i+1}" if i > 0 else dup for i in range(len(indices))]
                full_df.columns = cols.tolist()
                
                # Update rename_map with new unique names
                rename_map = dict(zip(included_rows['Original Name'], full_df.columns))
            
            st.session_state[preview_key] = {
                'df': full_df,
                'protein_group_col': rename_map.get(protein_group_cols[0]) if protein_group_cols else None,
                'species_col': rename_map.get(species_cols[0]) if species_cols else None,
                'peptide_col': rename_map.get(peptide_cols[0]) if kind == "peptide" and peptide_cols else None,
                'control_cols': [rename_map.get(c) for c in control_cols if rename_map.get(c)],
                'treatment_cols': [rename_map.get(c) for c in treatment_cols if rename_map.get(c)],
                'original_to_trimmed': rename_map
            }
            st.success(f"✓ Full dataset loaded: **{len(full_df):,} rows**")
            st.rerun()
        
        # Continue if full dataset is loaded
        if st.session_state.get(preview_key) is None:
            st.markdown("""
            <div class="status-warning">
              <strong>⏸️ Waiting</strong><br>
              Click "Load Full Dataset" to continue.
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Extract loaded data
        loaded_data = st.session_state[preview_key]
        df = loaded_data['df']
        
        # Build condition mapping
        condition_map = {}
        for col in loaded_data['control_cols']:
            condition_map[col] = 'Control'
        for col in loaded_data['treatment_cols']:
            condition_map[col] = 'Treatment'
        
        # Store final result (SEPARATE for protein and peptide)
        result = {
            'data': df,
            'quant_cols': loaded_data['control_cols'] + loaded_data['treatment_cols'],
            'quant_cols_trimmed': {col: col for col in (loaded_data['control_cols'] + loaded_data['treatment_cols'])},  # Already trimmed
            'meta_cols': [c for c in df.columns if c not in condition_map],
            'species_col': loaded_data['species_col'],
            'group_col': loaded_data['protein_group_col'],
            'condition': condition_map
        }
        
        if kind == "peptide":
            result['peptide_id_col'] = loaded_data['peptide_col']
        
        st.session_state[session_key] = result
        
        # Success message
        st.markdown(f"""
        <div class="status-success">
          <strong>✓ Complete!</strong> {kind.capitalize()}-level data ready for analysis.<br>
          <small>{len(result['quant_cols'])} quantitative columns | {len(result['meta_cols'])} metadata columns</small>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# MAIN: TWO-COLUMN UPLOAD INTERFACE
# ============================================================================
st.info("📝 **Instructions:** Follow the step-by-step process to upload and configure your data. Column names will be automatically trimmed.")

st.markdown("---")

colA, colB = st.columns(2, gap="large")

# Protein upload (left column)
upload_annotation_block(
    "protein",
    id_keys={
        'quant_cols': 'quant_cols_prot',
        'spec_map': 'spec_map_prot',
        'group_col': 'group_col_prot',
        'mode': 'mode_prot'
    },
    col=colA
)

# Peptide upload (right column)
upload_annotation_block(
    "peptide",
    id_keys={
        'quant_cols': 'quant_cols_pept',
        'spec_map': 'spec_map_pept',
        'group_col': 'group_col_pept',
        'peptid_col': 'peptid_col_pept',
        'mode': 'mode_pept'
    },
    col=colB
)

# ============================================================================
# SUMMARY PANEL
# ============================================================================
st.markdown("---")

if st.session_state.get('protein_upload') or st.session_state.get('peptide_upload'):
    st.subheader("📊 Upload Summary")
    st.caption("Overview of loaded datasets ready for downstream analysis")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        if st.session_state.get('protein_upload'):
            prot_data = st.session_state.protein_upload
            st.success("✅ **Protein-level data loaded**")
            
            # Metrics
            met1, met2, met3 = st.columns(3)
            with met1:
                st.metric("Proteins", f"{len(prot_data['data']):,}")
            with met2:
                st.metric("Quant Columns", len(prot_data['quant_cols']))
            with met3:
                st.metric("Metadata Columns", len(prot_data['meta_cols']))
            
            # Column assignments
            st.markdown("**Column Assignments:**")
            st.write(f"🔹 **Protein Group:** {prot_data['group_col']}")
            st.write(f"🔹 **Species Mapping:** {prot_data['species_col']}")
            
            # Condition breakdown
            st.markdown("**Sample Assignments:**")
            for col, condition in prot_data['condition'].items():
                emoji = "🟦" if condition == "Control" else "🟥"
                st.write(f"{emoji} **{col}** → {condition}")
            
            conditions = prot_data['condition']
            n_control = sum(1 for v in conditions.values() if v == "Control")
            n_treatment = sum(1 for v in conditions.values() if v == "Treatment")
            st.info(f"**Total:** {n_control} Control | {n_treatment} Treatment")
            
            with st.expander("📋 Preview data"):
                st.dataframe(prot_data['data'].head(10), width='stretch')  # FIXED
    
    with col2:
        if st.session_state.get('peptide_upload'):
            pept_data = st.session_state.peptide_upload
            st.success("✅ **Peptide-level data loaded**")
            
            # Metrics
            met1, met2, met3 = st.columns(3)
            with met1:
                st.metric("Peptides", f"{len(pept_data['data']):,}")
            with met2:
                st.metric("Quant Columns", len(pept_data['quant_cols']))
            with met3:
                st.metric("Metadata Columns", len(pept_data['meta_cols']))
            
            # Column assignments
            st.markdown("**Column Assignments:**")
            st.write(f"🔹 **Protein Group:** {pept_data['group_col']}")
            st.write(f"🔹 **Species Mapping:** {pept_data['species_col']}")
            st.write(f"🔹 **Peptide ID:** {pept_data.get('peptide_id_col', 'N/A')}")
            
            # Condition breakdown
            st.markdown("**Sample Assignments:**")
            for col, condition in pept_data['condition'].items():
                emoji = "🟦" if condition == "Control" else "🟥"
                st.write(f"{emoji} **{col}** → {condition}")
            
            conditions = pept_data['condition']
            n_control = sum(1 for v in conditions.values() if v == "Control")
            n_treatment = sum(1 for v in conditions.values() if v == "Treatment")
            st.info(f"**Total:** {n_control} Control | {n_treatment} Treatment")
            
            with st.expander("📋 Preview data"):
                st.dataframe(pept_data['data'].head(10), width='stretch')  # FIXED

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div class="footer">
  <strong>Proprietary & Confidential | For Internal Use Only</strong>
  <p>© 2024 Thermo Fisher Scientific Inc. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
