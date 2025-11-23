"""
Thermo Fisher Proteomics App - Module 1: Data Import
UPDATED: Theme-aware design, step-by-step process, pre-load column selection
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
# REUSABLE UPLOAD BLOCK FUNCTION (UPDATED WITH STEP-BY-STEP)
# ============================================================================
def upload_annotation_block(kind, id_keys, col):
    """
    Step-by-step upload with column selection BEFORE full load
    
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
        
        st.markdown("---")
        
        # STEP 2: Column Selection (BEFORE full load)
        st.markdown(f"""
        <div class="step-header">
          <span class="step-indicator">2</span>
          <span>Select Columns to Keep</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption(f"📊 Detected **{len(num_cols)}** quantitative columns")
        
        # Auto-trim for display
        trimmed_names = trim_column_names(num_cols)
        
        # Interactive column selection with checkboxes
        st.markdown("**Select quantitative columns to include in analysis:**")
        
        # Create selection dataframe
        col_selection_df = pd.DataFrame({
            'Include': [True] * len(num_cols),
            'Trimmed Name': [trimmed_names[col] for col in num_cols],
            'Original Name': num_cols
        })
        
        edited_df = st.data_editor(
            col_selection_df,
            hide_index=True,
            use_container_width=True,
            disabled=['Trimmed Name', 'Original Name'],
            column_config={
                'Include': st.column_config.CheckboxColumn(
                    'Include',
                    help='Uncheck to exclude column',
                    default=True
                ),
                'Trimmed Name': st.column_config.TextColumn(
                    'Trimmed Name',
                    help='Cleaned column name'
                ),
                'Original Name': st.column_config.TextColumn(
                    'Original Name',
                    help='Original column name from file'
                )
            },
            key=f"col_select_{kind}"
        )
        
        # Get selected columns
        selected_mask = edited_df['Include'].values
        quant_cols_sel = [col for col, include in zip(num_cols, selected_mask) if include]
        dropped_count = len(num_cols) - len(quant_cols_sel)
        
        if dropped_count > 0:
            st.warning(f"⚠️ {dropped_count} column(s) will be excluded from analysis")
        
        if len(quant_cols_sel) == 0:
            st.error("❌ You must select at least one quantitative column")
            return
        
        st.success(f"✓ {len(quant_cols_sel)} columns selected")
        
        # Button to load full dataset
        if st.button(f"Load Full Dataset ({kind})", key=f"load_full_{kind}"):
            # Now read the FULL file with only selected columns
            user_file.seek(0)  # Reset file pointer
            
            # Read with selected columns only
            cols_to_read = quant_cols_sel + meta_cols
            full_df = pd.read_csv(user_file, sep=sep, usecols=cols_to_read)
            
            st.session_state[preview_key] = full_df
            st.success(f"✓ Full dataset loaded: **{len(full_df):,} rows**")
        
        # Continue if full dataset is loaded
        if st.session_state.get(preview_key) is None:
            st.markdown("""
            <div class="status-warning">
              <strong>⏸️ Waiting</strong><br>
              Click "Load Full Dataset" to continue with selected columns.
            </div>
            """, unsafe_allow_html=True)
            return
        
        df = st.session_state[preview_key]
        
        st.markdown("---")
        
        # STEP 3: Species mapping column
        st.markdown(f"""
        <div class="step-header">
          <span class="step-indicator">3</span>
          <span>Select Species Mapping Column</span>
        </div>
        """, unsafe_allow_html=True)
        
        mapping_options = get_default_species_mapping_cols(df) or meta_cols
        mapping_sel = st.selectbox(
            "Species column",
            mapping_options,
            key=id_keys['spec_map']
        )
        
        # STEP 4: Protein group column
        st.markdown(f"""
        <div class="step-header">
          <span class="step-indicator">4</span>
          <span>Select Protein Group Column</span>
        </div>
        """, unsafe_allow_html=True)
        
        group_col_default = get_default_group_col(df, meta_cols)
        group_col_sel = st.selectbox(
            "Protein group",
            group_col_default if group_col_default else meta_cols,
            key=id_keys['group_col']
        )
        
        # STEP 5: Peptide-specific
        peptid_sel = None
        if kind == "peptide":
            st.markdown(f"""
            <div class="step-header">
              <span class="step-indicator">5</span>
              <span>Select Peptide Identifier Column</span>
            </div>
            """, unsafe_allow_html=True)
            
            peptid_default = get_default_peptide_id_col(df, meta_cols)
            peptid_sel = st.selectbox(
                "Peptide/Precursor ID",
                peptid_default if peptid_default else meta_cols,
                key=id_keys.get('peptid_col')
            )
        
        st.markdown("---")
        
        # STEP 6: Assign Control/Treatment
        step_num = "6" if kind == "peptide" else "5"
        st.markdown(f"""
        <div class="step-header">
          <span class="step-indicator">{step_num}</span>
          <span>Assign Control/Treatment Conditions</span>
        </div>
        """, unsafe_allow_html=True)
        
        mode = st.radio(
            "Assignment mode",
            ["Auto-split", "All Control", "All Treatment", "Manual"],
            horizontal=True,
            key=id_keys['mode']
        )
        
        # Build annotation
        annot = []
        if mode == "Auto-split":
            annot = ["Control" if i < len(quant_cols_sel)//2 else "Treatment"
                    for i in range(len(quant_cols_sel))]
        elif mode == "All Control":
            annot = ["Control"] * len(quant_cols_sel)
        elif mode == "All Treatment":
            annot = ["Treatment"] * len(quant_cols_sel)
        else:  # Manual
            st.write("**Assign each column:**")
            for q in quant_cols_sel:
                trimmed = trimmed_names[q]
                annot.append(st.selectbox(
                    f"{trimmed}",
                    options=["Control", "Treatment"],
                    key=f"man_assign_{kind}_{q}"
                ))
        
        # Store result
        result = {
            'data': df,
            'quant_cols': quant_cols_sel,
            'quant_cols_trimmed': {col: trimmed_names[col] for col in quant_cols_sel},
            'meta_cols': meta_cols,
            'species_col': mapping_sel,
            'group_col': group_col_sel,
            'condition': dict(zip(quant_cols_sel, annot))
        }
        
        if kind == "peptide":
            result['peptide_id_col'] = peptid_sel
        
        st.session_state[session_key] = result
        
        # Success message
        st.markdown(f"""
        <div class="status-success">
          <strong>✓ Complete!</strong> {kind.capitalize()}-level data ready for analysis.<br>
          <small>{len(quant_cols_sel)} columns selected, {dropped_count} columns excluded.</small>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# MAIN: TWO-COLUMN UPLOAD INTERFACE
# ============================================================================
st.info("📝 **Instructions:** Follow the step-by-step process to upload and configure your data.")

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
            
            # Condition breakdown
            st.markdown("**Sample Assignments:**")
            trimmed = prot_data.get('quant_cols_trimmed', {})
            for orig_col, condition in prot_data['condition'].items():
                display_name = trimmed.get(orig_col, orig_col)
                emoji = "🟦" if condition == "Control" else "🟥"
                st.write(f"{emoji} **{display_name}** → {condition}")
            
            conditions = prot_data['condition']
            n_control = sum(1 for v in conditions.values() if v == "Control")
            n_treatment = sum(1 for v in conditions.values() if v == "Treatment")
            st.info(f"**Total:** {n_control} Control | {n_treatment} Treatment")
            
            with st.expander("📋 Preview data"):
                st.dataframe(prot_data['data'].head(10), use_container_width=True)
    
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
            
            # Condition breakdown
            st.markdown("**Sample Assignments:**")
            trimmed = pept_data.get('quant_cols_trimmed', {})
            for orig_col, condition in pept_data['condition'].items():
                display_name = trimmed.get(orig_col, orig_col)
                emoji = "🟦" if condition == "Control" else "🟥"
                st.write(f"{emoji} **{display_name}** → {condition}")
            
            conditions = pept_data['condition']
            n_control = sum(1 for v in conditions.values() if v == "Control")
            n_treatment = sum(1 for v in conditions.values() if v == "Treatment")
            st.info(f"**Total:** {n_control} Control | {n_treatment} Treatment")
            
            with st.expander("📋 Preview data"):
                st.dataframe(pept_data['data'].head(10), use_container_width=True)

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
