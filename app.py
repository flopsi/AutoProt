"""
Thermo Fisher Proteomics App - Module 1: Data Import
Two-column upload system with full Thermo Fisher branding
"""
import streamlit as st
import pandas as pd
from config import (
    get_numeric_columns,
    get_metadata_columns,
    get_default_species_mapping_cols,
    get_default_group_col,
    get_default_peptide_id_col,
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
# THERMO FISHER CSS & BRANDING
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

/* Global styles */
body {{
  font-family: Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background-color: #f8f9fa;
  color: var(--primary-gray);
}}

[data-testid="stAppViewContainer"] {{
  background-color: #f8f9fa;
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

/* Module Headers */
.module-header {{
  background: linear-gradient(90deg, var(--primary-red) 0%, var(--dark-red) 100%);
  padding: 20px 30px;
  border-radius: 8px;
  margin: 20px 0;
  color: white;
}}

.module-header h2 {{
  margin: 0;
  font-size: 20px;
  color: white;
}}

.module-header p {{
  margin: 5px 0 0 0;
  opacity: 0.9;
  font-size: 13px;
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
}}

.status-info {{
  background-color: rgba(38, 34, 98, 0.15);
  border-left: 4px solid var(--navy);
  padding: 15px;
  border-radius: 4px;
  margin: 15px 0;
}}

/* Upload columns */
.upload-section {{
  background-color: white;
  border: 1px solid var(--light-gray);
  border-radius: 8px;
  padding: 25px;
}}

/* Metrics */
.stMetric {{
  background-color: white;
  padding: 15px;
  border-radius: 8px;
  border: 1px solid var(--light-gray);
}}

/* Footer */
.footer {{
  text-align: center;
  padding: 30px 0;
  color: var(--primary-gray);
  font-size: 12px;
  border-top: 1px solid var(--light-gray);
  margin-top: 60px;
}}

.footer strong {{
  display: block;
  margin-bottom: 10px;
}}

/* Dividers */
hr {{
  border: none;
  border-top: 1px solid var(--light-gray);
  margin: 30px 0;
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

# ============================================================================
# REUSABLE UPLOAD BLOCK FUNCTION
# ============================================================================
def upload_annotation_block(kind, help_txt, id_keys, col):
    """
    Reusable upload and annotation interface
    
    Args:
        kind: "protein" or "peptide"
        help_txt: Help text when no file uploaded
        id_keys: Dict with widget key IDs
        col: Streamlit column to render in
    """
    with col:
        # Section header
        st.markdown(f"""
        <div class="module-header">
          <h2>{kind.capitalize()}-Level Upload</h2>
          <p>Upload and annotate {kind}-level quantification data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        user_file = st.file_uploader(
            f"Upload {kind}-level file",
            key=f"upl_{kind}",
            type=['csv', 'tsv', 'txt']
        )
        
        session_key = f"{kind}_upload"
        
        if user_file:
            # Read file
            sep = '\t' if user_file.name.endswith(('.tsv', '.txt')) else ','
            df = pd.read_csv(user_file, sep=sep)
            
            st.write(f"**File loaded:** {user_file.name}")
            
            # Detect columns
            num_cols = get_numeric_columns(df)
            meta_cols = get_metadata_columns(df, num_cols)
            
            # Show metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            
            st.markdown("---")
            
            # 1. Select quantitative columns
            st.markdown("**1️⃣ Select quantitative columns to keep**")
            quant_cols_sel = st.multiselect(
                "Quant columns",
                num_cols,
                default=num_cols,
                key=id_keys['quant_cols']
            )
            
            # 2. Species mapping column
            st.markdown("**2️⃣ Select species mapping column**")
            mapping_options = get_default_species_mapping_cols(df) or meta_cols
            mapping_sel = st.selectbox(
                "Species column",
                mapping_options,
                key=id_keys['spec_map']
            )
            
            # 3. Protein group column
            st.markdown("**3️⃣ Select protein group column**")
            group_col_default = get_default_group_col(df, meta_cols)
            group_col_sel = st.selectbox(
                "Protein group",
                group_col_default if group_col_default else meta_cols,
                key=id_keys['group_col']
            )
            
            # 4. Peptide-specific: peptide identifier
            peptid_sel = None
            if kind == "peptide":
                st.markdown("**4️⃣ Select peptide identifier column**")
                peptid_default = get_default_peptide_id_col(df, meta_cols)
                peptid_sel = st.selectbox(
                    "Peptide/Precursor ID",
                    peptid_default if peptid_default else meta_cols,
                    key=id_keys.get('peptid_col')
                )
            
            st.markdown("---")
            
            # 5. Assign Control/Treatment
            st.markdown("**5️⃣ Assign Control/Treatment conditions**")
            
            mode = st.radio(
                "Assignment mode",
                ["Auto-split", "All Control", "All Treatment", "Manual"],
                horizontal=True,
                key=id_keys['mode']
            )
            
            # Build annotation
            annot = []
            if mode == "Auto-split":
                # First half = Control, second half = Treatment
                annot = ["Control" if i < len(quant_cols_sel)//2 else "Treatment"
                        for i in range(len(quant_cols_sel))]
            elif mode == "All Control":
                annot = ["Control"] * len(quant_cols_sel)
            elif mode == "All Treatment":
                annot = ["Treatment"] * len(quant_cols_sel)
            else:  # Manual
                for q in quant_cols_sel:
                    annot.append(st.selectbox(
                        f"{q}",
                        options=["Control", "Treatment"],
                        key=f"man_assign_{kind}_{q}"
                    ))
            
            # Store result in session state
            result = {
                'data': df,
                'quant_cols': quant_cols_sel,
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
              <strong>✓ Success!</strong> {kind.capitalize()}-level data loaded and annotated.
            </div>
            """, unsafe_allow_html=True)
            
        else:
            # No file uploaded - show help text
            st.markdown(f"""
            <div class="status-info">
              <strong>ℹ Info</strong><br>
              {help_txt}
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# MAIN: TWO-COLUMN UPLOAD INTERFACE
# ============================================================================
st.info("📝 **Instructions:** Upload protein-level and/or peptide-level data. You may upload either one or both.")

st.markdown("---")

colA, colB = st.columns(2, gap="large")

# Protein upload (left column)
upload_annotation_block(
    "protein",
    "Upload a protein-level quantification file. You may upload either or both levels below.",
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
    "Upload a peptide-level quantification file (optional, for advanced statistics). You may upload either or both levels below.",
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
    st.markdown("""
    <div class="module-header">
      <h2>📊 Upload Summary</h2>
      <p>Overview of loaded datasets ready for downstream analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
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
            conditions = prot_data['condition']
            n_control = sum(1 for v in conditions.values() if v == "Control")
            n_treatment = sum(1 for v in conditions.values() if v == "Treatment")
            st.write(f"**Conditions:** {n_control} Control | {n_treatment} Treatment")
            
            # Preview
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
            conditions = pept_data['condition']
            n_control = sum(1 for v in conditions.values() if v == "Control")
            n_treatment = sum(1 for v in conditions.values() if v == "Treatment")
            st.write(f"**Conditions:** {n_control} Control | {n_treatment} Treatment")
            
            # Preview
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
