"""
Complete Thermo Fisher Branded Proteomics App
Integrated design with two-column upload system
"""
import streamlit as st
import pandas as pd
from config import (
    get_numeric_columns, 
    get_metadata_columns,
    get_default_species_mapping_cols,
    get_default_group_col,
    get_default_peptide_id_col
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Proteomics Data Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# THERMO FISHER BRANDING & CSS
# ============================================================================
st.markdown("""
<style>
/* Thermo Fisher Brand Colors */
:root {
  --primary-red: #E71316;
  --primary-gray: #54585A;
  --primary-white: #FFFFFF;
  --light-gray: #E2E3E4;
  --navy: #262262;
  --dark-red: #A6192E;
  --orange: #EA7600;
  --yellow: #F1B434;
  --green: #B5BD00;
  --sky: #9BD3DD;
}

/* Global Styles */
body {
  font-family: Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background-color: #f8f9fa;
  color: var(--primary-gray);
}

/* Streamlit Override */
[data-testid="stAppViewContainer"] {
  background-color: #f8f9fa;
}

/* Header Styling */
.header-banner {
  background: linear-gradient(90deg, var(--primary-red) 0%, var(--dark-red) 100%);
  padding: 30px 40px;
  border-radius: 0;
  margin-bottom: 30px;
  color: white;
}

.header-banner h1 {
  margin: 0;
  font-size: 28pt;
  color: white;
}

.header-banner p {
  margin: 5px 0 0 0;
  font-size: 14px;
  opacity: 0.95;
}

/* Module Section Headers */
.module-header {
  background: linear-gradient(90deg, var(--primary-red) 0%, var(--dark-red) 100%);
  padding: 20px 30px;
  border-radius: 8px;
  margin: 30px 0 20px 0;
  color: white;
}

.module-header h2 {
  margin: 0;
  font-size: 22px;
  color: white;
}

.module-header p {
  margin: 5px 0 0 0;
  opacity: 0.9;
  font-size: 14px;
}

/* Upload Columns */
.upload-column {
  background-color: white;
  border: 1px solid var(--light-gray);
  border-radius: 8px;
  padding: 25px;
  height: 100%;
}

/* Buttons */
.stButton button {
  background-color: var(--primary-red);
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 6px;
  font-weight: 500;
  transition: background-color 0.3s;
}

.stButton button:hover {
  background-color: var(--dark-red);
}

/* Status Messages */
.status-success {
  background-color: rgba(181, 189, 0, 0.15);
  border-left: 4px solid var(--green);
  padding: 15px;
  border-radius: 4px;
  margin: 10px 0;
}

.status-info {
  background-color: rgba(38, 34, 98, 0.15);
  border-left: 4px solid var(--navy);
  padding: 15px;
  border-radius: 4px;
  margin: 10px 0;
}

/* Footer */
.footer {
  text-align: center;
  padding: 30px;
  color: var(--primary-gray);
  font-size: 12px;
  border-top: 1px solid var(--light-gray);
  margin-top: 60px;
}

.footer strong {
  display: block;
  margin-bottom: 10px;
}
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
# UPLOAD ANNOTATION BLOCK (REUSABLE FUNCTION)
# ============================================================================
def upload_annotation_block(kind, help_txt, id_keys, col):
    """
    Reusable upload + annotation block for protein or peptide level
    
    Args:
        kind: "protein" or "peptide"
        help_txt: Info text to display
        id_keys: Dict with widget key names
        col: Streamlit column object to render in
    """
    with col:
        st.markdown(f"""
        <div class="module-header">
          <h2>{kind.capitalize()}-Level Upload</h2>
          <p>Upload and annotate your {kind}-level quantification data</p>
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
            
            st.success(f"✓ File loaded: {user_file.name}")
            st.write(f"**{len(df):,} rows × {len(df.columns)} columns**")
            
            # Column detection
            num_cols = get_numeric_columns(df)
            meta_cols = get_metadata_columns(df, num_cols)
            
            st.markdown("---")
            st.markdown("**1️⃣ Select Quantitative Columns**")
            quant_cols_sel = st.multiselect(
                "Quant columns",
                num_cols,
                default=num_cols,
                key=id_keys['quant_cols']
            )
            
            st.markdown("**2️⃣ Select Species Mapping Column**")
            mapping_options = get_default_species_mapping_cols(df) or meta_cols
            mapping_sel = st.selectbox(
                "Species column",
                mapping_options,
                key=id_keys['spec_map']
            )
            
            st.markdown("**3️⃣ Select Protein Group Column**")
            group_col_default = get_default_group_col(df, meta_cols)
            group_col_sel = st.selectbox(
                "Protein group",
                group_col_default if group_col_default else meta_cols,
                key=id_keys['group_col']
            )
            
            # Peptide-specific
            peptid_sel = None
            if kind == "peptide":
                st.markdown("**4️⃣ Select Peptide Identifier Column**")
                peptid_default = get_default_peptide_id_col(df, meta_cols)
                peptid_sel = st.selectbox(
                    "Peptide/Precursor ID",
                    peptid_default if peptid_default else meta_cols,
                    key=id_keys.get('peptid_col')
                )
            
            st.markdown("---")
            st.markdown("**5️⃣ Assign Control/Treatment Conditions**")
            
            # Assignment mode
            mode = st.radio(
                "Assignment mode",
                ["Auto-split", "All Control", "All Treatment", "Manual"],
                horizontal=True,
                key=id_keys['mode']
            )
            
            # Assign conditions
            annot = []
            if mode == "Auto-split":
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
            
            # Store in session state
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
            
            st.markdown(f"""
            <div class="status-success">
              <strong>✓ Success!</strong> {kind.capitalize()}-level data loaded and annotated.
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown(f"""
            <div class="status-info">
              <strong>ℹ Info</strong><br>
              {help_txt}
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# TWO-COLUMN UPLOAD LAYOUT
# ============================================================================
st.markdown("""
<div class="module-header">
  <h2>Data Upload</h2>
  <p>Upload protein-level and/or peptide-level quantification matrices</p>
</div>
""", unsafe_allow_html=True)

st.info("📝 **Instructions:** You may upload either protein-level, peptide-level, or both. Each file will be processed independently.")

col_A, col_B = st.columns(2, gap="large")

# Protein upload
upload_annotation_block(
    "protein",
    "Upload a protein-level quantification file. You may upload either or both levels below.",
    id_keys={
        'quant_cols': 'quant_cols_prot',
        'spec_map': 'spec_map_prot',
        'group_col': 'group_col_prot',
        'mode': 'mode_prot'
    },
    col=col_A
)

# Peptide upload
upload_annotation_block(
    "peptide",
    "Upload a peptide-level quantification file (optional, for advanced stats). You may upload either or both levels below.",
    id_keys={
        'quant_cols': 'quant_cols_pept',
        'spec_map': 'spec_map_pept',
        'group_col': 'group_col_pept',
        'peptid_col': 'peptid_col_pept',
        'mode': 'mode_pept'
    },
    col=col_B
)

# ============================================================================
# SUMMARY PANEL
# ============================================================================
st.markdown("---")

if st.session_state.get('protein_upload') or st.session_state.get('peptide_upload'):
    st.markdown("""
    <div class="module-header">
      <h2>📊 Upload Summary</h2>
      <p>Overview of loaded datasets</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.get('protein_upload'):
            prot_data = st.session_state.protein_upload
            st.success("✅ **Protein-level data loaded**")
            st.metric("Proteins", f"{len(prot_data['data']):,}")
            st.metric("Quant Columns", len(prot_data['quant_cols']))
            
            # Show condition breakdown
            conditions = prot_data['condition']
            n_control = sum(1 for v in conditions.values() if v == "Control")
            n_treatment = sum(1 for v in conditions.values() if v == "Treatment")
            st.write(f"**Conditions:** {n_control} Control, {n_treatment} Treatment")
            
            with st.expander("Preview"):
                st.dataframe(prot_data['data'].head(5), use_container_width=True)
    
    with col2:
        if st.session_state.get('peptide_upload'):
            pept_data = st.session_state.peptide_upload
            st.success("✅ **Peptide-level data loaded**")
            st.metric("Peptides", f"{len(pept_data['data']):,}")
            st.metric("Quant Columns", len(pept_data['quant_cols']))
            
            # Show condition breakdown
            conditions = pept_data['condition']
            n_control = sum(1 for v in conditions.values() if v == "Control")
            n_treatment = sum(1 for v in conditions.values() if v == "Treatment")
            st.write(f"**Conditions:** {n_control} Control, {n_treatment} Treatment")
            
            with st.expander("Preview"):
                st.dataframe(pept_data['data'].head(5), use_container_width=True)

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
