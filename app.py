import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

# Import from config
from config import (
    detect_column_types,
    detect_data_level,
    auto_assign_conditions,
    PRIMARY_RED,
    DARK_RED,
    PRIMARY_GRAY,
    LIGHT_GRAY,
    SKY,
    GREEN,
    ORANGE,
    get_condition_colors,
    MISSING_VALUES
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="DIA Proteomics Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'protein_data' not in st.session_state:
    st.session_state.protein_data = None
if 'peptide_data' not in st.session_state:
    st.session_state.peptide_data = None
if 'protein_annotations' not in st.session_state:
    st.session_state.protein_annotations = {}
if 'protein_metadata_cols' not in st.session_state:
    st.session_state.protein_metadata_cols = []
if 'protein_quant_cols' not in st.session_state:
    st.session_state.protein_quant_cols = []
if 'protein_data_level' not in st.session_state:
    st.session_state.protein_data_level = None

# ============================================================
# CACHED CSS
# ============================================================
@st.cache_resource
def load_css():
    return f"""
    <style>
        * {{font-family: Arial, sans-serif;}}
        .module-header {{
            background: linear-gradient(90deg, {PRIMARY_RED} 0%, {DARK_RED} 100%);
            padding: 30px; border-radius: 8px; margin-bottom: 40px; color: white;
        }}
        .stButton > button[kind="primary"] {{
            background-color: {PRIMARY_RED}; color: white; padding: 12px 24px;
            border-radius: 6px; font-weight: 500; border: none;
        }}
        .stButton > button[kind="primary"]:hover {{background-color: {DARK_RED};}}
        .data-level-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            margin: 10px 0;
        }}
        .badge-protein {{
            background-color: rgba(155, 211, 221, 0.2);
            border: 2px solid {SKY};
            color: {PRIMARY_GRAY};
        }}
        .badge-peptide {{
            background-color: rgba(231, 19, 22, 0.1);
            border: 2px solid {PRIMARY_RED};
            color: {PRIMARY_GRAY};
        }}
    </style>
    """

st.markdown(load_css(), unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def clean_intensity_data(df, quant_cols):
    """Treat 0, 1, and NaN as missing values"""
    df_clean = df[quant_cols].copy()
    df_clean = df_clean.replace(MISSING_VALUES, np.nan)
    return df_clean

# ============================================================
# PLOTTING FUNCTIONS (unchanged)
# ============================================================
@st.cache_data
def plot_intensity_distribution(df, quant_cols, annotations):
    """Intensity distribution violin plot"""
    df_clean = clean_intensity_data(df, quant_cols)
    
    data_list = []
    for col in quant_cols:
        condition = annotations[col]['condition']
        trimmed_name = annotations[col]['trimmed_name']
        values = df_clean[col].dropna()
        log_values = np.log10(values[values > 0])
        
        for val in log_values:
            data_list.append({
                'Sample': trimmed_name,
                'Condition': condition,
                'Log10_Intensity': val
            })
    
    plot_df = pd.DataFrame(data_list)
    colors = get_condition_colors()
    
    fig = px.violin(plot_df, x='Sample', y='Log10_Intensity', color='Condition',
                   color_discrete_map=colors, box=True, points=False)
    fig.update_layout(title="Intensity Distribution (Log10)", height=500, template="plotly_white")
    return fig

@st.cache_data
def plot_cv_analysis(df, quant_cols, annotations):
    """Coefficient of Variation"""
    df_clean = clean_intensity_data(df, quant_cols)
    
    control_cols = [col for col in quant_cols if annotations[col]['condition'] == 'Control']
    treatment_cols = [col for col in quant_cols if annotations[col]['condition'] == 'Treatment']
    
    cv_data = []
    for idx, row in df_clean.iterrows():
        control_vals = row[control_cols].dropna()
        if len(control_vals) > 1:
            cv = (control_vals.std() / control_vals.mean()) * 100
            if cv < 500:
                cv_data.append({'Condition': 'Control', 'CV': cv})
        
        treatment_vals = row[treatment_cols].dropna()
        if len(treatment_vals) > 1:
            cv = (treatment_vals.std() / treatment_vals.mean()) * 100
            if cv < 500:
                cv_data.append({'Condition': 'Treatment', 'CV': cv})
    
    cv_df = pd.DataFrame(cv_data)
    colors = get_condition_colors()
    
    fig = px.violin(cv_df, x='Condition', y='CV', color='Condition',
                   color_discrete_map=colors, box=True, points=False)
    fig.update_layout(title="Coefficient of Variation by Condition", height=500,
                     template="plotly_white", yaxis_title="CV (%)")
    return fig

@st.cache_data
def plot_pca(df, quant_cols, annotations):
    """PCA clustering"""
    df_clean = clean_intensity_data(df, quant_cols)
    df_pca = df_clean.dropna(thresh=len(quant_cols) * 0.5)
    df_pca = df_pca.fillna(df_pca.mean())
    
    data_transposed = df_pca.T
    pca = PCA(n_components=2)
    components = pca.fit_transform(data_transposed)
    
    plot_data = []
    for idx, col in enumerate(quant_cols):
        plot_data.append({
            'PC1': components[idx, 0],
            'PC2': components[idx, 1],
            'Sample': annotations[col]['trimmed_name'],
            'Condition': annotations[col]['condition']
        })
    
    plot_df = pd.DataFrame(plot_data)
    colors = get_condition_colors()
    
    fig = px.scatter(plot_df, x='PC1', y='PC2', color='Condition', text='Sample',
                    color_discrete_map=colors, size_max=15)
    fig.update_traces(marker=dict(size=12), textposition='top center')
    fig.update_layout(
        title=f"PCA: Sample Clustering (PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%})",
        height=500, template="plotly_white")
    return fig

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div style="background: {PRIMARY_RED}; padding: 20px; margin: -1rem -1rem 2rem -1rem; color: white;">
    <h1 style="margin: 0; font-size: 28px;">DIA Proteomics Analysis Pipeline</h1>
    <p style="margin: 5px 0 0 0; font-size: 14px;">Multi-Level Data Import & Analysis</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MODULE SELECTION
# ============================================================
tab1, tab2 = st.tabs(["Module 1: Data Import", "Module 2: Quality Control"])

# ============================================================
# MODULE 1: DATA IMPORT (DUAL UPLOAD)
# ============================================================
with tab1:
    st.markdown("""
    <div class="module-header">
        <h2 style="margin:0; font-size:24px;">Data Import & Validation</h2>
        <p style="margin:5px 0 0 0; opacity:0.9;">Upload protein-level and/or peptide-level quantification data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # PROTEIN-LEVEL UPLOAD
    # ============================================================
    st.subheader("Step 1A: Upload Protein-Level Data")
    st.caption("Generic protein quantification matrix - any number of columns supported")
    
    protein_file = st.file_uploader(
        "Drop protein-level CSV or TSV file here",
        type=["csv", "tsv", "txt"],
        key="protein_upload",
        help="Supports any format: non-numerical columns = metadata, numerical columns = quantification"
    )
    
    if protein_file is not None:
        try:
            sep = "\t" if protein_file.name.endswith((".tsv", ".txt")) else ","
            df_protein = pd.read_csv(protein_file, sep=sep)
            
            st.session_state.protein_data = df_protein
            
            # Detect data level
            metadata_cols, quant_cols = detect_column_types(df_protein)
            data_level = detect_data_level(df_protein, metadata_cols)
            
            st.session_state.protein_metadata_cols = metadata_cols
            st.session_state.protein_quant_cols = quant_cols
            st.session_state.protein_data_level = data_level
            
            # Display data level badge
            if data_level == 'protein':
                st.markdown(f'<div class="data-level-badge badge-protein">Data Level: Protein ✓</div>', unsafe_allow_html=True)
            elif data_level == 'peptide':
                st.markdown(f'<div class="data-level-badge badge-peptide">Data Level: Peptide (uploaded in protein section)</div>', unsafe_allow_html=True)
            elif data_level == 'both':
                st.markdown(f'<div class="data-level-badge badge-protein">Data Level: Both Protein & Peptide</div>', unsafe_allow_html=True)
            else:
                st.info("Data level: Unknown - no specific protein/peptide identifiers detected")
            
            st.success(f"File loaded: {protein_file.name} • {len(df_protein):,} rows • {len(df_protein.columns)} columns")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{len(df_protein):,}")
            with col2:
                st.metric("Total Columns", len(df_protein.columns))
            with col3:
                st.metric("Metadata Columns", len(metadata_cols))
            with col4:
                st.metric("Quantitative Columns", len(quant_cols))
            
            st.divider()
            
            # Column Details
            with st.expander("View Column Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Metadata Columns:**")
                    for col in metadata_cols:
                        st.text(f"• {col}")
                with col2:
                    st.markdown("**Quantitative Columns:**")
                    for col in quant_cols:
                        st.text(f"• {col}")
            
            st.divider()
            
            # Condition Assignment
            st.subheader("Step 2: Assign Conditions to Quantitative Columns")
            
            if not st.session_state.protein_annotations and len(quant_cols) > 0:
                st.session_state.protein_annotations = auto_assign_conditions(quant_cols)
            
            st.info(f"Auto-assignment: {len(quant_cols)} quantitative columns • First {len(quant_cols)//2} = Control • Remaining = Treatment")
            
            # Assignment table
            assignment_data = []
            for col in quant_cols:
                ann = st.session_state.protein_annotations[col]
                assignment_data.append({
                    'Original Name': col,
                    'Trimmed Name': ann['trimmed_name'],
                    'Condition': ann['condition']
                })
            
            assignment_df = pd.DataFrame(assignment_data)
            st.dataframe(assignment_df, use_container_width=True, hide_index=True)
            
            # Modify assignment
            st.markdown("**Modify Condition Assignment:**")
            condition_mode = st.selectbox(
                "Select assignment mode",
                options=["Auto (First half Control, Second half Treatment)", 
                        "All Control", 
                        "All Treatment",
                        "Custom"],
                index=0,
                key="protein_condition_mode"
            )
            
            if condition_mode == "All Control":
                for col in quant_cols:
                    st.session_state.protein_annotations[col]['condition'] = 'Control'
            elif condition_mode == "All Treatment":
                for col in quant_cols:
                    st.session_state.protein_annotations[col]['condition'] = 'Treatment'
            elif condition_mode == "Custom":
                st.warning("Custom mode: Set conditions individually below")
                for col in quant_cols:
                    ann = st.session_state.protein_annotations[col]
                    is_treatment = st.checkbox(
                        f"{ann['trimmed_name']} → Treatment",
                        value=ann['condition'] == 'Treatment',
                        key=f"custom_protein_{col}"
                    )
                    st.session_state.protein_annotations[col]['condition'] = 'Treatment' if is_treatment else 'Control'
            
            # Summary metrics
            n_control = sum(1 for ann in st.session_state.protein_annotations.values() if ann['condition'] == 'Control')
            n_treatment = sum(1 for ann in st.session_state.protein_annotations.values() if ann['condition'] == 'Treatment')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Quant Columns", len(quant_cols))
            with col2:
                st.metric("Control Samples", n_control)
            with col3:
                st.metric("Treatment Samples", n_treatment)
            
        except Exception as e:
            st.error(f"Error loading protein file: {str(e)}")
    
    st.divider()
    
    # ============================================================
    # PEPTIDE-LEVEL UPLOAD (PLACEHOLDER)
    # ============================================================
    st.subheader("Step 1B: Upload Peptide-Level Data (Optional)")
    st.caption("Coming soon - peptide-level quantification upload")
    
    peptide_file = st.file_uploader(
        "Drop peptide-level CSV or TSV file here",
        type=["csv", "tsv", "txt"],
        key="peptide_upload",
        help="Upload peptide-level data for peptide-specific QC",
        disabled=True
    )
    
    st.info("Peptide-level upload will be enabled in the next iteration")

# ============================================================
# MODULE 2: QUALITY CONTROL
# ============================================================
with tab2:
    st.markdown("""
    <div class="module-header">
        <h2 style="margin:0; font-size:24px;">Quality Control Visualization</h2>
        <p style="margin:5px 0 0 0; opacity:0.9;">Protein-level data quality assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.protein_data is not None and st.session_state.protein_annotations:
        df = st.session_state.protein_data
        quant_cols = st.session_state.protein_quant_cols
        annotations = st.session_state.protein_annotations
        
        st.info("Data processing: Values 0, 1, and NaN are treated as missing")
        st.divider()
        
        st.subheader("1. Intensity Distribution")
        fig1 = plot_intensity_distribution(df, quant_cols, annotations)
        st.plotly_chart(fig1, use_container_width=True)
        st.divider()
        
        st.subheader("2. Coefficient of Variation (CV)")
        fig2 = plot_cv_analysis(df, quant_cols, annotations)
        st.plotly_chart(fig2, use_container_width=True)
        st.divider()
        
        st.subheader("3. PCA: Sample Clustering")
        fig3 = plot_pca(df, quant_cols, annotations)
        st.plotly_chart(fig3, use_container_width=True)
        
    else:
        st.warning("No protein data uploaded. Please upload data in Module 1 first.")
import streamlit as st
import pandas as pd
from config import (
    get_numeric_columns, get_metadata_columns,
    get_default_species_mapping_cols,
    get_default_group_col, get_default_peptide_id_col
)

st.set_page_config('Proteomics Multi-Uploader', layout='wide')

colA, colB = st.columns(2)

with colA:
    st.header("Protein-level Upload")
    protein_file = st.file_uploader("Upload protein-level file", key="upl_protein", type=["csv", "tsv", "txt"])
    if protein_file:
        sep = "\t" if protein_file.name.endswith(('.tsv', '.txt')) else ','
        df_prot = pd.read_csv(protein_file, sep=sep)
        st.write("File loaded:", protein_file.name)
        num_cols = get_numeric_columns(df_prot)
        meta_cols = get_metadata_columns(df_prot, num_cols)

        st.markdown("**Select quantitative columns to keep:**")
        quant_cols_sel = st.multiselect("Quant columns", num_cols, default=num_cols, key="quant_cols_prot")
        st.markdown("**Select species mapping column:**")
        mapping_sel = st.selectbox("Species column", get_default_species_mapping_cols(df_prot) or meta_cols, key="specmap_prot")
        st.markdown("**Select protein group column:**")
        group_col_sel = st.selectbox("Protein group", [get_default_group_col(df_prot)] + meta_cols, key="groupcol_prot")
        st.write("")

        st.markdown("**Assign Control/Treatment**")
        auto_split = st.radio("Assignment mode", ["Auto-split", "All Control", "All Treatment", "Manual"], horizontal=True, key="mode_prot")
        if auto_split == "Auto-split":
            annot = ['Control' if i < len(quant_cols_sel)//2 else 'Treatment' for i in range(len(quant_cols_sel))]
        elif auto_split == "All Control":
            annot = ['Control' for _ in quant_cols_sel]
        elif auto_split == "All Treatment":
            annot = ['Treatment' for _ in quant_cols_sel]
        else:
            annot = []
            for q in quant_cols_sel:
                annot.append(st.selectbox(f"{q}", options=["Control", "Treatment"], key=f"man_assign_prot_{q}"))

        st.session_state["protein_upload"] = {
            "data": df_prot,
            "quant_cols": quant_cols_sel,
            "meta_cols": meta_cols,
            "species_col": mapping_sel,
            "group_col": group_col_sel,
            "condition": dict(zip(quant_cols_sel, annot))
        }
        st.success("Protein-level data loaded and annotated.")

with colB:
    st.header("Peptide-level Upload")
    peptide_file = st.file_uploader("Upload peptide-level file", key="upl_peptide", type=["csv", "tsv", "txt"])
    if peptide_file:
        sep = "\t" if peptide_file.name.endswith(('.tsv', '.txt')) else ','
        df_pept = pd.read_csv(peptide_file, sep=sep)
        st.write("File loaded:", peptide_file.name)
        num_cols = get_numeric_columns(df_pept)
        meta_cols = get_metadata_columns(df_pept, num_cols)

        st.markdown("**Select quantitative columns to keep:**")
        quant_cols_sel = st.multiselect("Quant columns", num_cols, default=num_cols, key="quant_cols_pept")
        st.markdown("**Select species mapping column:**")
        mapping_sel = st.selectbox("Species column", get_default_species_mapping_cols(df_pept) or meta_cols, key="specmap_pept")
        st.markdown("**Select protein group column:**")
        group_col_sel = st.selectbox("Protein group", [get_default_group_col(df_pept)] + meta_cols, key="groupcol_pept")
        st.markdown("**Select peptide identifier column:**")
        pept_id_sel = st.selectbox("Peptide/Precursor ID", [get_default_peptide_id_col(df_pept)] + meta_cols, key="peptidcol_pept")
        st.write("")

        st.markdown("**Assign Control/Treatment**")
        auto_split = st.radio("Assignment mode", ["Auto-split", "All Control", "All Treatment", "Manual"], horizontal=True, key="mode_pept")
        if auto_split == "Auto-split":
            annot = ['Control' if i < len(quant_cols_sel)//2 else 'Treatment' for i in range(len(quant_cols_sel))]
        elif auto_split == "All Control":
            annot = ['Control' for _ in quant_cols_sel]
        elif auto_split == "All Treatment":
            annot = ['Treatment' for _ in quant_cols_sel]
        else:
            annot = []
            for q in quant_cols_sel:
                annot.append(st.selectbox(f"{q}", options=["Control", "Treatment"], key=f"man_assign_pept_{q}"))

        st.session_state["peptide_upload"] = {
            "data": df_pept,
            "quant_cols": quant_cols_sel,
            "meta_cols": meta_cols,
            "species_col": mapping_sel,
            "group_col": group_col_sel,
            "peptide_id_col": pept_id_sel,
            "condition": dict(zip(quant_cols_sel, annot))
        }
        st.success("Peptide-level data loaded and annotated.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(f"""
<div style="text-align:center; padding:20px; color:{PRIMARY_GRAY}; font-size:12px;">
    <strong>Proprietary & Confidential | For Internal Use Only</strong><br>
    © 2024 Thermo Fisher Scientific Inc. All rights reserved.
</div>
""", unsafe_allow_html=True)
