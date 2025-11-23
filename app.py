import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

# Import from config
from config import (
    detect_column_types,
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
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'column_annotations' not in st.session_state:
    st.session_state.column_annotations = {}
if 'metadata_cols' not in st.session_state:
    st.session_state.metadata_cols = []
if 'quant_cols' not in st.session_state:
    st.session_state.quant_cols = []

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
# PLOTTING FUNCTIONS
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
    
    fig = px.violin(
        plot_df,
        x='Sample',
        y='Log10_Intensity',
        color='Condition',
        color_discrete_map=colors,
        box=True,
        points=False
    )
    
    fig.update_layout(
        title="Intensity Distribution (Log10)",
        height=500,
        template="plotly_white"
    )
    
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
    
    fig = px.violin(
        cv_df,
        x='Condition',
        y='CV',
        color='Condition',
        color_discrete_map=colors,
        box=True,
        points=False
    )
    
    fig.update_layout(
        title="Coefficient of Variation by Condition",
        height=500,
        template="plotly_white",
        yaxis_title="CV (%)"
    )
    
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
    
    fig = px.scatter(
        plot_df,
        x='PC1',
        y='PC2',
        color='Condition',
        text='Sample',
        color_discrete_map=colors,
        size_max=15
    )
    
    fig.update_traces(marker=dict(size=12), textposition='top center')
    
    fig.update_layout(
        title=f"PCA: Sample Clustering (PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%})",
        height=500,
        template="plotly_white"
    )
    
    return fig

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div style="background: {PRIMARY_RED}; padding: 20px; margin: -1rem -1rem 2rem -1rem; color: white;">
    <h1 style="margin: 0; font-size: 28px;">DIA Proteomics Analysis Pipeline</h1>
    <p style="margin: 5px 0 0 0; font-size: 14px;">Multi-Module Data Analysis Platform</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MODULE SELECTION
# ============================================================
tab1, tab2 = st.tabs(["Module 1: Data Import", "Module 2: Quality Control"])

# ============================================================
# MODULE 1: DATA IMPORT
# ============================================================
with tab1:
    st.markdown("""
    <div class="module-header">
        <h2 style="margin:0; font-size:24px;">Data Import & Validation</h2>
        <p style="margin:5px 0 0 0; opacity:0.9;">Import mass spectrometry data with automatic format detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Step 1: Upload Data File")
    
    uploaded_file = st.file_uploader(
        "Drop CSV or TSV file here",
        type=["csv", "tsv", "txt"],
        help="Supports CSV/TSV files from Spectronaut, DIA-NN, MaxQuant, FragPipe"
    )
    
    if uploaded_file is not None:
        try:
            sep = "\t" if uploaded_file.name.endswith((".tsv", ".txt")) else ","
            df = pd.read_csv(uploaded_file, sep=sep)
            
            st.session_state.uploaded_data = df
            
            st.success(f"File loaded: {uploaded_file.name} • {len(df):,} rows • {len(df.columns)} columns")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("File Size", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            st.divider()
            
            # Column Detection
            st.subheader("Step 2: Column Detection")
            
            metadata_cols, quant_cols = detect_column_types(df)
            st.session_state.metadata_cols = metadata_cols
            st.session_state.quant_cols = quant_cols
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Metadata Columns", len(metadata_cols))
                with st.expander("View metadata columns"):
                    for col in metadata_cols:
                        st.text(f"• {col}")
            
            with col2:
                st.metric("Quantitative Columns", len(quant_cols))
                with st.expander("View quantitative columns"):
                    for col in quant_cols:
                        st.text(f"• {col}")
            
            st.divider()
            
            # Condition Assignment
            st.subheader("Step 3: Assign Conditions")
            
            # Auto-assign on first load
            if not st.session_state.column_annotations:
                st.session_state.column_annotations = auto_assign_conditions(quant_cols)
            
            st.info(f"Auto-assignment: {len(quant_cols)} columns detected • First {len(quant_cols)//2} = Control • Remaining = Treatment")
            
            # Show assignment table
            st.markdown("### Column Assignment")
            
            assignment_data = []
            for col in quant_cols:
                ann = st.session_state.column_annotations[col]
                assignment_data.append({
                    'Original Name': col,
                    'Trimmed Name': ann['trimmed_name'],
                    'Condition': ann['condition']
                })
            
            assignment_df = pd.DataFrame(assignment_data)
            st.dataframe(assignment_df, use_container_width=True, hide_index=True)
            
            # Dropdown to change condition assignment
            st.markdown("### Modify Condition Assignment")
            
            condition_mode = st.selectbox(
                "Select assignment mode",
                options=["Auto (First half Control, Second half Treatment)", 
                        "All Control", 
                        "All Treatment",
                        "Custom"],
                index=0
            )
            
            if condition_mode == "All Control":
                for col in quant_cols:
                    st.session_state.column_annotations[col]['condition'] = 'Control'
            elif condition_mode == "All Treatment":
                for col in quant_cols:
                    st.session_state.column_annotations[col]['condition'] = 'Treatment'
            elif condition_mode == "Custom":
                st.warning("Custom mode: Use checkboxes below to set conditions individually")
                
                for col in quant_cols:
                    ann = st.session_state.column_annotations[col]
                    is_treatment = st.checkbox(
                        f"{ann['trimmed_name']} (Treatment)",
                        value=ann['condition'] == 'Treatment',
                        key=f"custom_{col}"
                    )
                    st.session_state.column_annotations[col]['condition'] = 'Treatment' if is_treatment else 'Control'
            
            # Summary
            n_control = sum(1 for ann in st.session_state.column_annotations.values() if ann['condition'] == 'Control')
            n_treatment = sum(1 for ann in st.session_state.column_annotations.values() if ann['condition'] == 'Treatment')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Columns", len(quant_cols))
            with col2:
                st.metric("Control Samples", n_control)
            with col3:
                st.metric("Treatment Samples", n_treatment)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================================
# MODULE 2: QUALITY CONTROL
# ============================================================
with tab2:
    st.markdown("""
    <div class="module-header">
        <h2 style="margin:0; font-size:24px;">Quality Control Visualization</h2>
        <p style="margin:5px 0 0 0; opacity:0.9;">Comprehensive data quality assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is not None and st.session_state.column_annotations:
        df = st.session_state.uploaded_data
        quant_cols = st.session_state.quant_cols
        annotations = st.session_state.column_annotations
        
        st.info("Data processing: Values 0, 1, and NaN are treated as missing")
        
        st.divider()
        
        # Plot 1
        st.subheader("1. Intensity Distribution")
        fig1 = plot_intensity_distribution(df, quant_cols, annotations)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.divider()
        
        # Plot 2
        st.subheader("2. Coefficient of Variation (CV)")
        fig2 = plot_cv_analysis(df, quant_cols, annotations)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.divider()
        
        # Plot 3
        st.subheader("3. PCA: Sample Clustering")
        fig3 = plot_pca(df, quant_cols, annotations)
        st.plotly_chart(fig3, use_container_width=True)
        
    else:
        st.warning("No data uploaded. Please upload data in Module 1 first.")

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
