"""
app.py - ThermoFisher Scientific Proteomics Analysis Platform
=============================================================

Main Streamlit application entry point with multi-page navigation.
Uses ThermoFisher Scientific brand theme throughout.

Features:
- Professional brand styling with theme system
- Multi-page navigation (7 pages)
- Unified session state management
- Data persistence and caching
- Dark/Light mode support

Pages:
1. Data Upload - Upload and configure proteomics data
2. Visual EDA - Exploratory data analysis
3. Data Filtering - Filter and QC analysis
4. Missing Value Imputation - Handle missing data
5. Post-Imputation EDA - Post-imputation analysis
6. Differential Abundance - Statistical comparisons
(Additional pages can be added as needed)

Author: Your Team
Date: December 2025
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import theme utilities
from theme import apply_theme_css, get_theme_colors, PRIMARY_RED, theme_toggle

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="ThermoFisher Proteomics Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply ThermoFisher brand theme
apply_theme_css()
colors = get_theme_colors()

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown("---")
    
    # Logo/Title
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0;">
        <h2 style="color: {PRIMARY_RED}; margin: 0;">🧬 ThermoFisher</h2>
        <p style="color: #54585A; font-size: 12px; margin: 8px 0;">
            Proteomics Analysis Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation Menu
    st.subheader("📖 Navigation", divider="red")
    
    page = st.radio(
        "Select Page:",
        options=[
            "🏠 Home",
            "📤 Data Upload",
            "📊 Visual EDA",
            "🔍 Data Filtering",
            "🔧 Missing Value Imputation",
            "📈 Post-Imputation EDA",
            "📉 Differential Abundance",
        ],
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    
    # Status Section
    st.subheader("⚙️ Status", divider="red")
    
    if st.session_state.get("data_ready", False):
        st.success("✅ Data Loaded")
        
        # Show data summary if available
        if "df_raw" in st.session_state:
            df = st.session_state.df_raw
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
        
        # Reset Data Button
        if st.button("🔄 Load New Data", width="stretch"):
            st.session_state.data_ready = False
            st.session_state.clear()
            st.rerun()
    else:
        st.info("⏳ No data loaded yet")
        st.caption("Upload data from the Data Upload page to begin.")
    
    st.markdown("---")
    
    # Theme Toggle
    st.subheader("🎨 Appearance", divider="red")
    theme_toggle()
    
    st.markdown("---")
    
    # Footer
    st.caption(
        "🔬 Built with Streamlit & ThermoFisher Scientific Branding\n"
        "Version 1.0 | December 2025"
    )

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Home Page
if page == "🏠 Home":
    st.markdown(f"""
    <div style="padding: 20px 0; border-bottom: 3px solid {PRIMARY_RED}; margin-bottom: 30px;">
        <h1 style="color: {PRIMARY_RED}; margin: 0;">🧬 ThermoFisher Proteomics Analysis</h1>
        <p style="color: #54585A; margin: 8px 0 0 0;">Professional proteomics data analysis platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="
            background-color: rgba(231, 19, 22, 0.05);
            border-left: 4px solid {PRIMARY_RED};
            padding: 20px;
            border-radius: 8px;
        ">
            <h3 style="color: {PRIMARY_RED}; margin-top: 0;">📤 Upload</h3>
            <p>Import your proteomics data in CSV or Excel format with automatic detection and preprocessing.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background-color: rgba(181, 189, 0, 0.05);
            border-left: 4px solid #B5BD00;
            padding: 20px;
            border-radius: 8px;
        ">
            <h3 style="color: #B5BD00; margin-top: 0;">🔍 Analyze</h3>
            <p>Explore data quality, patterns, and distributions with interactive visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="
            background-color: rgba(234, 118, 0, 0.05);
            border-left: 4px solid #EA7600;
            padding: 20px;
            border-radius: 8px;
        ">
            <h3 style="color: #EA7600; margin-top: 0;">📊 Visualize</h3>
            <p>Generate publication-ready charts and statistical comparisons with ThermoFisher branding.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Start Guide
    st.header("🚀 Quick Start Guide", divider="red")
    
    with st.expander("📋 Step 1: Prepare Your Data", expanded=True):
        st.markdown("""
        **Expected Data Format:**
        - CSV or Excel file (.csv, .xlsx, .xls)
        - Rows: Proteins or peptides
        - Columns: Metadata (ID, Gene, Species, etc.) + Abundance samples
        
        **Supported Formats:**
        - **WIDE**: Samples as columns, proteins as rows (most common)
        - **LONG**: Proteins in rows, abundance in specific columns
        
        **Required Columns:**
        - At least one ID column (protein/peptide identifiers)
        - At least one numeric column (abundance data)
        - Optional: Species, Gene, or other metadata columns
        """)
    
    with st.expander("📤 Step 2: Upload Data"):
        st.markdown("""
        1. Go to **Data Upload** page (📤 in navigation)
        2. Click "Choose CSV or Excel file"
        3. Select your data file
        4. Configure:
           - Select which columns to keep
           - Rename columns if desired
           - Identify species
           - Set experimental conditions
        5. Click "Process & Save Data"
        """)
    
    with st.expander("📊 Step 3: Explore & Analyze"):
        st.markdown("""
        After uploading, use these pages in order:
        
        1. **Visual EDA** - Overview of data distribution
        2. **Data Filtering** - Remove low-quality features
        3. **Missing Value Imputation** - Handle missing data
        4. **Post-Imputation EDA** - Verify imputations
        5. **Differential Abundance** - Statistical comparisons
        
        Each page builds on previous results.
        """)
    
    st.markdown("---")
    
    # Features Overview
    st.header("✨ Features", divider="red")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Management**
        - ✅ Multi-format support (CSV, Excel)
        - ✅ Automatic format detection (WIDE/LONG)
        - ✅ Column selection and renaming
        - ✅ Intelligent metadata detection
        - ✅ Species and condition mapping
        
        **Analysis**
        - ✅ Quality control metrics
        - ✅ Missing value analysis
        - ✅ Differential abundance testing
        - ✅ Statistical visualizations
        """)
    
    with col2:
        st.markdown("""
        **Visualization**
        - ✅ Interactive charts (Plotly)
        - ✅ Publication-ready styling
        - ✅ ThermoFisher brand colors
        - ✅ Dark/Light mode support
        
        **Performance**
        - ✅ Vectorized operations
        - ✅ Smart caching
        - ✅ Disk persistence
        - ✅ Real-time updates
        """)
    
    st.markdown("---")
    
    # Getting Started Button
    st.header("Ready to Start?", divider="red")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button(
            "📤 Go to Data Upload",
            type="primary",
            width="stretch",
            key="home_upload_btn"
        ):
            st.switch_page("pages/1_Data_Upload.py")

# Data Upload Page
elif page == "📤 Data Upload":
    from pages import page_1_data_upload
    page_1_data_upload.render()

# Visual EDA Page
elif page == "📊 Visual EDA":
    if not st.session_state.get("data_ready", False):
        st.error("❌ No data loaded. Please upload data first.")
        st.info("👉 Go to **Data Upload** page to load your data.")
    else:
        try:
            from pages import page_2_eda
            page_2_eda.render()
        except FileNotFoundError:
            st.warning("⚠️ Visual EDA page not yet implemented.")
            st.info("This page will be available in a future update.")

# Data Filtering Page
elif page == "🔍 Data Filtering":
    if not st.session_state.get("data_ready", False):
        st.error("❌ No data loaded. Please upload data first.")
        st.info("👉 Go to **Data Upload** page to load your data.")
    else:
        try:
            from pages import page_3_filtering
            page_3_filtering.render()
        except FileNotFoundError:
            st.warning("⚠️ Data Filtering page not yet implemented.")
            st.info("This page will be available in a future update.")

# Missing Value Imputation Page
elif page == "🔧 Missing Value Imputation":
    if not st.session_state.get("data_ready", False):
        st.error("❌ No data loaded. Please upload data first.")
        st.info("👉 Go to **Data Upload** page to load your data.")
    else:
        try:
            from pages import page_4_imputation
            page_4_imputation.render()
        except FileNotFoundError:
            st.warning("⚠️ Missing Value Imputation page not yet implemented.")
            st.info("This page will be available in a future update.")

# Post-Imputation EDA Page
elif page == "📈 Post-Imputation EDA":
    if not st.session_state.get("data_ready", False):
        st.error("❌ No data loaded. Please upload data first.")
        st.info("👉 Go to **Data Upload** page to load your data.")
    else:
        try:
            from pages import page_5_post_imputation_eda
            page_5_post_imputation_eda.render()
        except FileNotFoundError:
            st.warning("⚠️ Post-Imputation EDA page not yet implemented.")
            st.info("This page will be available in a future update.")

# Differential Abundance Page
elif page == "📉 Differential Abundance":
    if not st.session_state.get("data_ready", False):
        st.error("❌ No data loaded. Please upload data first.")
        st.info("👉 Go to **Data Upload** page to load your data.")
    else:
        try:
            from pages import page_6_differential_abundance
            page_6_differential_abundance.render()
        except FileNotFoundError:
            st.warning("⚠️ Differential Abundance page not yet implemented.")
            st.info("This page will be available in a future update.")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if __name__ == "__main__":
    # Initialize session state variables
    if "data_ready" not in st.session_state:
        st.session_state.data_ready = False
    
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "light"

