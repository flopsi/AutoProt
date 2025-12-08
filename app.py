"""
welcome.py - MAIN ENTRY POINT
AutoProt Streamlit Application - Proteomics Data Analysis Platform
Handles session initialization, navigation, and global configuration
"""

import streamlit as st
from datetime import datetime
import uuid

# ============================================================================
# PAGE CONFIGURATION (MUST BE FIRST)
# Set up Streamlit page settings before any other Streamlit commands
# ============================================================================

st.set_page_config(
    page_title="AutoProt - Proteomics Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# AutoProt\nAutomated proteomics data analysis platform"
    }
)

# ============================================================================
# SESSION STATE INITIALIZATION
# Initialize session state variables with defaults
# ============================================================================

def init_session_state(key: str, default_value):
    """Initialize session state variable if not already set."""
    if key not in st.session_state:
        st.session_state[key] = default_value


# Core session variables
init_session_state("session_id", str(uuid.uuid4())[:8])
init_session_state("session_start", datetime.now().isoformat())
init_session_state("theme", "light")
init_session_state("data_ready", False)
init_session_state("data_type", "protein")

# Data variables
init_session_state("protein_data", None)
init_session_state("peptide_data", None)
init_session_state("df_raw", None)
init_session_state("numeric_cols", [])
init_session_state("id_col", None)
init_session_state("species_col", None)
init_session_state("sequence_col", None)

# Analysis cache
init_session_state("transform_cache", {})
init_session_state("current_transform", "log2")
init_session_state("analysis_results", None)
init_session_state("statistical_results", None)

# ============================================================================
# SIDEBAR - GLOBAL SETTINGS & NAVIGATION
# Theme selector and app-wide configuration
# ============================================================================

with st.sidebar:
    st.title("🧬 AutoProt")
    st.caption("Proteomics Analysis Platform v1.0")
    
    st.markdown("---")
    
    # Theme selector
    st.subheader("⚙️ Settings")
    theme_options = {
        "light": "☀️ Light",
        "dark": "🌙 Dark",
    }
    
    selected_theme = st.selectbox(
        "Color Theme",
        options=list(theme_options.keys()),
        format_func=lambda x: theme_options.get(x, x.title()),
        key="theme_selector",
        help="Select color scheme for plots and visualizations"
    )
    
    st.session_state.theme = selected_theme
    
    st.markdown("---")
    
    # Workflow guide
    st.subheader("📋 Workflow")
    st.markdown("""
    1. **Upload Data** - Load proteomics data
    2. **Visual EDA** - Explore & transform
    3. **Statistical EDA** - Compare groups
    """)
    
    st.markdown("---")
    
    # Data status
    st.subheader("📊 Current Session")
    
    if st.session_state.get("data_ready") and st.session_state.get("protein_data"):
        st.success("✅ Data loaded")
        
        protein_data = st.session_state.protein_data
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Proteins", f"{protein_data.n_proteins:,}")
            st.metric("Samples", protein_data.n_samples)
        
        with col2:
            st.metric("Missing %", f"{protein_data.missing_rate:.1f}%")
            st.metric("Type", "Protein")
        
        if st.button("🔄 Reset Data", use_container_width=True, key="btn_reset"):
            st.session_state.data_ready = False
            st.session_state.protein_data = None
            st.session_state.peptide_data = None
            st.rerun()
    
    elif st.session_state.get("data_ready") and st.session_state.get("peptide_data"):
        st.success("✅ Data loaded")
        
        peptide_data = st.session_state.peptide_data
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Peptides", f"{peptide_data.n_peptides:,}")
            st.metric("Samples", peptide_data.n_samples)
        
        with col2:
            st.metric("Missing %", f"{peptide_data.missing_rate:.1f}%")
            st.metric("Type", "Peptide")
        
        if st.button("🔄 Reset Data", use_container_width=True, key="btn_reset_pep"):
            st.session_state.data_ready = False
            st.session_state.protein_data = None
            st.session_state.peptide_data = None
            st.rerun()
    
    else:
        st.info("📥 No data loaded")
        st.caption("Start by uploading data on the **Data Upload** page")
    
    st.markdown("---")
    
    # Session info (collapsed)
    with st.expander("ℹ️ Session Info", expanded=False):
        st.caption(f"**Session ID:** `{st.session_state.session_id}`")
        st.caption(f"**Started:** {st.session_state.session_start[:10]}")
        st.caption(f"**Theme:** {selected_theme.title()}")

# ============================================================================
# MAIN CONTENT - LANDING PAGE
# ============================================================================

st.title("🧬 AutoProt")
st.markdown("**Automated Proteomics Data Analysis Platform**")

st.markdown("""
Welcome to **AutoProt**, a comprehensive proteomics data analysis tool built with Streamlit.
This platform automates common proteomics workflows including data transformation, 
exploratory data analysis, statistical testing, and quality control.

**Designed for:**
- Quantitative proteomics (LFQ, TMT, SILAC, DIA)
- Peptide-level and protein-level analysis
- Multi-condition experimental designs
- Batch effect detection
""")

st.markdown("---")

# ============================================================================
# FEATURE OVERVIEW
# ============================================================================

st.header("✨ Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📁 Data Upload
    
    - **Multi-format:** CSV, TSV, Excel
    - **Auto-detection:** Numeric columns, IDs, species
    - **Validation:** Quality checks, missing data analysis
    - **Flexibility:** Protein or peptide data
    """)

with col2:
    st.markdown("""
    ### 📊 Visual EDA
    
    - **10+ transformations:** log2, log10, Yeo-Johnson, quantile norm
    - **Normality tests:** Shapiro-Wilk, Anderson-Darling
    - **Visualizations:** Q-Q plots, distributions, heatmaps
    - **Interactive:** Plotly-based plots with hover details
    """)

with col3:
    st.markdown("""
    ### 🧪 Statistical Analysis
    
    - **Testing:** Welch's t-test, ANOVA, Mann-Whitney U
    - **Correction:** Benjamini-Hochberg FDR
    - **Visualizations:** Volcano plots, MA plots, heatmaps
    - **Export:** CSV, Excel with full annotations
    """)

st.markdown("---")

# ============================================================================
# QUICK START GUIDE
# ============================================================================

st.header("🚀 Quick Start")

tab1, tab2, tab3 = st.tabs(["📁 Step 1: Upload", "📊 Step 2: Explore", "🧪 Step 3: Analyze"])

with tab1:
    st.subheader("Upload Your Proteomics Data")
    
    st.markdown("""
    Navigate to **📁 Data Upload** page in the sidebar.
    
    **Requirements:**
    - First column: Protein/Gene/Peptide IDs (e.g., `P12345`, `ENSEMBL_ID`)
    - Numeric columns: Sample abundance values (intensities, counts)
    - Column naming: `A1`, `A2`, `B1`, `B2` (condition_replicate)
    - Optional: Species annotation column
    
    **Supported formats:**
    - CSV / TSV (comma or tab-separated)
    - Excel (.xlsx, .xls)
    - Max file size: 100 MB
    
    **Example format:**
    ```
    Protein_ID    Species    A1      A2      B1      B2
    P12345        Human      1000    950     200     180
    P67890        Human      5000    4800    3000    3100
    ```
    """)

with tab2:
    st.subheader("Transform & Explore Data")
    
    st.markdown("""
    After uploading, go to **📊 Visual EDA - Proteins** page.
    
    **Transformations available:**
    - **log2** - Standard in proteomics (preferred for fold-change)
    - **log10** - Alternative log-scale
    - **Yeo-Johnson** - Handles zeros, skewed distributions
    - **Box-Cox** - Optimal for normality
    - **Quantile** - Distribution normalization
    - **Robust scaling** - Outlier-resistant
    - **Standard scaling** - Z-score normalization
    - **MinMax scaling** - [0,1] range
    - **None** - Raw intensities
    
    **Analysis tools:**
    - Distribution plots (box, violin, histogram)
    - Q-Q plots for normality assessment
    - Heatmaps and correlation analysis
    - PCA and dimensionality reduction
    - Missing data patterns
    """)

with tab3:
    st.subheader("Statistical Testing & Export")
    
    st.markdown("""
    In **🧪 Statistical EDA** page:
    
    1. **Define conditions** - Group samples by experimental condition
    2. **Select statistical test:**
       - Welch's t-test (2 groups, unequal variance)
       - ANOVA (3+ groups)
       - Mann-Whitney U (non-parametric)
    3. **Set thresholds:**
       - log2FC cutoff (default: ±1.0)
       - Adjusted p-value (default: 0.05)
    4. **Generate results:**
       - Volcano plots with regulation status
       - Summary statistics table
       - Export to CSV/Excel
    
    **Output includes:**
    - Protein ID, log2FC, p-value, adj. p-value
    - Regulation status (Up/Down/Unchanged)
    - Mean abundance by condition
    """)

st.markdown("---")

# ============================================================================
# WORKFLOW DIAGRAM
# ============================================================================

st.header("📈 Analysis Workflow")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    ### 1️⃣ Upload
    
    📥 Load data file
    
    ✓ Validate format
    """)

with col2:
    st.markdown("""
    ### 2️⃣ Explore
    
    🔍 Visual inspection
    
    ✓ Normality tests
    """)

with col3:
    st.markdown("""
    ### 3️⃣ Transform
    
    📊 Apply transformation
    
    ✓ Assess effectiveness
    """)

with col4:
    st.markdown("""
    ### 4️⃣ Analyze
    
    🧪 Statistical testing
    
    ✓ Export results
    """)

st.markdown("---")

# ============================================================================
# SYSTEM INFORMATION
# ============================================================================

st.header("ℹ️ System Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Current Session:**
    - Session ID: `{}`
    - Theme: **{}**
    """.format(
        st.session_state.session_id,
        st.session_state.theme.title()
    ))

with col2:
    st.markdown("""
    **Data Status:**
    - Data Loaded: **{}**
    - Data Type: **{}**
    """.format(
        "Yes ✅" if st.session_state.data_ready else "No ❌",
        st.session_state.data_type.title()
    ))

with col3:
    st.markdown("""
    **Performance:**
    - Caching: **Enabled**
    - Memory: **Optimized**
    - Plots: **Plotly (Interactive)**
    """)

st.markdown("---")

# ============================================================================
# FOOTER & HELP
# ============================================================================

st.header("📚 Documentation & Support")

col1, col2 = st.columns(2)

with col1:
    with st.expander("❓ FAQ", expanded=False):
        st.markdown("""
        **Q: What file formats are supported?**
        A: CSV, TSV, TXT, and Excel (.xlsx, .xls)
        
        **Q: Can I analyze both proteins and peptides?**
        A: Yes! Select data type at upload.
        
        **Q: How do I handle missing values?**
        A: AutoProt handles NaN values automatically in statistics.
        
        **Q: Can I combine multiple datasets?**
        A: Not in current version. Upload combined file instead.
        """)

with col2:
    with st.expander("🔗 Links & Resources", expanded=False):
        st.markdown("""
        - [Streamlit Documentation](https://docs.streamlit.io)
        - [Plotly Visualization](https://plotly.com)
        - [Scipy Statistics](https://docs.scipy.org/doc/scipy/)
        - [Pandas DataFrames](https://pandas.pydata.org/)
        """)

st.markdown("---")

st.caption("""
**AutoProt** | Proteomics Analysis Platform | 
Built with ❤️ using Streamlit | v1.0
""")

# ============================================================================
# KEYBOARD SHORTCUTS (Info only)
# ============================================================================

with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
    st.markdown("""
    - **`r`** - Rerun app
    - **`s`** - Open settings
    - **`c`** - Clear cache
    """)
