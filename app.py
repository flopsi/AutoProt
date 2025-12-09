"""
app.py - Main Entry Point

AutoProt Streamlit Application - Proteomics Data Analysis Platform
Handles session initialization, navigation, and global configuration
"""

import streamlit as st
from datetime import datetime
import uuid
import logging
from pathlib import Path

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Create logs directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "autoprot.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================================================
# PAGE CONFIGURATION (MUST BE FIRST)
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
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables at app startup."""
    defaults = {
        # Session metadata
        'session_id': str(uuid.uuid4())[:8],
        'session_start': datetime.now().isoformat(),
        
        # App settings
        'theme': 'light',
        'data_ready': False,
        'data_type': 'protein',
        
        # Data containers
        'protein_data': None,
        'peptide_data': None,
        'df_raw': None,
        'df_raw_polars': None,
        
        # Column information
        'numeric_cols': [],
        'id_col': None,
        'species_col': None,
        'sequence_col': None,
        'metadata_columns': [],
        'column_mapping': {},
        'reverse_mapping': {},
        
        # Analysis cache
        'transform_cache': {},
        'current_transform': 'log2',
        'analysis_results': None,
        'statistical_results': None,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    logger.info(f"Session {st.session_state.session_id} initialized")


# Initialize session on app start
initialize_session_state()


# ============================================================================
# SIDEBAR - GLOBAL SETTINGS & NAVIGATION
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
    
    if st.session_state.get("data_ready"):
        if st.session_state.get("protein_data"):
            st.success("✅ Data loaded")
            protein_data = st.session_state.protein_data
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Proteins", f"{protein_data.n_proteins:,}")
                st.metric("Samples", protein_data.n_samples)
            with col2:
                st.metric("Missing %", f"{protein_data.missing_rate:.1f}%")
                st.metric("Type", "Protein")
            
            if st.button("🔄 Reset Data", width="stretch", key="btn_reset"):
                st.session_state.data_ready = False
                st.session_state.protein_data = None
                st.session_state.peptide_data = None
                logger.info(f"Session {st.session_state.session_id}: Data reset")
                st.rerun()
        
        elif st.session_state.get("peptide_data"):
            st.success("✅ Data loaded")
            peptide_data = st.session_state.peptide_data
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Peptides", f"{peptide_data.n_peptides:,}")
                st.metric("Samples", peptide_data.n_samples)
            with col2:
                st.metric("Missing %", f"{peptide_data.missing_rate:.1f}%")
                st.metric("Type", "Peptide")
            
            if st.button("🔄 Reset Data", width="stretch", key="btn_reset_pep"):
                st.session_state.data_ready = False
                st.session_state.protein_data = None
                st.session_state.peptide_data = None
                logger.info(f"Session {st.session_state.session_id}: Data reset")
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
    
    - **5 transformations:** log2, yeo-johnson, arcsin, quantile, raw
    - **Normality tests:** Shapiro-Wilk, transformation comparison
    - **Visualizations:** Distributions, Q-Q plots, heatmaps
    - **Interactive:** Plotly-based plots with hover details
    """)

with col3:
    st.markdown("""
    ### 🧪 Statistical Analysis
    
    - **Testing:** Condition detection, filtering
    - **Correction:** Multiple filtering strategies
    - **Visualizations:** Volcano plots, MA plots
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
    Protein_ID Species A1 A2 B1 B2
    P12345 Human 1000 950 200 180
    P67890 Human 5000 4800 3000 3100
    ```
    """)

with tab2:
    st.subheader("Transform & Explore Data")
    
    st.markdown("""
    After uploading, go to **📊 Visual EDA - Proteins** page.
    
    **Transformations available:**
    
    - **log2** - Standard in proteomics (preferred for fold-change)
    - **yeo-johnson** - Handles zeros, skewed distributions
    - **arcsin** - Variance stabilization for rare proteins
    - **quantile** - Distribution normalization
    - **raw** - Original data
    
    **Analysis tools:**
    
    - Distribution plots (box, violin, histogram)
    - Q-Q plots for normality assessment
    - Heatmaps and correlation analysis
    - PCA visualization
    - Missing data patterns
    """)

with tab3:
    st.subheader("Filtering & Statistical Testing")
    
    st.markdown("""
    In **🧪 Statistical EDA** page:
    
    1. **Filter data:**
       - By missing data rate (remove sparse proteins)
       - By coefficient of variation (remove high-variance proteins)
       - By intensity threshold (minimum abundance)
       - By valid samples per condition
    
    2. **Compare conditions:**
       - Auto-detect from column names (A1, A2, B1, B2)
       - Compute per-sample statistics
       - Generate summary statistics
    
    3. **Export results:**
       - Download filtered data
       - Summary statistics tables
       - Comparison metrics
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
    ### 4️⃣ Filter & Analyze
    
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
    st.markdown(f"""
    **Current Session:**
    
    - Session ID: `{st.session_state.session_id}`
    - Theme: **{st.session_state.theme.title()}**
    """)

with col2:
    st.markdown(f"""
    **Data Status:**
    
    - Data Loaded: **{'Yes ✅' if st.session_state.data_ready else 'No ❌'}**
    - Data Type: **{st.session_state.data_type.title()}**
    """)

with col3:
    st.markdown("""
    **Performance:**
    
    - Caching: **Enabled**
    - Memory: **Optimized**
    - Plots: **Plotly (Interactive)**
    """)

st.markdown("---")


# ============================================================================
# DOCUMENTATION & HELP
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
        
        **Q: How can I filter my data?**
        
        A: Use multiple filtering strategies:
        - Missing data rate threshold
        - Coefficient of variation within conditions
        - Minimum intensity requirement
        - Minimum valid samples per condition
        """)

with col2:
    with st.expander("🔗 Links & Resources", expanded=False):
        st.markdown("""
        - [Streamlit Documentation](https://docs.streamlit.io)
        - [Plotly Visualization](https://plotly.com)
        - [Scipy Statistics](https://docs.scipy.org/doc/scipy/)
        - [Pandas DataFrames](https://pandas.pydata.org/)
        - [Scikit-learn](https://scikit-learn.org/)
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

logger.info(f"Session {st.session_state.session_id}: Landing page loaded")
