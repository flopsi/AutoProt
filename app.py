"""
app.py

Main entry point for AutoProt Streamlit application
Handles session initialization, navigation, and global configuration
"""

import streamlit as st
from helpers.core import get_theme_names
from helpers.ui import init_session_state

# ============================================================================
# PAGE CONFIGURATION
# Set up Streamlit page settings (must be first Streamlit command)
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
# SESSION INITIALIZATION
# Initialize session state variables and audit logging
# ============================================================================

# Initialize audit session (tracks user actions)
init_audit_session()

# Initialize session state variables with defaults
init_session_state("theme", "light")
init_session_state("protein_data", None)
init_session_state("transform_cache", {})
init_session_state("current_transform", "log2")
init_session_state("analysis_results", None)

# ============================================================================
# SIDEBAR - GLOBAL SETTINGS
# Theme selector and app-wide configuration
# ============================================================================

with st.sidebar:
    st.title("🧬 AutoProt")
    st.caption("Proteomics Data Analysis Platform")
    
    st.markdown("---")
    
    # Theme selector
    st.subheader("⚙️ Settings")
    theme_labels = {
        "light": "☀️ Light",
        "dark": "🌙 Dark",
        "colorblind": "🎨 Colorblind-Friendly",
        "journal": "📄 Journal (B&W)"
    }
    
    selected_theme = st.selectbox(
        "Color Theme",
        options=get_theme_names(),
        format_func=lambda x: theme_labels.get(x, x.title()),
        key="theme",
        help="Select color scheme for plots and visualizations"
    )
    
    st.markdown("---")
    
    # Navigation info
    st.subheader("📊 Workflow")
    st.markdown("""
    1. **Upload Data** - Load CSV/TSV/Excel
    2. **Visual EDA** - Transform & explore
    3. **Statistical EDA** - Compare groups
    """)
    
    st.markdown("---")
    
    # Session info
    if st.session_state.get("protein_data"):
        st.success("✅ Data loaded")
        protein_data = st.session_state.protein_data
        st.metric("Proteins", protein_data.n_proteins)
        st.metric("Samples", protein_data.n_samples)
        st.metric("Missing", f"{protein_data.missing_rate:.1f}%")
    else:
        st.info("📁 No data loaded")

# ============================================================================
# MAIN CONTENT
# Landing page with instructions and quick start guide
# ============================================================================

st.title("🧬 AutoProt - Proteomics Analysis Platform")

st.markdown("""
Welcome to **AutoProt**, a comprehensive proteomics data analysis tool built with Streamlit.
This platform automates common proteomics workflows including data transformation, 
exploratory data analysis, and statistical testing.
""")

# Feature overview
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📁 Data Upload
    - **Multi-format support**: CSV, TSV, Excel
    - **Auto-detection**: Numeric columns, protein IDs, species
    - **Validation**: Quality checks and missing data analysis
    """)

with col2:
    st.markdown("""
    ### 📊 Visual EDA
    - **10+ transformations**: log2, Yeo-Johnson, quantile, etc.
    - **Normality testing**: Shapiro-Wilk, Q-Q plots
    - **Interactive plots**: PCA, heatmaps, distributions
    """)

with col3:
    st.markdown("""
    ### 🧪 Statistical Testing
    - **Differential expression**: Welch's t-test, ANOVA
    - **FDR correction**: Benjamini-Hochberg
    - **Visualization**: Volcano plots, MA plots
    """)

st.markdown("---")

# Quick start guide
st.header("🚀 Quick Start Guide")

with st.expander("**Step 1: Upload Your Data**", expanded=True):
    st.markdown("""
    Navigate to **📁 1_Data_Upload** in the sidebar and upload your proteomics data file.
    
    **Required format:**
    - First column: Protein/Gene IDs
    - Numeric columns: Sample intensities (e.g., A1, A2, B1, B2)
    - Optional: Species annotation column
    
    **Supported files:** CSV, TSV, TXT, XLSX (max 100 MB)
    """)

with st.expander("**Step 2: Transform & Explore**"):
    st.markdown("""
    Go to **📊 2_Visual_EDA** to:
    - Select optimal transformation (log2, Yeo-Johnson, etc.)
    - Assess normality with Shapiro-Wilk tests
    - Visualize distributions, PCA, and heatmaps
    - Filter by species if applicable
    
    **Tip:** Use Q-Q plots to evaluate transformation effectiveness.
    """)

with st.expander("**Step 3: Statistical Analysis**"):
    st.markdown("""
    Navigate to **🧪 3_Statistical_EDA** to:
    - Define experimental groups (e.g., Control vs Treatment)
    - Run differential expression analysis (t-test or ANOVA)
    - Generate volcano plots with custom thresholds
    - Export results as CSV or Excel
    
    **Metrics:** log2FC, p-value, FDR, regulation status
    """)

st.markdown("---")

# System info
st.header("ℹ️ System Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Current Session:**
    - Session ID: `{}`
    - Theme: **{}**
    - Data loaded: **{}**
    """.format(
        st.session_state.get("session_id", "unknown"),
        st.session_state.get("theme", "light").title(),
        "Yes" if st.session_state.get("protein_data") else "No"
    ))

with col2:
    st.markdown("""
    **Performance Features:**
    - ⚡ Cached transformations (1 hour TTL)
    - ⚡ Cached statistical tests
    - ⚡ Lazy imports for faster loading
    - 💾 Session state persistence
    """)

st.markdown("---")

# Footer
st.caption("""
**AutoProt** | Built with Streamlit | 
[Documentation](#) | [GitHub](#) | [Report Issue](#)
""")
