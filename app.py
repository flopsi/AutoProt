"""
app.py
Main Streamlit entry point for AutoProt proteomics analysis platform
"""

import streamlit as st

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AutoProt",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# SIDEBAR: GLOBAL SETTINGS
# ============================================================================

with st.sidebar:
    st.title("⚙️ Settings")
    
    # Theme selector
    theme = st.selectbox(
        "🎨 Visual Theme",
        options=["light", "dark", "colorblind", "journal"],
        index=0,
        help="Change color scheme across all plots"
    )
    
    # Store in session
    st.session_state.theme = theme
    
    st.divider()
    
    # About section
    st.markdown("""
    ### About AutoProt
    
    **Version:** 0.1.0  
    **Phase:** 1 (Core Analysis)
    
    Proteomics analysis platform with:
    - Multi-format file support
    - Statistical testing (t-test, ROC, precision-recall)
    - 4 color themes
    - Audit trail logging
    
    Use the navigation to:
    1. Upload data
    2. Explore with EDA
    3. Preprocess & filter
    4. Run analysis
    """)
    
    st.divider()
    
    # Session state debug (optional)
    if st.checkbox("🔍 Debug Info"):
        st.write("**Session State Keys:**")
        for key in st.session_state.keys():
            st.write(f"  • {key}")

# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("🧬 AutoProt - Proteomics Analysis Platform")

st.markdown("""
Welcome to **AutoProt**, an open-source proteomics analysis pipeline.

### Quick Start

1. **Upload Data** → Load your proteomics data (CSV, TSV, Excel)
2. **Explore (EDA)** → Visualize data distribution and quality
3. **Preprocess** → Apply transforms and filters
4. **Analyze** → Differential expression, statistics, plots
5. **Download** → Export results and visualizations

### Features

✅ **Multi-format support** (CSV, TSV, Excel)  
✅ **Statistical tests** (t-test, FDR, ROC curves, precision-recall)  
✅ **4 color themes** (Light, Dark, Colorblind-friendly, Journal)  
✅ **6 transformations** (log2, log10, sqrt, cbrt, Yeo-Johnson, quantile)  
✅ **Peptide support** (Aggregation to protein level)  
✅ **Audit trail** (Full data lineage tracking)  
✅ **Species detection** (Human, Yeast, E.coli, Mouse)  

### Data Requirements

Your data should be in one of these formats:

**Protein-level data:**
| Protein ID | Condition_Rep1 | Condition_Rep2 | Treatment_Rep1 | Treatment_Rep2 | Species |
|---|---|---|---|---|---|
| PROTEIN_A | 1000 | 1200 | 5000 | 4800 | HUMAN |
| PROTEIN_B | 2000 | 2100 | 1500 | 1600 | HUMAN |

**Peptide-level data:**
| Peptide Sequence | Protein ID | A1 | A2 | B1 | B2 |
|---|---|---|---|---|---|
| PEPTIDESEQ1 | PROTEIN_A | 100 | 120 | 500 | 480 |
| PEPTIDESEQ2 | PROTEIN_A | 50 | 60 | 200 | 190 |

### Navigation

Use the sidebar to select a page:
- **1_Data_Upload** - Upload and explore your data
- **2_Visual_EDA** - Distribution analysis
- **3_Statistical_EDA** - Quality control metrics
- **4_Preprocessing** - Transformation and filtering
- **6_Analysis** - Differential expression analysis

### Tips

💡 **Missing data OK** - The platform handles NaN values gracefully  
💡 **Column names flexible** - Auto-detects protein ID and species columns  
💡 **Themes persistent** - Your theme selection carries through all pages  
💡 **Audit trail enabled** - All operations are logged for reproducibility  

---

**Ready to start?** Navigate to **1_Data_Upload** using the sidebar menu.

For detailed documentation, see the docs/ folder.
""")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Documentation**
    - [ARCHITECTURE.md](docs/ARCHITECTURE.md)
    - [THEME_GUIDE.md](docs/THEME_GUIDE.md)
    """)

with col2:
    st.markdown("""
    **Quick Links**
    - [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)
    - [IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)
    """)

with col3:
    st.markdown("""
    **Support**
    - Version: 0.1.0
    - Status: Phase 1
    - Feedback: See docs/
    """)

st.caption("AutoProt © 2025 | Proteomics Analysis Framework")
