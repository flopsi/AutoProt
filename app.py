"""
DIA Proteomics Analysis Framework - Homepage
Thermo Fisher Scientific
"""
import streamlit as st
from config import THERMO_COLORS
from models import SessionKeys

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Proteomics Analysis | Thermo Fisher",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
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
  border-radius: 8px;
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

/* Cards */
.feature-card {{
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 25px;
  margin: 15px 0;
  transition: box-shadow 0.3s;
}}

.feature-card:hover {{
  box-shadow: 0 4px 12px rgba(231, 19, 22, 0.15);
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="header-banner">
  <h1>🧬 DIA Proteomics Analysis Framework</h1>
  <p>Comprehensive Data Import, Validation & Statistical Analysis | Thermo Fisher Scientific</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# WELCOME CONTENT
# ============================================================================

st.markdown("## Welcome to the DIA Proteomics Analysis Platform")

st.info("""
📝 **Getting Started:** Use the sidebar to navigate between pages:
- **Protein-Level Upload**: Import and configure protein-level data
- **Peptide-Level Upload**: Import and configure peptide-level data (optional)
- **Analysis**: Perform statistical analysis on uploaded data
""")

# Feature cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>✨ Key Features</h3>
        <ul>
            <li><strong>Smart Column Detection</strong> - Auto-detects quant/metadata columns</li>
            <li><strong>Name Trimming</strong> - Cleans file paths and prefixes automatically</li>
            <li><strong>Interactive Selection</strong> - Choose exactly which columns to keep</li>
            <li><strong>Flexible Annotation</strong> - Assign Control/Treatment conditions</li>
            <li><strong>Dual-Level Support</strong> - Protein and/or peptide-level data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Statistical Analysis</h3>
        <ul>
            <li><strong>Multiple Tests</strong> - t-test, Mann-Whitney, ANOVA, Kruskal-Wallis</li>
            <li><strong>Normalization</strong> - Log2, Median, Quantile, Z-Score</li>
            <li><strong>Imputation</strong> - Handle missing values intelligently</li>
            <li><strong>Quality Control</strong> - CV analysis, intensity balance</li>
            <li><strong>Visualization</strong> - Interactive plots and charts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# STATUS OVERVIEW
# ============================================================================
st.markdown("---")
st.subheader("📋 Current Status")

col1, col2, col3 = st.columns(3)

with col1:
    if SessionKeys.PROTEIN_DATASET.value in st.session_state:
        dataset = st.session_state[SessionKeys.PROTEIN_DATASET.value]
        st.success("✅ **Protein Data Loaded**")
        st.metric("Proteins", f"{dataset.n_proteins:,}")
        st.metric("Samples", dataset.n_samples)
    else:
        st.warning("⏳ **Protein Data Not Loaded**")
        st.caption("Go to 'Protein-Level Upload' to get started")

with col2:
    if SessionKeys.PEPTIDE_DATASET.value in st.session_state:
        dataset = st.session_state[SessionKeys.PEPTIDE_DATASET.value]
        st.success("✅ **Peptide Data Loaded**")
        st.metric("Peptides", f"{dataset.n_proteins:,}")
        st.metric("Samples", dataset.n_samples)
    else:
        st.info("ℹ️ **Peptide Data Optional**")
        st.caption("Peptide-level data provides additional statistical power")

with col3:
    if SessionKeys.RESULTS.value in st.session_state:
        st.success("✅ **Analysis Complete**")
        st.caption("View results in Analysis page")
    else:
        st.info("ℹ️ **Ready for Analysis**")
        st.caption("Upload data and run analysis")

# ============================================================================
# QUICK START GUIDE
# ============================================================================
st.markdown("---")
st.markdown("### 🚀 Quick Start Guide")

st.markdown("""
1. **Upload Protein Data** (Required)
   - Navigate to "Protein-Level Upload" in the sidebar
   - Upload your protein quantification file (CSV/TSV)
   - Select columns and assign Control/Treatment conditions
   
2. **Upload Peptide Data** (Optional)
   - Navigate to "Peptide-Level Upload" for enhanced analysis
   - Follow the same process as protein-level
   
3. **Run Analysis**
   - Go to "Analysis" page
   - Configure statistical parameters
   - View results and download reports
""")

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
