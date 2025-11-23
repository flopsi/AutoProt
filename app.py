import streamlit as st
from models import SessionKeys

st.set_page_config(
    page_title="DIA Proteomics App",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional: import your color scheme if you'd like to use it here
try:
    from config import THERMO_COLORS
    th_color = THERMO_COLORS['PRIMARY_RED']
    border_color = THERMO_COLORS.get('LIGHT_GRAY','#E2E3E4')
    bg_color = "#fff"
except:
    th_color = "#E71316"
    border_color = "#E2E3E4"
    bg_color = "#fff"

st.markdown(f"""
<style>
.header-banner {{
    background: linear-gradient(90deg, {th_color} 0%, #A6192E 100%);
    padding: 30px 40px;
    border-radius: 0;
    margin-bottom: 30px;
    color: white;
}}
.sidebar-title {{
    font-weight: bold;
    font-size: 16px;
    color: {th_color};
    margin-bottom: 8px;
}}
.status-card {{
    background: {bg_color};
    border: 1px solid {border_color};
    border-radius: 8px;
    margin-bottom: 12px;
    padding: 1.4em 1.8em;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-banner">
  <h1>🧬 DIA Proteomics Analysis Framework</h1>
  <p>Comprehensive Data Import, Validation & Statistical Analysis | Thermo Fisher Scientific</p>
</div>
""", unsafe_allow_html=True)

st.markdown("## Welcome!")
st.write(
    "This application provides a complete pipeline for DIA proteomics data upload, column selection, annotation, and statistical analysis. "
    "Navigate using the sidebar to access the protein, peptide, or next step modules."
)

# Sidebar nav (visual/informational only, Streamlit native sidebar handles pages)
st.sidebar.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
st.sidebar.info("Use the sidebar pages below to access upload and analysis modules.\
\n\nTo return to this home page at any time, click the app name or house icon at the top.")

# Show status for each flow
protein_status = st.session_state.get(SessionKeys.PROTEIN_DATASET.value, None)
peptide_status = st.session_state.get(SessionKeys.PEPTIDE_DATASET.value, None)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="status-card">', unsafe_allow_html=True)
    st.subheader("🔬 Protein Data")
    if protein_status:
        st.success("✅ Protein-level data loaded")
        st.metric("Proteins", f"{protein_status.n_proteins:,}")
        st.metric("Samples", protein_status.n_samples)
    else:
        st.warning("⚠️ Not loaded")
        st.caption("Go to the Protein page")
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown('<div class="status-card">', unsafe_allow_html=True)
    st.subheader("🧪 Peptide Data")
    if peptide_status:
        st.success("✅ Peptide-level data loaded")
        st.metric("Peptides", f"{peptide_status.n_proteins:,}")
        st.metric("Samples", peptide_status.n_samples)
    else:
        st.info("ℹ️ Optional")
        st.caption("Go to the Peptide page")
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown('<div class="status-card">', unsafe_allow_html=True)
    st.subheader("📊 Analysis")
    if protein_status or peptide_status:
        st.info("Ready to analyze")
        st.caption("Go to the Next page to run statistical analysis.")
    else:
        st.warning("Import data to begin.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### How To Use This App")
st.markdown("""
1. **Navigate to the Protein page** (sidebar) and upload your protein quantification file. Select and annotate columns as needed.
2. **Optionally upload peptide-level data** via the Peptide page for enhanced analysis.
3. **Proceed to the Next page** for statistical tests, normalization, and visualization (coming soon).
""")

st.markdown("---")
st.caption("© 2025 Your Organization | Confidential")
