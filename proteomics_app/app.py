import streamlit as st
from pathlib import Path
from components.header import render_header

st.set_page_config(
    page_title="LFQ Proteomics Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

css_path = Path("styles") / "thermo_fisher.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if 'protein_data' not in st.session_state:
    st.session_state.protein_data = None
if 'protein_uploaded' not in st.session_state:
    st.session_state.protein_uploaded = False
if 'peptide_data' not in st.session_state:
    st.session_state.peptide_data = None
if 'peptide_uploaded' not in st.session_state:
    st.session_state.peptide_uploaded = False

render_header()

st.title("LFQ Proteomics Analysis Platform")

st.markdown('''
Welcome to the Thermo Fisher Scientific Proteomics Data Analysis Platform.

## Getting Started

1. **Upload Protein Data** - Navigate to Protein Upload page
2. **Upload Peptide Data** (Optional) - Load peptide-level data
3. **Review Data Quality** - Assess metrics

## LFQ Bench Workflow

Compare two experimental conditions (A vs B) with:
- Automatic species detection (HUMAN, ECOLI, YEAST)
- Semi-automated condition assignment
- Interactive visualizations
''')

st.info("👈 Use the sidebar to navigate")

st.markdown("---\n**Version**: 1.0.0 | © 2025 Thermo Fisher Scientific Inc.")