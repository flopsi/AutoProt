import streamlit as st

st.set_page_config(
    page_title="DIA Proteomics App",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("DIA Proteomics Analysis Framework")
st.markdown("""
Welcome to the DIA Proteomics App!

- The sidebar (left) lets you access individual modules:
    - Protein-Level Upload
    - Peptide-Level Upload
    - (Next, Analysis, or others as you add them)

Each module is a separate page in the app. The widgets used on each page will always be local to that page—**they don't appear or interfere across pages**.

**To get started, select a page from the sidebar.**
""")

# Optional project/company info, styling, branding, or instructions...
st.markdown("---")
st.caption("© 2025 Your Organization | Confidential")
