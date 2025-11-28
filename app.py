# app.py
import streamlit as st
st.set_page_config(page_title="DIA Proteomics Pipeline", layout="wide")

st.title("DIA Proteomics Pipeline")

st.markdown(
    """
    - Go to **Upload Protein & Peptide** in the sidebar.
    - There you can upload a table and map each column using the 2‑column mapping UI.
    """
)

st.info("Use the sidebar navigation on the left to open the upload page.")

