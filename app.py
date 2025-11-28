# app.py
import streamlit as st

st.set_page_config(page_title="DIA Proteomics Pipeline", layout="wide")

st.markdown(
    """
    # DIA Proteomics Pipeline

    Use the sidebar to navigate through the workflow.
    """
)
st.markdown(
    """
    ### Modules

    - Protein Import (upload protein + metadata, map design)
    - Downstream pages (QC, statistics, visualization) will consume the
      processed data stored in `st.session_state.protein_upload`.
    """
)
