# app.py
import streamlit as st

st.set_page_config(page_title="DIA Proteomics Pipeline", layout="wide")

def main():
    st.title("DIA Proteomics Pipeline")

    st.markdown(
        """
        Welcome to the DIA Proteomics Pipeline.

        - Start on **Upload Protein and Peptide Data** in the sidebar.
        - Configure your protein and/or peptide tables there.
        - Downstream pages (QC, statistics, visualization) will use
          the processed objects stored in `st.session_state["protein_upload"]`
          and `st.session_state["peptide_upload"]`.
        """
    )

    st.info(
        "Use the navigation in the sidebar to switch between pages. "
        "If you only see this text and the sidebar, click on the "
        "**Upload Protein and Peptide Data** page."
    )

if __name__ == "__main__":
    main()
