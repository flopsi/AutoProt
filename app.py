# app.py
import streamlit as st

# IMPORTANT: Only call set_page_config once, and at the top-level of app.py. [web:164]
st.set_page_config(
    page_title="DIA Proteomics Pipeline",
    layout="wide",
)

def main():
    st.title("DIA Proteomics Pipeline")

    st.markdown(
        """
        This is the main entry point of the DIA Proteomics Pipeline.

        - In the sidebar, click **Upload Protein & Peptide** to open the upload page.
        - On that page you can:
          1. Upload a wide-format protein or peptide table.
          2. Assign roles to each column using the 2‑column mapping table
             (Protein Group ID, Peptide Sequence, Quantitative, Species, etc.).
          3. Preview the transformed DataFrame with the correct index
             (Protein Group for proteins, peptide sequence or generated ID for peptides).
        - Downstream pages (QC, statistics, visualization) can read from:
          - `st.session_state["protein_upload"]`
          - `st.session_state["peptide_upload"]`
        """
    )

    st.info(
        "If you only see this page and the sidebar, click the "
        "**Upload Protein & Peptide** page in the sidebar to start."
    )

if __name__ == "__main__":
    main()
