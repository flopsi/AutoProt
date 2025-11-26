# shared.py
import streamlit as st

def restart_button():
    st.markdown('<div style="height:120px;"></div>', unsafe_allow_html=True)
    if st.button("Restart Full Analysis", type="primary", use_container_width=True):
        # ONLY this button clears everything
        for key in list(st.session_state.keys()):
            if key.startswith(("prot_", "pept_", "uploaded_")):
                del st.session_state[key]
        st.cache_data.clear()
        st.success("Everything cleared — start fresh")
        st.rerun()

# NEW: global place to keep the actual uploaded file objects
def get_protein_file():
    return st.session_state.get("uploaded_protein_file")

def set_protein_file(file):
    st.session_state["uploaded_protein_file"] = file

def get_peptide_file():
    return st.session_state.get("uploaded_peptide_file")

def set_peptide_file(file):
    st.session_state["uploaded_peptide_file"] = file
