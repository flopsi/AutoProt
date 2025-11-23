import streamlit as st
from components.header import render_header

render_header()
st.title("Peptide Data Upload")

if not st.session_state.get('protein_uploaded'):
    st.warning("⚠️ Please upload protein data first")
    if st.button("Go to Protein Upload"):
        st.switch_page("pages/1_📊_Protein_Upload.py")
else:
    st.info("Peptide upload functionality coming soon!")
    st.markdown("For now, use protein-level data analysis.")