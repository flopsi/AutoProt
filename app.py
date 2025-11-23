import streamlit as st

st.set_page_config(page_title="DIA Proteomics App", page_icon="🧬", layout="wide")

st.title("DIA Proteomics Analysis Framework")
st.markdown("## Welcome!")
st.write("This is the main app page. Use the sidebar to navigate to other modules (protein, peptide, etc.).")

st.sidebar.header("Navigation")
st.sidebar.info("Use the sidebar page list to switch between modules.")
