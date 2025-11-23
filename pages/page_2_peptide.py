import streamlit as st

st.set_page_config(
    page_title="Peptide Page",
    page_icon="🧬"
)

st.title("Peptide Page")
st.info("This is the Peptide page. Use the sidebar to return to the main page or navigate further.")

st.sidebar.header("Navigation")
if st.sidebar.button("Back to Main page"):
    st.switch_page("Home.py")
