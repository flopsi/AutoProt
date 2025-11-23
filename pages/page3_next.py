import streamlit as st

st.set_page_config(page_title="Next Step", page_icon="🧬")

st.title("Next Placeholder")
st.info("This page is a placeholder for future functionality.")

st.sidebar.header("Navigation")
if st.sidebar.button("Back to Main page"):
    st.switch_page("Home.py")
