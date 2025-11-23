import streamlit as st

st.set_page_config(
    page_title="Protein Page",
    page_icon="🧬"
)

st.title("Protein Page")
st.info("This is the Protein page. Use the sidebar to return to the main page or navigate further.")

st.sidebar.header("Navigation")
if st.sidebar.button("Back to Main page"):
    st.switch_page("Home.py")  # Will go to main entrypoint if using Streamlit 1.22+
# Or simply instruct users to use the sidebar nav options
