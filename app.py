import streamlit as st

pg = st.navigation([st.Page("page1_protein.py"), st.Page("page2_peptide.py")])

st.sidebar.selectbox("Group", ["A","B","C"], key="group")
st.sidebar.slider("Size", 1, 5, key="size")

pg.run()
