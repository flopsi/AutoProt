import streamlit as st


st.set_page_config(page_title="DIA Proteomics App", page_icon="🧬", layout="wide")

st.title("DIA Proteomics Analysis Framework")
st.markdown("## Welcome!")
st.write("This is the main app page. Use the sidebar to navigate to other modules (protein, peptide, etc.).")

st.sidebar.header("Navigation")
st.sidebar.info("Use the sidebar page list to switch between modules.")

st.set_page_config(
    page_title="DIA Proteomics App",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Header ----
st.title("DIA Proteomics Analysis Framework")
st.markdown("## Welcome!")
st.write("This is a placeholder for your app's main description. Use the sidebar to navigate between pages.")

# ---- Sidebar Navigation ----
st.sidebar.header("Navigation")
nav = st.sidebar.radio(
    "Go to page:",
    [
        "Main page",
        "Protein page",
        "Peptide page",
        "Next (placeholder)"
    ]
)

if nav == "Main page":
    st.markdown("### Your Main Page")
    st.info("This is your main page. Use the sidebar to visit other modules.")

elif nav == "Protein page":
    st.markdown("### Protein-Level Module")
    st.info("Navigate to the Protein page for protein-level annotation and upload.")
    # Optional: st.write("Or, click [here](./pages/1_Protein.py) to open directly.")  # direct streamlit links not supported, navigation is by sidebar

elif nav == "Peptide page":
    st.markdown("### Peptide-Level Module")
    st.info("Navigate to the Peptide page for peptide-level annotation and upload.")

elif nav == "Next (placeholder)":
    st.markdown("### Next Placeholder")
    st.info("This page is a placeholder for future steps.")

# ---- Link back to main page ----
st.sidebar.markdown("---")
st.sidebar.button("Back to Main page", on_click=lambda: st.experimental_rerun())  # Reruns app, resets sidebar

# ---- Footer ----
st.markdown("---")
st.caption("© 2024 Your Organization | Confidential")
