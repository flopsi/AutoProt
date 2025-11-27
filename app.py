# app.py
import streamlit as st

st.set_page_config(
    page_title="DIA Proteomics Pipeline",
    page_icon="Microscope",
    layout="wide"
)

# Global debug toggle (optional)
if "DEBUG" not in st.session_state:
    st.session_state.DEBUG = True  # Set False for production

st.markdown("""
<style>
    .big-title {font-size: 3.5rem !important; font-weight: 700; color: #E71316; text-align: center; margin: 2rem 0;}
    .subtitle {font-size: 1.4rem; text-align: center; color: #54585A; margin-bottom: 3rem;}
    .card {background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; height: 280px;}
    .card h3 {color: #E71316; margin: 1.5rem 0 1rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='big-title'>DIA Proteomics Pipeline</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Thermo Fisher Scientific • Proprietary & Confidential</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown("""
    ### Welcome!
    This pipeline supports:
    - Protein-level analysis
    - Peptide-level analysis
    - Equal replicate enforcement
    - Automatic species detection (incl. **ECOLI**)
    - Full data caching across pages
    """)

    st.markdown("### Start Analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Protein Import", type="primary", use_container_width=True):
            st.switch_page("pages/1_Protein_Import.py")
    with col_b:
        if st.button("Peptide Import", type="primary", use_container_width=True):
            st.switch_page("pages/2_Peptide_Import.py")


# Add this at the end of any page
st.markdown("""
<style>
    .restart-fixed {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 999;
        background: #E71316;
        color: white;
        padding: 15px 30px;
        border-radius: 10px;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        text-align: center;
    }
</style>
<div class="restart-fixed">
    🔄 Restart Entire Analysis
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2024 Thermo Fisher Scientific • Internal Use Only")
