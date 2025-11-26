
import streamlit as st
# ─────────────────────────────────────────────────────────────
# Session State Summary
# ─────────────────────────────────────────────────────────────
if "prot_df" in st.session_state or "pep_df" in st.session_state or "skip_prot" in st.session_state or "skip_pep" in st.session_state:
    st.markdown("### 📦 Current Session State")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "prot_df" in st.session_state:
            st.metric("**Protein Dataframe**", f"{st.session_state.prot_n:,} proteins")
            st.write(f"- Cond1: {len(st.session_state.prot_c1)} replicates")
            st.write(f"- Cond2: {len(st.session_state.prot_c2)} replicates")
            st.write(f"- Species: {', '.join(st.session_state.prot_species)}")
        elif "skip_prot" in st.session_state:
            st.warning("⏭️ Protein upload skipped")
        else:
            st.info("Protein data not yet loaded")
    
    with col2:
        if "pep_df" in st.session_state:
            st.metric("**Peptide Dataframe**", f"{st.session_state.pep_n:,} peptides")
            st.write(f"- Cond1: {len(st.session_state.pep_c1)} replicates")
            st.write(f"- Cond2: {len(st.session_state.pep_c2)} replicates")
            st.write(f"- Species: {', '.join(st.session_state.pep_species)}")
        elif "skip_pep" in st.session_state:
            st.warning("⏭️ Peptide upload skipped")
        else:
            st.info("Peptide data not yet loaded")

# Your Thermo Fisher CSS (same as before)
st.markdown("""
<style>
    :root {--primary-red: #E71316; --dark-red: #A6192E; --gray: #54585A; --light-gray: #E2E3E4;}
    html, body, [class*="css"] {font-family: Arial, sans-serif !important;}
    .header {background: linear-gradient(90deg, #E71316 0%, #A6192E 100%); padding: 20px 40px; color: white; margin: -60px -60px 40px -60px;}
    .header h1 {margin:0; font-size:28px; font-weight:600;}
    .header p {margin:5px 0 0 0; font-size:14px; opacity:0.95;}
    .nav {background:white; border-bottom:2px solid #E2E3E4; padding:0 40px; display:flex; gap:5px; margin:-20px -60px 40px -60px;}
    .nav-item {padding:15px 25px; font-size:14px; font-weight:500; color:#54585A; border-bottom:3px solid transparent; cursor:pointer;}
    .nav-item:hover {background:rgba(231,19,22,0.05);}
    .nav-item.active {border-bottom:3px solid #E71316; color:#E71316;}
    .module-header {background:linear-gradient(90deg,#E71316 0%,#A6192E 100%); padding:30px; border-radius:8px; margin-bottom:40px; color:white; display:flex; align-items:center; gap:20px;}
    .module-icon {width:60px; height:60px; background:rgba(255,255,255,0.2); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:32px;}
    .card {background:white; border:1px solid #E2E3E4; border-radius:8px; padding:25px; box-shadow:0 2px 4px rgba(0,0,0,0.05); margin-bottom:25px;}
    .card:hover {box-shadow:0 4px 12px rgba(0,0,0,0.1); transform:translateY(-2px);}
    .upload-area {border:2px dashed #E2E3E4; border-radius:8px; padding:60px 30px; text-align:center; background:#fafafa; cursor:pointer;}
    .upload-area:hover {border-color:#E71316; background:rgba(231,19,22,0.02);}
    .stButton>button {background:#E71316 !important; color:white !important; border:none !important; padding:12px 24px !important; border-radius:6px !important; font-weight:500 !important;}
    .stButton>button:hover {background:#A6192E !important;}
    .footer {text-align:center; padding:30px; color:#54585A; font-size:12px; border-top:1px solid #E2E3E4; margin-top:60px;}
</style>
""", unsafe_allow_html=True)



st.set_page_config(page_title="autoProt", page_icon ="🧬")


pages = {
    "Data Upload": [
        st.Page("pages/protein.py", title="Upload Protein Data"),
        st.Page("pages/peptide.py", title="Upload Peptide Data"),
    ],
    "Data Quality": [
        st.Page("pages/quality.py", title="Exploratory Data Analysis"),

    ],
}





