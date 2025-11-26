# app.py — Main entry point
import streamlit as st

st.set_page_config(
    page_title="LFQbench Proteomics Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Thermo Fisher CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {--primary-red: #E71316; --dark-red: #A6192E; --gray: #54585A; --light-gray: #E2E3E4;}
    html, body, [class*="css"] {font-family: Arial, sans-serif !important;}
    .header {background: linear-gradient(90deg, #E71316 0%, #A6192E 100%); padding: 20px 40px; color: white; margin: -60px -60px 40px -60px;}
    .header h1 {margin:0; font-size:32px; font-weight:600;}
    .header p {margin:10px 0 0 0; font-size:16px; opacity:0.95;}
    .card {background:white; border:1px solid #E2E3E4; border-radius:8px; padding:30px; box-shadow:0 2px 8px rgba(0,0,0,0.08); margin-bottom:25px;}
    .footer {text-align:center; padding:30px; color:#54585A; font-size:12px; border-top:1px solid #E2E3E4; margin-top:60px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>🔬 LFQbench Proteomics Analysis</h1>
    <p>Quantitative accuracy assessment platform</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="card">
    <h2>📊 Welcome to LFQbench</h2>
    <p>A comprehensive platform for analyzing label-free quantification (LFQ) proteomics data.</p>
    
    <h3>Getting Started:</h3>
    <ul>
        <li><strong>Protein Import</strong> — Upload and configure your protein-level data</li>
        <li><strong>Peptide Import</strong> — Upload and configure your peptide-level data</li>
    </ul>
    
    <h3>Navigation:</h3>
    <p>Use the sidebar to navigate between pages. All data is stored in the session state and persists as you switch pages.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Session State Summary
# ─────────────────────────────────────────────────────────────
if "prot_df" in st.session_state or "pep_df" in st.session_state or "skip_prot" in st.session_state or "skip_pep" in st.session_state:
    st.markdown("### 📦 Current Session State")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "prot_df" in st.session_state:
            st.metric("**Protein Data**", f"{st.session_state.prot_n:,} proteins")
            st.write(f"- Cond1: {len(st.session_state.prot_c1)} replicates")
            st.write(f"- Cond2: {len(st.session_state.prot_c2)} replicates")
            st.write(f"- Species: {', '.join(st.session_state.prot_species)}")
        elif "skip_prot" in st.session_state:
            st.warning("⏭️ Protein upload skipped")
        else:
            st.info("Protein data not yet loaded")
    
    with col2:
        if "pep_df" in st.session_state:
            st.metric("**Peptide Data**", f"{st.session_state.pep_n:,} peptides")
            st.write(f"- Cond1: {len(st.session_state.pep_c1)} replicates")
            st.write(f"- Cond2: {len(st.session_state.pep_c2)} replicates")
            st.write(f"- Species: {', '.join(st.session_state.pep_species)}")
        elif "skip_pep" in st.session_state:
            st.warning("⏭️ Peptide upload skipped")
        else:
            st.info("Peptide data not yet loaded")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Restart Button (Fixed Bottom)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .fixed-restart {
        position: fixed; 
        bottom: 20px; 
        left: 50%; 
        transform: translateX(-50%); 
        z-index: 999;
        width: 300px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="fixed-restart">', unsafe_allow_html=True)
if st.button("🔄 Restart Analysis", use_container_width=True, key="restart_main"):
    st.session_state.clear()
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)
