# app.py — Main entry point (FULLY FIXED)
import streamlit as st

st.set_page_config(
    page_title="LFQbench Proteomics Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Thermo Fisher CSS (FIXED)
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
    .feature-grid {
        display: grid; 
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); 
        gap: 20px; 
        margin: 20px 0;
    }
    .feature-box {
        background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
        border: 2px solid #E2E3E4;
        border-radius: 8px; 
        padding: 20px;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .feature-box:hover {
        border-color: #E71316;
        box-shadow: 0 4px 12px rgba(231, 19, 22, 0.15);
        transform: translateY(-2px);
    }
    .feature-box h4 {
        margin: 0 0 10px 0; 
        color: #E71316;
        font-size: 16px;
        font-weight: 600;
    }
    .feature-box p {
        margin: 0;
        color: #54585A;
        font-size: 14px;
        line-height: 1.5;
        flex-grow: 1;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>🔬 LFQbench Proteomics Analysis</h1>
    <p>Quantitative accuracy assessment platform for label-free quantification (LFQ)</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Session State Summary
# ─────────────────────────────────────────────────────────────
if "prot_df" in st.session_state or "pep_df" in st.session_state or "skip_prot" in st.session_state or "skip_pep" in st.session_state:
    st.markdown("---")
    st.markdown("### 📦 Current Session State")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "prot_df" in st.session_state:
            st.success("✅ **Protein Data Loaded**")
            st.metric("Proteins", f"{st.session_state.prot_n:,}")
            st.write(f"**Conditions:** Cond1 ({len(st.session_state.prot_c1)}), Cond2 ({len(st.session_state.prot_c2)})")
            st.write(f"**Species:** {', '.join(st.session_state.prot_species)}")
        elif "skip_prot" in st.session_state:
            st.warning("⏭️ **Protein upload skipped**")
        else:
            st.info("Protein data not yet loaded")
    
    with col2:
        if "pep_df" in st.session_state:
            st.success("✅ **Peptide Data Loaded**")
            st.metric("Peptides", f"{st.session_state.pep_n:,}")
            st.write(f"**Conditions:** Cond1 ({len(st.session_state.pep_c1)}), Cond2 ({len(st.session_state.pep_c2)})")
            st.write(f"**Species:** {', '.join(st.session_state.pep_species)}")
        elif "skip_pep" in st.session_state:
            st.warning("⏭️ **Peptide upload skipped**")
        else:
            st.info("Peptide data not yet loaded")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Tips Section
# ─────────────────────────────────────────────────────────────
with st.container():
    with st.expander("💡 Tips & Troubleshooting"):
        st.markdown("""
        **File Format:**
        - Supports CSV, TSV, and TXT formats
        - First row should contain column headers
        - Numeric columns are automatically detected for intensity data
        - Missing values can be represented as empty cells or #NUM!
        
        **Column Detection:**
        - Species column is auto-detected by searching for "HUMAN" (case-insensitive)
        - Condition columns are identified by pattern matching (e.g., _Y05-E45_)
        - Protein/Peptide columns default to "Protein.Group" or "PEP.StrippedSequence"
        
        **Data Validation:**
        - At least 1 replicate required per condition
        - Proteins/peptides with intensity >1 in ≥2/3 replicates are counted as "present"
        - Species are extracted from protein/peptide names using delimiters (e.g., _HUMAN_)
        
        **Session Management:**
        - Click "Restart Analysis" to clear all data and start fresh
        - Use "Skip" button to upload only protein or only peptide data
        - Navigate freely between pages — your data persists automatically
        """)

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
