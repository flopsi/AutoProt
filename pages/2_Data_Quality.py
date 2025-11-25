# pages/2_Data_Quality.py
import streamlit as st

st.set_page_config(page_title="Data Quality | Thermo Fisher", layout="wide")

st.markdown("""
<div style="background: linear-gradient(90deg, #E71316 0%, #A6192E 100%); padding: 20px 40px; color: white; margin: -60px -60px 40px -60px;">
    <h1>Data Quality Assessment</h1>
    <p>Intensity distribution • Missing values • CVs • PCA • Rank plots</p>
</div>
""", unsafe_allow_html=True)

# Detect available data
has_protein = "df" in st.session_state
has_peptide = "df_peptide" in st.session_state

if not has_protein and not has_peptide:
    st.error("No data found. Please complete Protein or Peptide upload first.")
    st.stop()

# Show what data is loaded
if has_protein and has_peptide:
    st.success("Both Protein and Peptide data loaded")
elif has_protein:
    st.success("Protein-level data loaded")
else:
    st.success("Peptide-level data loaded")

# Your quality plots will go here...
st.write("Data Quality module ready — plots coming next!")

# ─────────────────────────────────────────────────────────────
# UNIVERSAL NAVIGATION — Works from ANY page, ANY workflow
# ─────────────────────────────────────────────────────────────
st.markdown("---")

# Main navigation buttons
col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.8])

with col1:
    if st.button("Protein Upload", use_container_width=True):
        st.switch_page("app.py")  # your main protein page

with col2:
    if st.button("Peptide Upload", use_container_width=True):
        st.switch_page("pages/1_Peptide_Data_Import.py")

with col3:
    if st.button("Data Quality", type="primary", use_container_width=True):
        # Check what data exists and route intelligently
        if "df" in st.session_state or "df_peptide" in st.session_state:
            st.switch_page("pages/2_Data_Quality.py")
        else:
            st.error("Please upload protein or peptide data first")
            st.stop()

with col4:
    if st.button("Restart Entire Analysis", type="secondary", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# Fixed bottom restart button (always visible)
st.markdown("""
<div style="position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 9999;">
    <div style="background: #E71316; color: white; padding: 14px 32px; border-radius: 8px; font-weight: 600; box-shadow: 0 6px 16px rgba(0,0,0,0.3); cursor: pointer;" 
         onclick="document.getElementById('restart-btn').click()">
        Restart Analysis — Clear All Data
    </div>
</div>
<button id="restart-btn" style="display:none" onclick="location.reload()"></button>
""", unsafe_allow_html=True)

# Hidden real restart button
if st.button("hidden_restart", key="hidden_restart_key"):
    st.session_state.clear()
    st.rerun()
