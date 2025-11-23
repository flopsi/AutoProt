import streamlit as st
from components.header import render_header

render_header()
st.title("Data Quality Assessment")

protein_uploaded = st.session_state.get('protein_uploaded', False)
peptide_uploaded = st.session_state.get('peptide_uploaded', False)

if not protein_uploaded:
    st.warning("⚠️ No data loaded. Please upload protein data first.")
    if st.button("Go to Protein Upload"):
        st.switch_page("pages/1_📊_Protein_Upload.py")
    st.stop()

st.success(f"✓ Protein data loaded: {st.session_state.protein_data.n_proteins:,} proteins")
if peptide_uploaded:
    st.success(f"✓ Peptide data loaded: {st.session_state.peptide_data.n_rows:,} peptides")
else:
    st.info("ℹ️ Peptide data not loaded (optional)")

st.markdown("---")
st.info("📋 This module is under development.")

st.markdown("""
### Planned Features
- Missing value analysis
- Coefficient of Variation (CV%) per condition
- Intensity distribution plots
- PCA / sample clustering
- Correlation heatmaps between replicates

---

*Navigate to other pages using the sidebar.*
""")
