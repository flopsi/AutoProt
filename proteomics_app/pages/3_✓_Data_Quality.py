import streamlit as st
from components.header import render_header

render_header()
st.title("Data Quality Assessment")

if not st.session_state.get('protein_uploaded'):
    st.warning("⚠️ No data loaded")
    if st.button("Go to Protein Upload"):
        st.switch_page("pages/1_📊_Protein_Upload.py")
else:
    protein_data = st.session_state.protein_data
    st.success(f"✓ Protein data: {protein_data.n_proteins:,} proteins")
    
    st.markdown("---")
    st.info("📋 Quality metrics module under development")
    
    st.markdown('''
    ### Planned Features
    - Missing value analysis
    - CV% calculations
    - Intensity distributions
    - PCA / clustering
    - Correlation heatmaps
    ''')