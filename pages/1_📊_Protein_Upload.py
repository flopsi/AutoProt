import streamlit as st
import pandas as pd
from components.header import render_header

render_header()
st.title("Data Quality Assessment")

protein_uploaded = st.session_state.get('protein_uploaded', False)
peptide_uploaded = st.session_state.get('peptide_uploaded', False)

if not protein_uploaded:
    st.warning("⚠️ No data loaded. Please upload protein data first.")
    if st.button("Go to Protein Upload", type="primary", use_container_width=True):
        st.switch_page("pages/1_📊_Protein_Upload.py")
    st.stop()

# Show loaded data status
st.success(f"✓ Protein data loaded: {st.session_state.protein_data.n_proteins:,} proteins")
if peptide_uploaded:
    st.success(f"✓ Peptide data loaded: {st.session_state.peptide_data.n_rows:,} peptides")
else:
    st.info("ℹ️ Peptide data not loaded (optional)")

st.markdown("---")

# Data summary
if protein_uploaded:
    protein_data = st.session_state.protein_data
    
    st.markdown("### Current Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Proteins", f"{protein_data.n_proteins:,}")
    with col2:
        n_a = len([c for c in protein_data.condition_mapping.values() if c.startswith('A')])
        st.metric("Condition A Samples", n_a)
    with col3:
        n_b = len([c for c in protein_data.condition_mapping.values() if c.startswith('B')])
        st.metric("Condition B Samples", n_b)
    with col4:
        species_counts = protein_data.get_species_counts()
        st.metric("Species Detected", len([s for s in species_counts.values() if s > 0]))
    
    # Species breakdown
    st.markdown("#### Species Distribution")
    species_df = pd.DataFrame({
        'Species': [sp.capitalize() for sp in species_counts.keys()],
        'Count': list(species_counts.values()),
        'Percentage': [f"{(count/sum(species_counts.values())*100):.1f}%" 
                      for count in species_counts.values()]
    })
    st.dataframe(species_df, hide_index=True, use_container_width=True)

st.markdown("---")

# Placeholder content
st.info("📋 This module is under development.")

st.markdown("""
### Planned Features

The Data Quality Assessment module will include:

#### 1. **Missing Value Analysis**
- Heatmap of missing values across samples
- Percentage of proteins/peptides detected per sample
- Detection completeness metrics

#### 2. **Coefficient of Variation (CV%) Analysis**
- CV% distribution per condition (A vs B)
- Technical replicate variability assessment
- Quality thresholds and filtering recommendations

#### 3. **Intensity Distribution**
- Log-transformed intensity distributions per sample
- Box plots and violin plots
- Outlier detection

#### 4. **Multivariate Analysis**
- Principal Component Analysis (PCA)
- Sample clustering dendrograms
- Correlation heatmaps between replicates

---

*Navigate to other pages using the sidebar or buttons below.*
""")

# Navigation buttons
st.markdown("---")
st.markdown("### Navigation")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("← Back to Protein Upload", use_container_width=True):
        st.switch_page("pages/1_📊_Protein_Upload.py")

with nav_col2:
    if st.button("Upload Peptide Data", use_container_width=True):
        st.switch_page("pages/2_🔬_Peptide_Upload.py")

with nav_col3:
    if st.button("🔄 Start New Analysis", type="primary", use_container_width=True):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.switch_page("pages/1_📊_Protein_Upload.py")
