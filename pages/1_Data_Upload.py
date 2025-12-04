"""
pages/1_Data_Upload.py
Upload → Detect columns → Species annotation → Show raw histogram
"""

import streamlit as st
import pandas as pd
from helpers.constants import get_theme
from helpers.file_io import read_csv, read_excel
from helpers.plots import create_density_plot
from helpers.audit import log_event

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_species_keywords():
    """Initialize default species keywords if not present"""
    if 'species_keywords' not in st.session_state:
        st.session_state.species_keywords = [
            {'keyword': 'HUMAN', 'species': 'Human'},     # ✓ Clean name
            {'keyword': 'YEAST', 'species': 'Yeast'},
            {'keyword': 'ECOLI', 'species': 'E. coli'}
        ]
    
    if 'species_column' not in st.session_state:
        st.session_state.species_column = None
    
    if 'species_mapping' not in st.session_state:
        st.session_state.species_mapping = {}

# Call at page load
init_species_keywords()

# ============================================================================
# SIDEBAR: Theme Selection
# ============================================================================

with st.sidebar:
    theme = st.selectbox("🎨 Theme", options=["light", "dark", "colorblind", "journal"], index=0)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("📊 Data Upload")

# Upload file
uploaded_file = st.file_uploader("Upload CSV/TSV/Excel file")

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith('.xlsx'):
        df = read_excel(uploaded_file)
    else:
        df = read_csv(uploaded_file)
    
    st.write(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
    
    # ============================================================================
    # SPECIES ANNOTATION
    # ============================================================================
    
    st.subheader("Species Annotation")
    
    # Select protein ID column
    text_cols = [col for col in df.columns if df[col].dtype == 'object']
    
    if text_cols:
        species_col = st.selectbox(
            "Select column containing protein IDs",
            options=text_cols,
            help="This column should contain protein identifiers (e.g., Q9Y6K9_HUMAN)"
        )
        
        st.session_state.species_column = species_col
        
        # Configure keywords (expandable)
        with st.expander("⚙️ Configure Species Keywords"):
            st.caption("Define keywords to identify species in protein IDs")
            
            for i, kw in enumerate(st.session_state.species_keywords):
                col1, col2 = st.columns(2)
                with col1:
                    keyword = st.text_input(
                        f"Keyword {i+1}",
                        value=kw['keyword'],
                        key=f"kw_{i}"
                    )
                with col2:
                    species = st.text_input(
                        f"Species {i+1}",
                        value=kw['species'],
                        key=f"sp_{i}"
                    )
                st.session_state.species_keywords[i] = {'keyword': keyword, 'species': species}
        
        # Assign species function
        def assign_species(val: str, keywords: list) -> str:
            """Assign species based on keywords - returns clean species name"""
            if pd.isna(val):
                return 'Unknown'
            
            val_str = str(val).upper()
            
            # Build mapping: keyword -> species name
            mapping = {
                kw['keyword']: kw['species'] 
                for kw in keywords 
                if kw.get('keyword') and kw.get('species')
            }
            
            # Check each keyword
            for keyword, species_name in mapping.items():
                if keyword.upper() in val_str:
                    return species_name  # ✓ Returns 'Human', NOT '_HUMAN'
            
            return 'Unknown'
        
        # Apply species assignment
        species_series = df[species_col].apply(
            lambda x: assign_species(x, st.session_state.species_keywords)
        )
        
        # Store in session
        st.session_state.species_mapping = species_series.to_dict()
        
        # Show distribution
        species_counts = species_series.value_counts()
        
        st.markdown("**Species Distribution:**")
        st.bar_chart(species_counts)  # ✓ Now shows "Human", "Yeast", "E. coli"
        
        # Show metrics
        cols = st.columns(len(species_counts))
        for i, (species, count) in enumerate(species_counts.items()):
            with cols[i]:
                pct = (count / len(species_series)) * 100
                st.metric(species, f"{count} ({pct:.1f}%)")
        
        # Log
        log_event(
            "Species Annotation",
            f"Annotated {len(species_series)} proteins",
            {"species_detected": species_counts.to_dict()}
        )
    
    st.success("✅ Data uploaded and annotated!")
