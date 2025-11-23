import streamlit as st
from config import THERMO_COLORS

st.set_page_config(
    page_title="DIA Proteomics Analysis | Thermo Fisher",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Thermo Fisher branding
st.markdown(f"""
<style>
:root {{
  --primary-red: {THERMO_COLORS['PRIMARY_RED']};
  --primary-gray: {THERMO_COLORS['PRIMARY_GRAY']};
  --light-gray: {THERMO_COLORS['LIGHT_GRAY']};
  --navy: {THERMO_COLORS['NAVY']};
  --dark-red: {THERMO_COLORS['DARK_RED']};
  --orange: {THERMO_COLORS['ORANGE']};
  --yellow: {THERMO_COLORS['YELLOW']};
  --green: {THERMO_COLORS['GREEN']};
  --sky: {THERMO_COLORS['SKY']};
}}

@media (prefers-color-scheme: dark) {{
  :root {{
    --bg-primary: #0e1117;
    --bg-secondary: #262730;
    --text-primary: #fafafa;
    --text-secondary: #a3a8b8;
    --border-color: #3d4149;
  }}
}}

@media (prefers-color-scheme: light) {{
  :root {{
    --bg-primary: #f8f9fa;
    --bg-secondary: #ffffff;
    --text-primary: #54585A;
    --text-secondary: #6c757d;
    --border-color: #E2E3E4;
  }}
}}

.header-banner {{
  background: linear-gradient(90deg, var(--primary-red) 0%, var(--dark-red) 100%);
  padding: 30px 40px;
  border-radius: 0;
  margin-bottom: 30px;
  color: white;
}}

.header-banner h1 {{
  margin: 0;
  font-size: 28pt;
  color: white;
  font-weight: 600;
}}

.header-banner p {{
  margin: 5px 0 0 0;
  font-size: 14px;
  opacity: 0.95;
}}

.footer {{
  text-align: center;
  padding: 30px 0;
  color: var(--text-secondary);
  font-size: 12px;
  border-top: 1px solid var(--border-color);
  margin-top: 60px;
}}

.footer strong {{
  display: block;
  margin-bottom: 10px;
}}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class
""")

# Instructions
st.markdown("### How to Use This Application")

st.markdown("""
This application provides a complete workflow for DIA proteomics data analysis:

1. **Upload Protein Data** (Required)
   - Navigate to the Protein page using the sidebar
   - Upload your protein quantification file (CSV/TSV)
   - Select columns and assign Control/Treatment conditions
   
2. **Upload Peptide Data** (Optional)
   - Navigate to the Peptide page for enhanced statistical analysis
   - Follow the same process as protein-level data
   
3. **Run Statistical Analysis**
   - Go to the Analysis page
   - Configure normalization and statistical parameters
   - View results and download reports
""")

st.markdown("---")

# Status overview
st.markdown("### Current Status")

# Import session keys for status checking
try:
    from models import SessionKeys
    
    col
