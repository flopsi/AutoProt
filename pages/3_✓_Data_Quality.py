import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
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

# Data selection tabs
data_tab1, data_tab2 = st.tabs(["Protein Data (default)", "Peptide Data"])

with data_tab1:
    # Use protein data
    current_data = st.session_state.protein_data
    data_type = "Protein"
    st.success(f"✓ Analyzing protein data: {current_data.n_proteins:,} proteins")

with data_tab2:
    if peptide_uploaded:
        # Use peptide data
        current_data = st.session_state.peptide_data
        data_type = "Peptide"
        st.success(f"✓ Analyzing peptide data: {current_data.n_rows:,} peptides")
    else:
        st.info("ℹ️ Peptide data not loaded. Upload peptide data to enable this view.")
        st.stop()

# ============================================================
# PREPARE DATA FOR BOXPLOTS
# ============================================================

# Get condition data
condition_mapping = current_data.condition_mapping
quant_data = current_data.quant_data

# Create long-form DataFrame for Altair
intensity_data = []

for col in quant_data.columns:
    condition = condition_mapping.get(col, col)
    condition_letter = condition[0]  # 'A' or 'B'
    replicate_num = condition[1:]    # '1', '2', '3', etc.
    
    for idx, value in quant_data[col].items():
        if pd.notna(value) and value > 0:
            intensity_data.append({
                'Sample': condition,
                'Condition': condition_letter,
                'Replicate': replicate_num,
                'Log10_Intensity': np.log10(value),
                'Original_Column': col
            })

intensity_df = pd.DataFrame(intensity_data)

# ============================================================
# CREATE BOXPLOT CHART
# ============================================================

# Color scale for conditions
color_scale = alt.Scale(
    domain=['A', 'B'],
    range=['#E71316', '#9BD3DD']  # Red for A, Sky for B
)

# Create boxplot
boxplot = alt.Chart(intensity_df).mark_boxplot(
    size=40,
    extent='min-max'
).encode(
    x=alt.X('Sample:N', 
            title='Sample Replicate',
            sort=alt.EncodingSortField(field='Sample', order='ascending'),
            axis=alt.Axis(labelAngle=-45)),
    y=alt.Y('Log10_Intensity:Q', 
            title='Log₁₀ Intensity',
            scale=alt.Scale(zero=False)),
    color=alt.Color('Condition:N', 
                    scale=color_scale,
                    legend=alt.Legend(title='Condition', orient='top', direction='horizontal')),
    tooltip=[
        alt.Tooltip('Sample:N', title='Sample'),
        alt.Tooltip('Condition:N', title='Condition'),
        alt.Tooltip('min(Log10_Intensity):Q', title='Min', format='.2f'),
        alt.Tooltip('q1(Log10_Intensity):Q', title='Q1', format='.2f'),
        alt.Tooltip('median(Log10_Intensity):Q', title='Median', format='.2f'),
        alt.Tooltip('q3(Log10_Intensity):Q', title='Q3', format='.2f'),
        alt.Tooltip('max(Log10_Intensity):Q', title='Max', format='.2f')
    ]
).properties(
    title=f'{data_type} Intensity Distribution by Sample',
    height=450
).configure_view(
    strokeWidth=0
).configure_axis(
    labelFontSize=11,
    titleFontSize=12
).configure_title(
    fontSize=16,
    fontWeight=600,
    anchor='start'
)

# ============================================================
# DISPLAY WITH THEME TABS
# ============================================================

st.markdown("---")
st.markdown("### Intensity Distribution Analysis")

theme_tab1, theme_tab2 = st.tabs(["Streamlit theme (default)", "Altair native theme"])

with theme_tab1:
    st.altair_chart(boxplot, theme="streamlit", use_container_width=True)

with theme_tab2:
    st.altair_chart(boxplot, theme=None, use_container_width=True)

# ============================================================
# SUMMARY STATISTICS
# ============================================================

st.markdown("---")
st.markdown("### Sample Statistics")

# Calculate statistics per sample
stats_data = []
for sample in sorted(intensity_df['Sample'].unique()):
    sample_values = intensity_df[intensity_df['Sample'] == sample]['Log10_Intensity']
    condition = intensity_df[intensity_df['Sample'] == sample]['Condition'].iloc[0]
    
    stats_data.append({
        'Sample': sample,
        'Condition': condition,
        'Count': len(sample_values),
        'Mean': sample_values.mean(),
        'Median': sample_values.median(),
        'Std Dev': sample_values.std(),
        'Min': sample_values.min(),
        'Max': sample_values.max()
    })

stats_df = pd.DataFrame(stats_data)

# Format and display
st.dataframe(
    stats_df.style.format({
        'Mean': '{:.2f}',
        'Median': '{:.2f}',
        'Std Dev': '{:.2f}',
        'Min': '{:.2f}',
        'Max': '{:.2f}'
    }),
    hide_index=True,
    use_container_width=True
)

# ============================================================
# NAVIGATION
# ============================================================

st.markdown("---")
st.markdown("### Navigation")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("← View Results", use_container_width=True):
        st.session_state.upload_stage = 'summary'
        st.switch_page("pages/1_📊_Protein_Upload.py")

with nav_col2:
    if st.button("Upload Peptide Data", use_container_width=True):
        st.switch_page("pages/2_🔬_Peptide_Upload.py")

with nav_col3:
    if st.button("🔄 Start Over", type="primary", use_container_width=True):
        keys_to_delete = list(st.session_state.keys())
        for key in keys_to_delete:
            del st.session_state[key]
        st.switch_page("pages/1_📊_Protein_Upload.py")
