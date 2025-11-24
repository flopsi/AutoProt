import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

st.set_page_config(page_title="Proteomics Analysis", layout="wide")

st.title("Proteomics Differential Expression Analysis")

# ============================================================
# STEP 1: UPLOAD DATA
# ============================================================

st.header("Step 1: Upload Data")

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is None:
    st.info("👆 Upload your CSV file to begin")
    st.stop()

# Read data
df = pd.read_csv(uploaded_file)

st.success(f"✅ Loaded {len(df)} proteins")
st.dataframe(df.head(), use_container_width=True)

# ============================================================
# STEP 2: LOG2 TRANSFORM
# ============================================================

st.header("Step 2: Log₂ Transformation")

# Identify metadata and data columns
metadata_cols = ['protein', 'species']
data_cols = [col for col in df.columns if col not in metadata_cols]

st.write(f"Data columns: {', '.join(data_cols)}")

# Log2 transform
df_log2 = df.copy()
for col in data_cols:
    df_log2[col] = np.log2(df[col])

st.success("✅ Data log₂ transformed")
st.dataframe(df_log2.head(), use_container_width=True)

# ============================================================
# STEP 3: BOXPLOT PER SAMPLE
# ============================================================

st.header("Step 3: Boxplot per Sample (After Log₂)")

fig_box = go.Figure()

for col in data_cols:
    fig_box.add_trace(go.Box(
        y=df_log2[col].dropna(),
        name=col,
        boxmean='sd'
    ))

fig_box.update_layout(
    title="Log₂ Intensity Distribution per Sample",
    yaxis_title="Log₂ Intensity",
    xaxis_title="Sample",
    height=500,
    showlegend=False
)

st.plotly_chart(fig_box, use_container_width=True)

# ============================================================
# STEP 4: CALCULATE LOG2 FC PER SPECIES
# ============================================================

st.header("Step 4: Calculate Log₂ Fold-Changes")

# Identify A and B samples
a_samples = [col for col in data_cols if col.startswith('A')]
b_samples = [col for col in data_cols if col.startswith('B')]

st.write(f"Condition A: {', '.join(a_samples)}")
st.write(f"Condition B: {', '.join(b_samples)}")

# Calculate fold-changes for each species
fc_results = []

for species in df['species'].unique():
    species_data = df_log2[df_log2['species'] == species]
    
    st.write(f"\n**{species.capitalize()}:** {len(species_data)} proteins")
    
    # For each protein in this species
    for idx, row in species_data.iterrows():
        protein = row['protein']
        
        # Calculate FC for each A vs B pair
        for a_col, b_col in zip(a_samples, b_samples):
            a_val = row[a_col]
            b_val = row[b_col]
            
            if pd.notna(a_val) and pd.notna(b_val):
                log2fc = a_val - b_val
                
                fc_results.append({
                    'protein': protein,
                    'species': species,
                    'comparison': f"{a_col} - {b_col}",
                    'log2FC': log2fc
                })

# Create FC dataframe
fc_df = pd.DataFrame(fc_results)

st.success(f"✅ Calculated {len(fc_df)} fold-change values")
st.dataframe(fc_df.head(20), use_container_width=True)

# ============================================================
# STEP 5: DENSITY PLOT PER SPECIES
# ============================================================

st.header("Step 5: Density Plot per Species")

# Species colors
species_colors = {
    'human': '#199d76',
    'yeast': '#d85f02',
    'ecoli': '#7570b2'
}

# Expected fold-changes
expected_fc = {
    'human': 0,
    'yeast': 1,
    'ecoli': -2
}

fig_density = go.Figure()

for species in fc_df['species'].unique():
    species_fc = fc_df[fc_df['species'] == species]['log2FC'].values
    
    if len(species_fc) > 10:
        # Calculate KDE
        kde = gaussian_kde(species_fc, bw_method='scott')
        x_range = np.linspace(-3, 3, 300)
        density = kde(x_range)
        
        # Add density curve
        fig_density.add_trace(go.Scatter(
            x=x_range,
            y=density,
            mode='lines',
            name=species.capitalize(),
            line=dict(color=species_colors.get(species, '#7B7B7B'), width=0),
            fill='tozeroy',
            fillcolor=species_colors.get(species, '#7B7B7B'),
            opacity=0.6,
            hovertemplate=f'<b>{species.capitalize()}</b><br>Log₂FC: %{{x:.2f}}<br>Density: %{{y:.2f}}<extra></extra>'
        ))
        
        # Add expected FC line
        if species in expected_fc:
            fig_density.add_vline(
                x=expected_fc[species],
                line_dash="dash",
                line_color=species_colors.get(species, '#7B7B7B'),
                line_width=2,
                opacity=0.8
            )

fig_density.update_layout(
    title="Log₂ Fold-Change Distribution by Species",
    xaxis_title="Log₂(A/B)",
    yaxis_title="Density",
    height=500,
    xaxis=dict(range=[-3, 3], zeroline=True, zerolinecolor='rgba(0,0,0,0.2)'),
    legend=dict(x=0.8, y=0.95)
)

st.plotly_chart(fig_density, use_container_width=True)

# Summary statistics
st.subheader("Summary Statistics")

summary_data = []
for species in fc_df['species'].unique():
    species_fc = fc_df[fc_df['species'] == species]['log2FC']
    
    if species in expected_fc:
        expected = expected_fc[species]
        measured = species_fc.median()
        error = abs(measured - expected)
    else:
        expected = None
        measured = species_fc.median()
        error = None
    
    summary_data.append({
        'Species': species.capitalize(),
        'N': len(species_fc),
        'Median Log₂FC': f"{measured:.2f}",
        'Expected': f"{expected:.1f}" if expected is not None else "N/A",
        'Error': f"{error:.2f}" if error is not None else "N/A"
    })

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True)
