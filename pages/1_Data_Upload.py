"""
pages/1_Data_Upload.py
Complete upload page with checkbox table selection
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from helpers.file_io import read_csv, read_tsv, read_excel
from helpers.dataclasses import ProteinData
from helpers.audit import log_event
import altair as alt

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_species_name(name: str) -> str:
    """Remove leading/trailing underscores from species names."""
    if pd.isna(name):
        return name
    return str(name).strip().strip('_').upper()


def clean_column_name(name: str) -> str:
    """Clean column names: remove special chars, standardize spacing."""
    name = re.sub(r'[\(\)\[\]\{\}]', '', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.replace(' ', '_')
    name = name.strip('_')
    return name


def detect_numeric_columns(df: pd.DataFrame) -> list:
    """Detect numeric columns."""
    return df.select_dtypes(include=['number']).columns.tolist()


def detect_protein_id_column(df: pd.DataFrame) -> str:
    """Detect protein/peptide ID column."""
    patterns = ['protein', 'gene', 'id', 'accession', 'sequence', 'uniprot']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in patterns):
            if df[col].dtype not in ['int64', 'float64']:
                return col
    
    # Fallback: first non-numeric column
    for col in df.columns:
        if df[col].dtype == 'object':
            return col
    
    return df.columns[0]


def detect_species_column(df: pd.DataFrame) -> str:
    """Detect species column."""
    patterns = ['human', 'yeast', 'ecoli', 'species', 'organism', 'homo', 'saccharomyces']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in patterns):
            return col
        
        # Check values
        unique_vals = df[col].dropna().astype(str).str.lower().unique()[:10]
        if any(any(p in val for p in patterns) for val in unique_vals):
            return col
    
    return None


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Data Upload", layout="wide")

with st.sidebar:
    st.title("📊 Upload Settings")
    theme = st.session_state.get("theme", "light")
    st.info("""
    **Supported Formats:**
    - CSV (.csv)
    - TSV (.tsv, .txt)
    - Excel (.xlsx)
    """)

# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("📊 Data Upload & Configuration")

st.markdown("""
Upload your proteomics data file and configure analysis settings.
""")

# ============================================================================
# STEP 1: FILE UPLOAD
# ============================================================================

st.subheader("1️⃣ Upload File")

uploaded_file = st.file_uploader(
    "Choose a proteomics data file",
    type=["csv", "tsv", "txt", "xlsx"],
    help="Supports CSV, TSV, and Excel formats"
)

if uploaded_file is None:
    st.warning("⚠️ Please upload a data file to continue")
    st.stop()

# ============================================================================
# STEP 2: READ FILE
# ============================================================================

st.subheader("2️⃣ Loading Data...")

try:
    filename = uploaded_file.name.lower()
    
    if filename.endswith('.xlsx'):
        df = read_excel(uploaded_file)
        file_format = "Excel"
    elif filename.endswith('.tsv') or filename.endswith('.txt'):
        df = read_tsv(uploaded_file)
        file_format = "TSV"
    else:
        df = read_csv(uploaded_file)
        file_format = "CSV"
    
    st.success(f"✅ Loaded {file_format} file: {len(df):,} rows × {len(df.columns)} columns")
    
except Exception as e:
    st.error(f"❌ Error reading file: {str(e)}")
    st.stop()

# Clean column names immediately
df.columns = [clean_column_name(col) for col in df.columns]

# ============================================================================
# STEP 3: FILL NaN AND 0 WITH 1
# ============================================================================

st.subheader("3️⃣ Data Cleaning")

# Count before
n_nan_before = df.isna().sum().sum()
n_zero_before = (df == 0).sum().sum()

st.info(f"Replacing NaN and 0 values with 1.0 for log transformation compatibility...")

# Detect numeric columns first
numeric_cols_detected = detect_numeric_columns(df)

# Replace NaN and 0 with 1 in numeric columns
for col in numeric_cols_detected:
    df[col] = df[col].fillna(1.0)
    df.loc[df[col] == 0, col] = 1.0

n_nan_after = df.isna().sum().sum()
n_zero_after = (df == 0).sum().sum()

col1, col2 = st.columns(2)
with col1:
    st.metric("NaN replaced", n_nan_before, delta=f"-{n_nan_before}")
with col2:
    st.metric("Zeros replaced", n_zero_before, delta=f"-{n_zero_before}")

st.success("✅ Data cleaning complete")

# ============================================================================
# STEP 4: SELECT QUANTITATIVE COLUMNS (CHECKBOX TABLE)
# ============================================================================

st.subheader("4️⃣ Select Quantitative Columns")

st.markdown("""
**Check the columns you want to use for quantitative analysis.**
Columns must be numeric (measurements, intensities, abundances).
""")

# Create dataframe for checkbox table
df_cols = pd.DataFrame({
    "Select": [col in numeric_cols_detected for col in df.columns],
    "Column": df.columns.tolist(),
    "Type": [str(df[col].dtype) for col in df.columns],
    "Sample": [str(df[col].iloc[0])[:30] if len(df) > 0 else "" for col in df.columns]
})

st.info("💡 **Click checkboxes to select/deselect columns**")

# Interactive checkbox table
edited_cols = st.data_editor(
    df_cols,
    column_config={
        "Select": st.column_config.CheckboxColumn(
            "✓ Use",
            help="Check to include in analysis",
            default=False,
            width="small"
        ),
        "Column": st.column_config.TextColumn(
            "Column Name",
            width="large",
            disabled=True
        ),
        "Type": st.column_config.TextColumn(
            "Data Type",
            width="small",
            disabled=True
        ),
        "Sample": st.column_config.TextColumn(
            "Sample Value",
            width="medium",
            disabled=True
        ),
    },
    hide_index=True,
    use_container_width=True,
    key="column_selector_table"
)

# Get selected columns
selected_numeric_cols = edited_cols[edited_cols["Select"]]["Column"].tolist()

# Validate
if len(selected_numeric_cols) < 4:
    st.warning(f"⚠️ Need at least 4 quantitative columns for analysis. You selected {len(selected_numeric_cols)}.")
    st.stop()

numeric_cols = selected_numeric_cols
st.success(f"✅ Selected {len(numeric_cols)} quantitative columns")

# ============================================================================
# STEP 5: RENAME COLUMNS (OPTIONAL)
# ============================================================================

st.subheader("5️⃣ Rename Columns (Optional)")

rename_dict = {}

rename_cols = st.columns(2)
with rename_cols[0]:
    should_rename = st.checkbox(
        "Enable column renaming?",
        value=False,
        help="Rename quantitative columns for clarity"
    )

with rename_cols[1]:
    if should_rename:
        st.info("Enter new names below")

if should_rename:
    st.markdown("**Original → New Name**")
    
    for col in numeric_cols:
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.text(col)
        
        with col2:
            st.write("→")
        
        with col3:
            new_name = st.text_input(
                "New name",
                value=col,
                label_visibility="collapsed",
                key=f"rename_{col}"
            )
            if new_name != col and new_name.strip():
                rename_dict[col] = new_name
    
    # Apply renaming
    if rename_dict:
        df = df.rename(columns=rename_dict)
        numeric_cols = [rename_dict.get(col, col) for col in numeric_cols]
        st.success(f"✅ Renamed {len(rename_dict)} columns")
        
        # Show mapping
        with st.expander("📋 View Column Mapping"):
            mapping_df = pd.DataFrame({
                "Original": list(rename_dict.keys()),
                "New": list(rename_dict.values())
            })
            st.dataframe(mapping_df, use_container_width=True)

# ============================================================================
# STEP 6: IDENTIFY METADATA COLUMNS
# ============================================================================

st.subheader("6️⃣ Identify Metadata Columns")

st.markdown("Select columns for protein/peptide ID and species (optional).")

non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

col1, col2 = st.columns(2)

with col1:
    protein_id_col = detect_protein_id_column(df)
    if protein_id_col not in non_numeric_cols and len(non_numeric_cols) > 0:
        protein_id_col = non_numeric_cols[0]
    
    if len(non_numeric_cols) > 0:
        protein_id_col = st.selectbox(
            "🔍 Protein/Peptide ID column",
            options=non_numeric_cols,
            index=non_numeric_cols.index(protein_id_col) if protein_id_col in non_numeric_cols else 0,
        )
    else:
        st.warning("⚠️ No non-numeric columns available for protein ID")
        protein_id_col = df.columns[0]

with col2:
    species_col = detect_species_column(df)
    species_options = ["(None)"] + non_numeric_cols
    
    default_idx = 0
    if species_col and species_col in non_numeric_cols:
        default_idx = species_options.index(species_col)
    
    species_col = st.selectbox(
        "🧬 Species column",
        options=species_options,
        index=default_idx,
    )
    
    if species_col == "(None)":
        species_col = None
    elif species_col:
        # Clean species names
        df[species_col] = df[species_col].apply(clean_species_name)

st.success("✅ Metadata columns identified")

# ============================================================================
# STEP 6.5: DROP UNUSED COLUMNS
# ============================================================================

st.subheader("6️⃣.5 Cleaning Dataset")

# Keep only: quantitative + protein ID + species
columns_to_keep = list(numeric_cols)
if protein_id_col: 
    columns_to_keep.insert(0, protein_id_col)
if species_col: 
    columns_to_keep.append(species_col)

columns_to_remove = [c for c in df.columns if c not in columns_to_keep]

if columns_to_remove:
    st.info(f"ℹ️ Removing {len(columns_to_remove)} unused columns")

df = df[columns_to_keep]
st.success(f"✅ Keeping {len(columns_to_keep)} columns")

# ============================================================================
# STEP 7: PREVIEW DATA (10 ROWS)
# ============================================================================

st.subheader("7️⃣ Data Preview (First 10 Rows)")

# Format for display - replace remaining NaN with empty string
df_display = df.head(10).copy()
for col in df_display.columns:
    if df_display[col].dtype in ['float64', 'float32']:
        df_display[col] = df_display[col].apply(
            lambda x: '' if pd.isna(x) else f"{x:.2f}"
        )
    else:
        df_display[col] = df_display[col].astype(str).replace('nan', '')

st.dataframe(df_display, use_container_width=True, height=350)

# ============================================================================
# STEP 8: BASIC STATISTICS
# ============================================================================

st.subheader("8️⃣ Basic Statistics")

# Create species mapping
species_mapping = {}
if species_col:
    species_mapping = dict(zip(df[protein_id_col], df[species_col]))

# Calculate stats
n_proteins = len(df)
n_samples = len(numeric_cols)
n_conditions = max(1, n_samples // 3)  # Estimate
# Count both NaN and 1.0 (imputed) as missing
missing_count = 0
for col in numeric_cols:
    missing_count += df[col].isna().sum()
    missing_count += (df[col] == 1.0).sum()

missing_rate = (missing_count / (n_proteins * n_samples) * 100)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Proteins", f"{n_proteins:,}")

with col2:
    st.metric("Quantitative Samples", n_samples)

with col3:
    st.metric("Estimated Conditions", n_conditions)

with col4:
    st.metric("Missing Values %", f"{missing_rate:.1f}%")



import altair as alt

if species_mapping and species_col:
    st.subheader("Species Breakdown by Sample")
    
    # Prepare data
    chart_data = []
    for sample in numeric_cols:
        species_in_sample = df[df[sample] > 1.0][species_col].value_counts()
        for species, count in species_in_sample.items():
            chart_data.append({
                'Sample': sample,
                'Species': species,
                'Count': count
            })
    
    chart_df = pd.DataFrame(chart_data)
    
# Species Breakdown by Sample (Stacked Bar Chart)
if species_mapping and species_col:
    st.subheader("Species Breakdown by Sample")
    
    # Import theme from constants
    from helpers.constants import get_theme
    
    # Get current theme
    theme_name = st.session_state.get("theme", "light")
    theme = get_theme(theme_name)
    
    # Prepare data: count proteins per sample per species
    chart_data = []
    for sample in numeric_cols:
        species_in_sample = df[df[sample] > 1.0][species_col].value_counts()
        for species, count in species_in_sample.items():
            chart_data.append({
                'Sample': sample,
                'Species': species,
                'Count': count
            })
    
    if chart_data:
        chart_df = pd.DataFrame(chart_data)
        
        # Create species color mapping from theme
        species_color_map = {
            'HUMAN': theme['color_human'],
            'YEAST': theme['color_yeast'],
            'ECOLI': theme['color_ecoli'],
        }
        
        # Plotly stacked bar with theme colors
        import plotly.express as px
        
        fig = px.bar(
            chart_df,
            x='Sample',
            y='Count',
            color='Species',
            title='Proteins per Sample by Species',
            labels={'Count': 'Number of Proteins'},
            barmode='stack',
            color_discrete_map=species_color_map,
            height=400
        )
        
        # Apply theme styling
        fig.update_xaxes(
            tickangle=-45,
            showgrid=True,
            gridcolor=theme['grid'],
            gridwidth=1
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridcolor=theme['grid'],
            gridwidth=1
        )
        
        fig.update_layout(
            plot_bgcolor=theme['bg_primary'],
            paper_bgcolor=theme['paper_bg'],
            font=dict(
                family="Arial",
                size=14,
                color=theme['text_primary']
            ),
            title_font=dict(size=16),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Total species counts with metrics
    st.subheader("Total Species Distribution")
    species_totals = df[species_col].value_counts()
    
    cols = st.columns(len(species_totals))
    for col, (species, count) in zip(cols, species_totals.items()):
        with col:
            st.metric(species, f"{count:,}")


# ============================================================================
# STEP 9: CREATE PROTEIN DATA OBJECT & STORE
# ============================================================================

st.subheader("9️⃣ Finalizing...")

# Create ProteinData object
protein_data = ProteinData(
    raw=df,
    numeric_cols=numeric_cols,
    species_col=species_col,
    species_mapping=species_mapping,
    index_col=protein_id_col,
    file_path=uploaded_file.name,
    file_format=file_format,
)

# Store in session state
st.session_state.protein_data = protein_data
st.session_state.column_mapping = rename_dict

# Audit logging
log_event(
    "Data Upload",
    f"Uploaded {uploaded_file.name}",
    {
        "filename": uploaded_file.name,
        "file_format": file_format,
        "n_proteins": n_proteins,
        "n_samples": n_samples,
        "missing_rate": float(missing_rate),
        "columns_selected": len(numeric_cols),
        "columns_renamed": len(rename_dict),
        "nan_replaced": n_nan_before,
        "zeros_replaced": n_zero_before,
    }
)

st.success("✅ Data loaded and configured successfully!")

# ============================================================================
# STEP 10: NEXT STEPS
# ============================================================================

st.markdown("---")

st.markdown("""
### ✨ Next Steps

Your data is ready for analysis! Use the sidebar to navigate to:

1. **📊 EDA** - Exploratory data analysis and visualization
2. **🔬 Preprocessing** - Data transformation and normalization
3. **🔍 Filtering** - Quality control and filtering
4. **📈 Analysis** - Statistical analysis and results

Your configuration is saved and will be available across all pages.
""")

st.info("💡 **Tip:** You can return to this page anytime to upload a different dataset.")
