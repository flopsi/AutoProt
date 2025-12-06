"""
pages/1_Data_Upload.py

Complete upload page with explicit data confirmation button
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from helpers.io import read_csv, read_tsv, read_excel
from helpers.core import ProteinData
from helpers.audit import log_event
from helpers.core import get_theme
from helpers.cleaning import drop_proteins_with_invalid_intensities

# ... [keep all existing helper functions: clean_species_name, clean_column_name, 
# detect_numeric_columns, detect_protein_id_column, detect_species_column,
# infer_species_from_protein_name, longest_common_prefix] ...

st.set_page_config(page_title="Data Upload", layout="wide")

with st.sidebar:
    st.title("⚙️ Upload Settings")
    st.info("""**Supported Formats:**
- CSV (.csv)
- TSV (.tsv, .txt)
- Excel (.xlsx)
""")

st.title("📁 Data Upload & Configuration")
st.markdown("Upload your proteomics data file and configure analysis settings.")

# ============================================================================
# STEP 1: UPLOAD FILE
# ============================================================================

st.subheader("1️⃣ Upload File")

uploaded_file = st.file_uploader(
    "Choose a proteomics data file",
    type=['csv', 'tsv', 'txt', 'xlsx'],
    help="Supports CSV, TSV, and Excel formats"
)

if uploaded_file is None:
    st.warning("⚠️ Please upload a data file to continue")
    st.stop()

# ============================================================================
# STEP 2: LOAD DATA
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

# Clean column names
df.columns = [clean_column_name(col) for col in df.columns]

# Auto-trim shared prefix
long_cols = [c for c in df.columns if len(c) > 40]
prefix = longest_common_prefix(long_cols) if long_cols else ""
if prefix and len(prefix) > 20:
    st.info(f"🔧 Removing shared prefix from {len(long_cols)} columns: `{prefix}`")
    new_cols = []
    for c in df.columns:
        if c in long_cols and c.startswith(prefix):
            new_cols.append(c[len(prefix):].lstrip('_').lstrip('.'))
        else:
            new_cols.append(c)
    df.columns = new_cols

# ============================================================================
# STEP 3: DATA CLEANING
# ============================================================================

st.subheader("3️⃣ Data Cleaning")

nnan_before = df.isna().sum().sum()
nzero_before = (df == 0).sum().sum()

st.info("🔄 Replacing NaN and 0 values with 1.0 for log transformation compatibility...")

numeric_cols_detected = detect_numeric_columns(df)
for col in numeric_cols_detected:
    df[col] = df[col].fillna(1.0)
    df.loc[df[col] == 0, col] = 1.0

nnan_after = df.isna().sum().sum()
nzero_after = (df == 0).sum().sum()

c1, c2 = st.columns(2)
with c1:
    st.metric("NaN replaced", nnan_before, delta=f"-{nnan_before}")
with c2:
    st.metric("Zeros replaced", nzero_before, delta=f"-{nzero_before}")

st.success("✅ Data cleaning complete")

# ============================================================================
# STEP 4: SELECT QUANTITATIVE COLUMNS
# ============================================================================

st.subheader("4️⃣ Select Quantitative Columns")
st.markdown("Check the columns you want to use for quantitative analysis.")

df_cols = pd.DataFrame({
    'Select': [col in numeric_cols_detected for col in df.columns],
    'Column': df.columns.tolist(),
    'Type': [str(df[col].dtype) for col in df.columns],
    'Sample': [str(df[col].iloc[0])[:30] if len(df) > 0 else "" for col in df.columns],
})

st.info("💡 Click checkboxes to select/deselect columns")

edited_cols = st.data_editor(
    df_cols,
    column_config={
        "Select": st.column_config.CheckboxColumn(
            "Use",
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
    width="stretch",
    key="column_selector_table"
)

selected_numeric_cols = edited_cols[edited_cols['Select']]['Column'].tolist()

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
rc1, rc2 = st.columns([2, 1])
with rc1:
    should_rename = st.checkbox("Enable column renaming?", value=False, help="Rename quantitative columns for clarity")
with rc2:
    if should_rename:
        st.info("📝 Enter new names below")

if should_rename:
    st.markdown("**Original → New Name**")
    for col in numeric_cols:
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            st.text(col)
        with c2:
            st.write("→")
        with c3:
            new_name = st.text_input(
                "New name",
                value=col,
                label_visibility="collapsed",
                key=f"rename_{col}"
            )
            if new_name != col and new_name.strip():
                rename_dict[col] = new_name

    if rename_dict:
        df = df.rename(columns=rename_dict)
        numeric_cols = [rename_dict.get(col, col) for col in numeric_cols]
        st.success(f"✅ Renamed {len(rename_dict)} columns")
        
        with st.expander("📋 View Column Mapping"):
            mapping_df = pd.DataFrame({
                'Original': list(rename_dict.keys()),
                'New': list(rename_dict.values())
            })
            st.dataframe(mapping_df, width="stretch")

# ============================================================================
# STEP 6: IDENTIFY METADATA COLUMNS
# ============================================================================

st.subheader("6️⃣ Identify Metadata Columns")
st.markdown("Select columns for protein/peptide ID and species (optional).")

non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

mc1, mc2 = st.columns(2)

with mc1:
    protein_id_col = detect_protein_id_column(df)
    if protein_id_col not in non_numeric_cols and len(non_numeric_cols) > 0:
        protein_id_col = non_numeric_cols[0]
    
    if len(non_numeric_cols) > 0:
        protein_id_col = st.selectbox(
            "Protein/Peptide ID column",
            options=non_numeric_cols,
            index=non_numeric_cols.index(protein_id_col) if protein_id_col in non_numeric_cols else 0
        )
    else:
        st.warning("⚠️ No non-numeric columns available for protein ID")
        protein_id_col = df.columns[0]

with mc2:
    species_col_auto = detect_species_column(df)
    species_options = [None] + non_numeric_cols
    default_idx = 0
    if species_col_auto and species_col_auto in non_numeric_cols:
        default_idx = species_options.index(species_col_auto)
    
    species_col = st.selectbox(
        "Species column (optional, leave 'None' to infer from protein names)",
        options=species_options,
        index=default_idx
    )
    
    if species_col == "None":
        species_col = None
    elif species_col:
        df[species_col] = df[species_col].apply(clean_species_name)

st.success("✅ Metadata columns identified")

# ============================================================================
# STEP 6.5: CLEAN DATASET
# ============================================================================

st.subheader("6.5️⃣ Cleaning Dataset")

columns_to_keep = list(numeric_cols)
if protein_id_col:
    columns_to_keep.insert(0, protein_id_col)
if species_col:
    columns_to_keep.append(species_col)

columns_to_remove = [c for c in df.columns if c not in columns_to_keep]
if columns_to_remove:
    st.info(f"🗑️ Removing {len(columns_to_remove)} unused columns")
    df = df[columns_to_keep]

st.success(f"✅ Keeping {len(columns_to_keep)} columns")

# ============================================================================
# STEP 7: DATA PREVIEW
# ============================================================================

st.subheader("7️⃣ Data Preview (First 10 Rows)")

# Ensure unique column names
if df.columns.duplicated().any():
    cols = []
    seen = {}
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            cols.append(c)
    df.columns = cols

df_display = df.head(10).copy()
for col in df_display.columns:
    if pd.api.types.is_float_dtype(df_display[col]):
        df_display[col] = df_display[col].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    else:
        df_display[col] = df_display[col].astype(str).replace("nan", "")

st.dataframe(df_display, width="stretch", height=350)

# ============================================================================
# STEP 8: BASIC STATISTICS
# ============================================================================

st.subheader("8️⃣ Basic Statistics")

# Create species mapping
if species_col:
    species_series = df[species_col]
else:
    inferred_species = df[protein_id_col].apply(infer_species_from_protein_name)
    species_series = inferred_species
    df["INFERRED_SPECIES"] = inferred_species
    species_col = "INFERRED_SPECIES"

species_mapping = dict(zip(df[protein_id_col], species_series))

n_proteins = len(df)
n_samples = len(numeric_cols)
n_conditions = max(1, n_samples // 3)

missing_count = 0
for col in numeric_cols:
    missing_count += df[col].isna().sum()
    missing_count += (df[col] == 1.0).sum()

missing_rate = (missing_count / (n_proteins * n_samples)) * 100

bc1, bc2, bc3, bc4 = st.columns(4)
with bc1:
    st.metric("Total Proteins", f"{n_proteins:,}")
with bc2:
    st.metric("Quantitative Samples", n_samples)
with bc3:
    st.metric("Estimated Conditions", n_conditions)
with bc4:
    st.metric("Missing Values (%)", f"{missing_rate:.1f}")

theme_name = st.session_state.get("theme", "dark")
theme = get_theme(theme_name)

if species_mapping and species_col:
    st.subheader("Species Breakdown by Sample")
    
    chart_data = []
    for sample in numeric_cols:
        species_in_sample = df[df[sample] > 1.0][species_col].value_counts()
        for sp, count in species_in_sample.items():
            chart_data.append({"Sample": sample, "Species": sp, "Count": count})
    
    if chart_data:
        chart_df = pd.DataFrame(chart_data)
        species_color_map = {
            'HUMAN': theme['color_human'],
            'YEAST': theme['color_yeast'],
            'ECOLI': theme['color_ecoli'],
        }
        
        import plotly.express as px
        fig = px.bar(
            chart_df,
            x="Sample",
            y="Count",
            color="Species",
            title="Proteins per Sample by Species",
            labels={"Count": "Number of Proteins"},
            barmode="stack",
            color_discrete_map=species_color_map,
            height=400
        )
        
        fig.update_xaxes(tickangle=-45, showgrid=True, gridcolor=theme['grid'], gridwidth=1)
        fig.update_yaxes(showgrid=True, gridcolor=theme['grid'], gridwidth=1)
        fig.update_layout(
            plot_bgcolor=theme['bg_primary'],
            paper_bgcolor=theme['paper_bg'],
            font=dict(family="Arial", size=14, color=theme['text_primary']),
            title_font=dict(size=16),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, width="stretch")
    
    st.subheader("Total Species Distribution")
    species_totals = species_series.value_counts()
    if len(species_totals) > 0:
        n_species = int(len(species_totals))
        max_cols = min(n_species, 8)
        cols = st.columns(max_cols)
        
        top_species = species_totals.head(max_cols)
        for col_idx, (species, count) in enumerate(top_species.items()):
            with cols[col_idx]:
                st.metric(species, f"{int(count):,}")
        
        if n_species > max_cols:
            st.caption(f"...and {int(n_species - max_cols)} more species")
    else:
        st.info("ℹ️ No species data available")

# ============================================================================
# STEP 9: OPTIONAL CLEANING
# ============================================================================

st.subheader("9️⃣ Final Cleaning (Optional)")

drop_invalid = st.checkbox(
    "Drop proteins with any NaN or 1.00 in selected intensity columns",
    value=False,
    help="If enabled, any protein with at least one missing (NaN or 1.00) value in the selected intensity columns will be removed."
)

rows_dropped = 0
if drop_invalid:
    before_rows = len(df)
    df = drop_proteins_with_invalid_intensities(
        df=df,
        intensity_cols=numeric_cols,
        drop_value=1.0,
        drop_nan=True
    )
    after_rows = len(df)
    rows_dropped = before_rows - after_rows
    st.info(f"🗑️ Dropped {rows_dropped} proteins with at least one NaN or 1.00 intensity. Remaining: {after_rows} proteins.")

# ============================================================================
# STEP 10: CONFIRMATION & STORAGE
# ============================================================================

st.markdown("---")
st.subheader("🎯 Confirm & Store Data")

st.info("""
**Ready to proceed?** 

Click the button below to confirm your data selection and make it available for analysis in the next pages.

Your configuration:
- **File**: `{}`
- **Format**: {}
- **Proteins**: {:,}
- **Samples**: {}
- **Species detected**: {}
""".format(
    uploaded_file.name,
    file_format,
    len(df),
    len(numeric_cols),
    len(species_series.unique())
))

if st.button("✅ Confirm & Proceed to Analysis", type="primary", width="stretch"):
    # Create ProteinData object
    protein_data = ProteinData(
        raw=df,
        numeric_cols=numeric_cols,
        species_col=species_col,
        species_mapping=species_mapping,
        index_col=protein_id_col,
        file_path=uploaded_file.name,
        file_format=file_format
    )
    
    # Store in session state
    st.session_state.protein_data = protein_data
    st.session_state.column_mapping = rename_dict
    
    # Log event
    log_event(
        "Data Upload",
        f"Uploaded {uploaded_file.name}",
        {
            "filename": uploaded_file.name,
            "file_format": file_format,
            "n_proteins": int(n_proteins),
            "n_samples": int(n_samples),
            "missing_rate": float(missing_rate),
            "columns_selected": int(len(numeric_cols)),
            "columns_renamed": int(len(rename_dict)),
            "nan_replaced": int(nnan_before),
            "zeros_replaced": int(nzero_before),
            "proteins_dropped_any_invalid": int(rows_dropped)
        }
    )
    
    st.success("🎉 Data successfully stored! You can now proceed to the next pages.")
    st.balloons()
    
    st.markdown("---")
    st.markdown("""
    ### 📍 Next Steps
    
    Your data is ready for analysis! Use the sidebar to navigate to:
    
    1. **📊 Visual EDA** - Exploratory data analysis and visualization
    2. **📈 Statistical EDA** - Quality control and statistical metrics
    3. **🔬 Analysis** - Differential expression and results
    
    Your configuration is saved and will be available across all pages.
    """)
    
    st.info("💡 **Tip:** You can return to this page anytime to upload a different dataset.")
