import streamlit as st
import pandas as pd
from config import (
    get_numeric_columns,
    get_metadata_columns,
    get_default_species_mapping_cols,
    get_default_group_col,
    trim_column_names,
    THERMO_COLORS
)
from models import (
    DataLevel,
    Condition,
    ColumnMetadata,
    DatasetConfig,
    ProteomicsDataset,
    SessionKeys
)

# ========================================================================
# PAGE CONFIGURATION
# ========================================================================
st.set_page_config(
    page_title="Protein Upload | Proteomics",
    page_icon="🔬",
    layout="wide"
)

st.markdown(f"""
<style>
:root {{
  --primary-red: {THERMO_COLORS['PRIMARY_RED']};
  --primary-gray: {THERMO_COLORS['PRIMARY_GRAY']};
  --dark-red: {THERMO_COLORS['DARK_RED']};
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg-primary: #0e1117; --bg-secondary: #262730; --text-primary: #fafafa; --border-color: #3d4149; }}
}}
@media (prefers-color-scheme: light) {{
  :root {{ --bg-primary: #f8f9fa; --bg-secondary: #ffffff; --text-primary: #54585A; --border-color: #E2E3E4; }}
}}
.step-indicator {{
  display: inline-block; width: 32px; height: 32px; background: var(--primary-red);
  color: white; border-radius: 50%; text-align: center; line-height: 32px;
  font-weight: 600; margin-right: 10px;
}}
.step-header {{
  display: flex; align-items: center; margin: 25px 0 15px 0;
  font-size: 16px; font-weight: 600;
}}
</style>
""", unsafe_allow_html=True)

# ========================================================================
# HEADER
# ========================================================================
st.title("🔬 Protein-Level Data Upload")
st.caption("Import and configure your protein-level quantification data")

# ========================================================================
# STEP 1: FILE UPLOAD
# ========================================================================
st.markdown("""
<div class="step-header">
  <span class="step-indicator">1</span>
  <span>Upload Protein-Level File</span>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose protein quantification file",
    type=['csv', 'tsv', 'txt'],
    key="protein_file_upload",
    help="Supported formats: CSV, TSV, TXT"
)

if not uploaded_file:
    st.info("📁 Please upload a protein-level quantification file to begin.")
    st.stop()

sep = '\t' if uploaded_file.name.endswith(('.tsv', '.txt')) else ','
preview_df = pd.read_csv(uploaded_file, sep=sep, nrows=100)

st.success(f"✓ File loaded: **{uploaded_file.name}**")

col1, col2 = st.columns(2)
with col1:
    st.metric("Preview Rows", f"{len(preview_df):,}")
with col2:
    st.metric("Total Columns", len(preview_df.columns))

num_cols = get_numeric_columns(preview_df)
meta_cols = get_metadata_columns(preview_df, num_cols)

trimmed_quant = trim_column_names(num_cols)
trimmed_meta = trim_column_names(meta_cols)

st.markdown("---")

# ========================================================================
# STEP 2: COLUMN SELECTION AND ANNOTATION
# ========================================================================
st.markdown("""
<div class="step-header">
  <span class="step-indicator">2</span>
  <span>Select and Annotate Columns</span>
</div>
""", unsafe_allow_html=True)

st.caption(f"📊 Detected **{len(num_cols)}** quantitative and **{len(meta_cols)}** metadata columns")

default_species_cols = get_default_species_mapping_cols(preview_df)
default_group_cols = get_default_group_col(preview_df, meta_cols)

annotation_data = []

# Metadata columns first
for col in meta_cols:
    is_protein_group = col in (default_group_cols or [])
    is_species = col in (default_species_cols or [])
    annotation_data.append({
        'Include': True,
        'Trimmed Name': trimmed_meta[col],
        'Original Name': col,
        'Protein Group': is_protein_group,
        'Species Mapping': is_species,
        'Control': False,
        'Treatment': False
    })

# Quantitative columns
for idx, col in enumerate(num_cols):
    is_control = idx < len(num_cols) // 2
    is_treatment = not is_control
    annotation_data.append({
        'Include': True,
        'Trimmed Name': trimmed_quant[col],
        'Original Name': col,
        'Protein Group': False,
        'Species Mapping': False,
        'Control': is_control,
        'Treatment': is_treatment
    })

col_order = ['Include', 'Trimmed Name', 'Protein Group', 'Species Mapping', 'Control', 'Treatment', 'Original Name']
annotation_df = pd.DataFrame(annotation_data)[col_order]

st.markdown("**Select columns and assign roles:**")
st.caption("ℹ️ Uncheck 'Include' to drop columns. Check boxes to assign roles.")

column_config = {
    'Include': st.column_config.CheckboxColumn('Include', help='Uncheck to exclude', default=True),
    'Trimmed Name': st.column_config.TextColumn('Trimmed Name', help='Cleaned name', width='medium'),
    'Protein Group': st.column_config.CheckboxColumn('Protein Group', help='Use for grouping', default=False),
    'Species Mapping': st.column_config.CheckboxColumn('Species Mapping', help='Use for species', default=False),
    'Control': st.column_config.CheckboxColumn('Control', help='Control sample', default=False),
    'Treatment': st.column_config.CheckboxColumn('Treatment', help='Treatment sample', default=False),
    'Original Name': st.column_config.TextColumn('Original Name', help='Original column name', width='small')
}

edited_df = st.data_editor(
    annotation_df,
    hide_index=True,
    width='stretch',
    column_config=column_config,
    key="protein_col_editor"
)

included_rows = edited_df[edited_df['Include']]

if len(included_rows) == 0:
    st.error("❌ You must include at least one column")
    st.stop()

protein_group_cols = included_rows[included_rows['Protein Group']]['Original Name'].tolist()
species_cols = included_rows[included_rows['Species Mapping']]['Original Name'].tolist()
control_cols = included_rows[included_rows['Control']]['Original Name'].tolist()
treatment_cols = included_rows[included_rows['Treatment']]['Original Name'].tolist()

warnings = []
if len(protein_group_cols) == 0:
    warnings.append("⚠️ No Protein Group column selected")
if len(species_cols) == 0:
    warnings.append("⚠️ No Species Mapping column selected")
if len(control_cols) == 0 and len(treatment_cols) == 0:
    warnings.append("⚠️ No Control or Treatment samples selected")

for warn in warnings:
    st.warning(warn)

dropped_count = len(annotation_df) - len(included_rows)
if dropped_count > 0:
    st.info(f"ℹ️ {dropped_count} column(s) will be excluded")

st.success(f"✓ {len(included_rows)} columns selected | {len(control_cols)} Control | {len(treatment_cols)} Treatment")

st.markdown("---")

# ========================================================================
# STEP 3: LOAD FULL DATASET
# ========================================================================
st.markdown("""
<div class="step-header">
  <span class="step-indicator">3</span>
  <span>Load Full Dataset</span>
</div>
""", unsafe_allow_html=True)

if st.button("📥 Load Full Dataset", type="primary", use_container_width=True):
    if warnings:
        st.error("❌ Please fix warnings before loading")
        st.stop()

    with st.spinner("Loading full dataset..."):
        uploaded_file.seek(0)
        cols_to_read = included_rows['Original Name'].tolist()
        full_df = pd.read_csv(uploaded_file, sep=sep, usecols=cols_to_read)
        rename_map = dict(zip(included_rows['Original Name'], included_rows['Trimmed Name']))
        full_df = full_df.rename(columns=rename_map)

        # Handle duplicates
        if not full_df.columns.is_unique:
            st.warning("⚠️ Duplicate column names detected. Adding numeric suffixes.")
            cols = pd.Series(full_df.columns)
            for dup in cols[cols.duplicated()].unique():
                indices = cols[cols == dup].index.values.tolist()
                cols.iloc[indices] = [f"{dup}_{i+1}" if i > 0 else dup for i in range(len(indices))]
            full_df.columns = cols.tolist()
            rename_map = dict(zip(included_rows['Original Name'], full_df.columns))

        # Build metadata objects
        columns_metadata = []
        for _, row in included_rows.iterrows():
            trimmed = rename_map.get(row['Original Name'], row['Trimmed Name'])
            is_quant = row['Original Name'] in num_cols
            cond = None
            if row['Control']:
                cond = Condition.CONTROL
            elif row['Treatment']:
                cond = Condition.TREATMENT
            col_meta = ColumnMetadata(
                original_name=row['Original Name'],
                trimmed_name=trimmed,
                is_quantitative=is_quant,
                is_protein_group=row['Protein Group'],
                is_species_mapping=row['Species Mapping'],
                condition=cond
            )
            columns_metadata.append(col_meta)

        config = DatasetConfig(
            level=DataLevel.PROTEIN,
            file_name=uploaded_file.name,
            total_rows=len(full_df),
            columns=columns_metadata,
            protein_group_col=rename_map.get(protein_group_cols[0]) if protein_group_cols else None,
            species_col=rename_map.get(species_cols[0]) if species_cols else None
        )
        dataset = ProteomicsDataset(config=config, data=full_df)
        st.session_state[SessionKeys.PROTEIN_DATASET.value] = dataset

        st.success(f"✅ **Dataset loaded successfully!** {len(full_df):,} proteins")
        st.balloons()

if SessionKeys.PROTEIN_DATASET.value in st.session_state:
    dataset = st.session_state[SessionKeys.PROTEIN_DATASET.value]
    st.markdown("---")
    st.subheader("✅ Dataset Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Proteins", f"{dataset.n_proteins:,}")
    with col2:
        st.metric("Samples", dataset.n_samples)
    with col3:
        st.metric("Metadata Columns", len(dataset.config.metadata_columns))

    st.markdown("**Column Assignments:**")
    st.write(f"🔹 **Protein Group:** {dataset.config.protein_group_col}")
    st.write(f"🔹 **Species Mapping:** {dataset.config.species_col}")

    st.markdown("**Sample Conditions:**")
    for col_meta in dataset.config.quant_columns:
        if col_meta.condition:
            emoji = "🟦" if col_meta.condition == Condition.CONTROL else "🟥"
            st.write(f"{emoji} **{col_meta.trimmed_name}** → {col_meta.condition.value}")

    with st.expander("📋 Preview Data"):
        st.dataframe(dataset.data.head(10), width='stretch')

    st.info("✨ **Next Step:** Analyze your data or upload peptides for deeper analysis.")
