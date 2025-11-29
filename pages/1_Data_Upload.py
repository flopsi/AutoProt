import streamlit as st
import pandas as pd
import re

COLORS = {
    "red": "#E71316",
    "dark_red": "#A6192E",
    "gray": "#54585A",
    "light_gray": "#E2E3E4",
    "white": "#FFFFFF",
    "navy": "#262262",
    "orange": "#EA7600",
    "dark_surface": "#2D2D2D",
    "dark_text": "#E2E3E4",
}

st.set_page_config(
    page_title="Data Upload | Thermo Fisher Scientific",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
    body, .stMarkdown, .stText {{
        font-family: Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    .stButton > button {{
        background-color: {COLORS['red']};
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
        font-weight: 500;
    }}
    .stButton > button:hover {{
        background-color: {COLORS['dark_red']};
    }}
    .header-banner {{
        background: linear-gradient(90deg, {COLORS['red']} 0%, {COLORS['dark_red']} 100%);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 30px;
    }}
    .header-banner h1 {{
        color: white;
        margin: 0;
        font-size: 28pt;
    }}
    .header-banner p {{
        color: white;
        margin: 5px 0 0 0;
        opacity: 0.9;
    }}
    .status-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }}
    .badge-protein {{
        background-color: {COLORS['navy']};
        color: white;
    }}
    .badge-peptide {{
        background-color: {COLORS['orange']};
        color: white;
    }}
</style>
""", unsafe_allow_html=True)


def extract_condition_replicate(col_name: str) -> str:
    match = re.search(r'([A-Z]\d{2}-[A-Z]\d{2})_(\d{2})', col_name)
    if match:
        condition, rep = match.groups()
        return f"{condition}_{int(rep)}"
    return col_name


def detect_and_rename_numeric_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rename_map = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            rename_map[col] = extract_condition_replicate(col)
    return df.rename(columns=rename_map), rename_map


def filter_by_species(df: pd.DataFrame, col: str, species_tags: list[str]) -> pd.DataFrame:
    if not species_tags or not col:
        return df
    pattern = '|'.join(re.escape(tag) for tag in species_tags)
    return df[df[col].astype(str).str.contains(pattern, case=False, na=False)]


def render_header(title: str, subtitle: str):
    st.markdown(f"""
    <div class="header-banner">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


defaults = {
    "uploaded_df": None,
    "processed_df": None,
    "column_names": {},
    "protein_data": None,
    "peptide_data": None,
    "protein_index_col": None,
    "peptide_index_col": None,
    "protein_missing_mask": None,
    "peptide_missing_mask": None,
    "upload_key": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

col_nav1, col_nav2, _ = st.columns([1, 1, 4])
with col_nav1:
    if st.button("← Home"):
        st.switch_page("app.py")
with col_nav2:
    if st.button("EDA →"):
        st.switch_page("pages/2_EDA.py")

render_header("Data upload", "Import and configure mass spectrometry output matrices")

uploaded_file = st.file_uploader(
    "Upload proteomics CSV",
    type=["csv"],
    key=f"uploader_{st.session_state.upload_key}"
)

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    renamed_df, rename_map = detect_and_rename_numeric_cols(raw_df)

    if st.session_state.uploaded_df is None or not renamed_df.columns.equals(st.session_state.uploaded_df.columns):
        st.session_state.uploaded_df = renamed_df.copy()
        st.session_state.column_names = {c: c for c in renamed_df.columns}

    st.success(f"✓ Loaded {len(renamed_df):,} rows, {len(renamed_df.columns)} columns")

    non_numeric_cols = [c for c in renamed_df.columns if not pd.api.types.is_numeric_dtype(renamed_df[c])]
    numeric_cols = [c for c in renamed_df.columns if pd.api.types.is_numeric_dtype(renamed_df[c])]

    with st.expander("Column configuration", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Protein group column**")
            protein_group_col = st.selectbox(
                "Select protein group IDs",
                options=["None"] + non_numeric_cols,
                index=1 if non_numeric_cols else 0,
                label_visibility="collapsed"
            )
            protein_group_col = None if protein_group_col == "None" else protein_group_col

        with col2:
            st.markdown("**Sequence column (peptide data)**")
            sequence_col = st.selectbox(
                "Select peptide sequences",
                options=["None"] + non_numeric_cols,
                index=0,
                label_visibility="collapsed"
            )
            sequence_col = None if sequence_col == "None" else sequence_col
            is_peptide_data = sequence_col is not None

        with col3:
            st.markdown("**Data level**")
            if is_peptide_data:
                st.markdown('<span class="status-badge badge-peptide">Peptide</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge badge-protein">Protein</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_sp1, col_sp2 = st.columns([3, 1])
        with col_sp1:
            species_tags = st.multiselect(
                "Species filter tags",
                options=["_HUMAN", "_YEAST", "_ECOLI", "_MOUSE"],
                default=["_HUMAN"],
            )
        with col_sp2:
            custom_species = st.text_input("Custom tag")
            if custom_species and custom_species not in species_tags:
                species_tags.append(custom_species)

    with st.expander("Edit column names", expanded=False):
        edited_names = {}
        cols_per_row = 4
        for i in range(0, len(numeric_cols), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col in enumerate(numeric_cols[i:i+cols_per_row]):
                with row_cols[j]:
                    edited_names[col] = st.text_input(
                        col,
                        value=st.session_state.column_names.get(col, col),
                        key=f"edit_{col}"
                    )

        if st.button("Apply renames"):
            st.session_state.column_names.update(edited_names)
            st.rerun()

    final_rename = {k: v for k, v in st.session_state.column_names.items() if k != v}
    processed_df = renamed_df.rename(columns=final_rename)
    numeric_final = [st.session_state.column_names.get(c, c) for c in numeric_cols]

    filter_col = None
    for potential in ["PG.ProteinNames", "ProteinNames", "Protein.Names"]:
        if potential in processed_df.columns:
            filter_col = potential
            break

    if species_tags and filter_col:
        processed_df = filter_by_species(processed_df, filter_col, species_tags)
        st.info(f"Filtered to {len(processed_df):,} rows matching: {', '.join(species_tags)}")

    st.session_state.processed_df = processed_df

    st.markdown("### Data preview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(processed_df):,}")
    c2.metric("Numeric columns", len(numeric_final))
    c3.metric("Conditions", len(set(c.rsplit('_', 1)[0] for c in numeric_final if '_' in c)))
    c4.metric("Data level", "Peptide" if is_peptide_data else "Protein")

    st.dataframe(processed_df.head(10), use_container_width=True, height=300)

    st.markdown("---")
    data_type = "peptide" if is_peptide_data else "protein"
    existing_key = f"{data_type}_data"
    index_key = f"{data_type}_index_col"

    if st.session_state[existing_key] is not None:
        st.warning(f"{data_type.capitalize()} data already cached. Confirming will overwrite.")

    st.markdown(f"### Cache as {data_type} data?")
    st.caption(f"Missing values (NaN, 0) will be replaced with 1. Protein group column (`{protein_group_col}`) preserved for mapping.")

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        if st.button("Confirm & cache", type="primary"):
            cache_df = processed_df.copy()
            missing_mask = cache_df[numeric_final].isna() | (cache_df[numeric_final] == 0)
            cache_df[numeric_final] = cache_df[numeric_final].fillna(1).replace(0, 1)

            st.session_state[existing_key] = cache_df
            st.session_state[index_key] = protein_group_col
            st.session_state[f"{data_type}_missing_mask"] = missing_mask

            st.session_state.uploaded_df = None
            st.session_state.processed_df = None
            st.session_state.column_names = {}
            st.session_state.upload_key += 1
            st.rerun()

    with col_btn2:
        if st.button("Cancel"):
            st.session_state.uploaded_df = None
            st.session_state.processed_df = None
            st.session_state.column_names = {}
            st.session_state.upload_key += 1
            st.rerun()

else:
    st.info("Upload a CSV file to begin")

if st.session_state.protein_data is not None or st.session_state.peptide_data is not None:
    st.markdown("---")
    st.markdown("### Cached datasets")

    tab1, tab2 = st.tabs(["Protein data", "Peptide data"])

    with tab1:
        if st.session_state.protein_data is not None:
            st.caption(f"Index column: `{st.session_state.protein_index_col}`")
            st.dataframe(st.session_state.protein_data.head(5), use_container_width=True)
            if st.button("Clear protein data"):
                st.session_state.protein_data = None
                st.session_state.protein_index_col = None
                st.session_state.protein_missing_mask = None
                st.rerun()
        else:
            st.caption("No protein data uploaded yet")

    with tab2:
        if st.session_state.peptide_data is not None:
            st.caption(f"Index column: `{st.session_state.peptide_index_col}`")
            st.dataframe(st.session_state.peptide_data.head(5), use_container_width=True)
            if st.button("Clear peptide data"):
                st.session_state.peptide_data = None
                st.session_state.peptide_index_col = None
                st.session_state.peptide_missing_mask = None
                st.rerun()
        else:
            st.caption("No peptide data uploaded yet")

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {COLORS['gray']}; font-size: 12px; padding: 20px 0;">
    <p><strong>For research use only</strong></p>
    <p>© 2024 Thermo Fisher Scientific Inc. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
