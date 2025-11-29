import streamlit as st
import pandas as pd
import re
from components import inject_custom_css, render_header, render_navigation, render_footer, COLORS

st.set_page_config(
    page_title="Data Upload | Thermo Fisher Scientific",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

inject_custom_css()
render_header()


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


# Session state init
for key, default in [
    ("protein_data", None), ("peptide_data", None),
    ("protein_index_col", None), ("peptide_index_col", None),
    ("protein_missing_mask", None), ("peptide_missing_mask", None),
    ("upload_key", 0), ("raw_df", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default


st.markdown("## Data upload")
st.caption("Import and configure mass spectrometry output matrices")

# File uploader (outside fragment - needs full rerun on new file)
uploaded_file = st.file_uploader(
    "Upload proteomics CSV",
    type=["csv"],
    key=f"uploader_{st.session_state.upload_key}"
)

if uploaded_file:
    # Cache raw dataframe in session state to avoid re-reading
    if st.session_state.raw_df is None:
        raw_df = pd.read_csv(uploaded_file)
        renamed_df, _ = detect_and_rename_numeric_cols(raw_df)
        st.session_state.raw_df = renamed_df

    renamed_df = st.session_state.raw_df
    st.success(f"Loaded {len(renamed_df):,} rows, {len(renamed_df.columns)} columns")

    non_numeric_cols = [c for c in renamed_df.columns if not pd.api.types.is_numeric_dtype(renamed_df[c])]
    numeric_cols = [c for c in renamed_df.columns if pd.api.types.is_numeric_dtype(renamed_df[c])]

    @st.fragment
    def config_fragment():
        """Fragment for configuration - reruns fast on filter changes."""
        with st.expander("Column configuration", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Protein group column**")
                protein_group_col = st.selectbox(
                    "Select protein group IDs",
                    options=["None"] + non_numeric_cols,
                    index=1 if non_numeric_cols else 0,
                    label_visibility="collapsed",
                    key="pg_col"
                )
                protein_group_col = None if protein_group_col == "None" else protein_group_col

            with col2:
                st.markdown("**Sequence column (peptide data)**")
                sequence_col = st.selectbox(
                    "Select peptide sequences",
                    options=["None"] + non_numeric_cols,
                    index=0,
                    label_visibility="collapsed",
                    key="seq_col"
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
                    key="species"
                )
            with col_sp2:
                custom_species = st.text_input("Custom tag", key="custom_sp")
                if custom_species and custom_species not in species_tags:
                    species_tags = species_tags + [custom_species]

        # Apply species filter
        filter_col = None
        for potential in ["PG.ProteinNames", "ProteinNames", "Protein.Names"]:
            if potential in renamed_df.columns:
                filter_col = potential
                break

        processed_df = renamed_df.copy()
        if species_tags and filter_col:
            processed_df = filter_by_species(processed_df, filter_col, species_tags)
            st.info(f"Filtered to {len(processed_df):,} rows matching: {', '.join(species_tags)}")

        numeric_cols_final = [c for c in processed_df.columns if pd.api.types.is_numeric_dtype(processed_df[c])]

        st.markdown("### Data preview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(processed_df):,}")
        c2.metric("Numeric columns", len(numeric_cols_final))
        c3.metric("Conditions", len(set(c.rsplit('_', 1)[0] for c in numeric_cols_final if '_' in c)))
        c4.metric("Data level", "Peptide" if is_peptide_data else "Protein")

        st.dataframe(processed_df.head(10), use_container_width=True, height=300)

        st.markdown("---")
        data_type = "peptide" if is_peptide_data else "protein"
        existing_key = f"{data_type}_data"
        index_key = f"{data_type}_index_col"
        mask_key = f"{data_type}_missing_mask"

        if st.session_state.get(existing_key) is not None:
            st.warning(f"{data_type.capitalize()} data already cached. Confirming will overwrite.")

        st.markdown(f"### Cache as {data_type} data?")
        st.caption(f"Missing values (NaN, 0) will be replaced with 1.")

        col_btn1, col_btn2, _ = st.columns([1, 1, 3])
        with col_btn1:
            if st.button("Confirm & cache", type="primary"):
                cache_df = processed_df.copy()
                missing_mask = cache_df[numeric_cols_final].isna() | (cache_df[numeric_cols_final] <= 1)
                cache_df[numeric_cols_final] = cache_df[numeric_cols_final].fillna(1).replace(0, 1)

                st.session_state[existing_key] = cache_df
                st.session_state[index_key] = protein_group_col
                st.session_state[mask_key] = missing_mask
                st.session_state.raw_df = None
                st.session_state.upload_key += 1
                st.rerun()  # Full rerun to update header

        with col_btn2:
            if st.button("Cancel"):
                st.session_state.raw_df = None
                st.session_state.upload_key += 1
                st.rerun()

    config_fragment()

else:
    st.info("Upload a CSV file to begin")


@st.fragment
def cached_data_fragment():
    """Fragment for displaying cached data."""
    if st.session_state.protein_data is None and st.session_state.peptide_data is None:
        return

    st.markdown("---")
    st.markdown("### Cached datasets")

    tab1, tab2 = st.tabs(["Protein data", "Peptide data"])

    with tab1:
        if st.session_state.protein_data is not None:
            st.caption(f"Index column: `{st.session_state.protein_index_col}`")
            st.dataframe(st.session_state.protein_data.head(5), use_container_width=True)

            if st.session_state.protein_missing_mask is not None:
                mask = st.session_state.protein_missing_mask
                total = mask.size
                missing = mask.sum().sum()
                st.caption(f"Missing values: {missing:,} ({100*missing/total:.1f}%)")

            if st.button("Clear protein data", key="clear_protein"):
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

            if st.session_state.peptide_missing_mask is not None:
                mask = st.session_state.peptide_missing_mask
                total = mask.size
                missing = mask.sum().sum()
                st.caption(f"Missing values: {missing:,} ({100*missing/total:.1f}%)")

            if st.button("Clear peptide data", key="clear_peptide"):
                st.session_state.peptide_data = None
                st.session_state.peptide_index_col = None
                st.session_state.peptide_missing_mask = None
                st.rerun()
        else:
            st.caption("No peptide data uploaded yet")

cached_data_fragment()

render_navigation(back_page="app.py", next_page="pages/2_EDA.py")
render_footer()
