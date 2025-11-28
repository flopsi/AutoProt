# pages/1_Upload_Protein.py
import io
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Loading file...")
def read_table(b: bytes) -> pd.DataFrame:
    txt = b.decode("utf-8", errors="replace")
    if txt.startswith("\ufeff"):
        txt = txt[1:]
    return pd.read_csv(io.StringIO(txt), sep=None, engine="python")


def detect_species_from_tags(df: pd.DataFrame, col: str | None, key_suffix: str) -> pd.Series:
    """
    Map species by searching for tags like 'HUMAN' inside strings, e.g. A0A0B4J2E5_HUMAN → HUMAN.
    """
    if col is None or col not in df.columns:
        return pd.Series(["Unknown"] * len(df), index=df.index)

    species_tags = {
        "HUMAN": "HUMAN",
        "MOUSE": "MOUSE",
        "RAT": "RAT",
        "ECOLI": "ECOLI",
        "YEAST": "YEAST",
        "BOVIN": "BOVIN",
    }

    selected = st.multiselect(
        "Species tags to search for (e.g. _HUMAN in IDs)",
        options=list(species_tags.keys()),
        default=["HUMAN", "ECOLI", "YEAST"],
        key=f"species_tags_{key_suffix}",
    )
    lookup = {tag: species_tags[tag] for tag in selected}

    def detect(v):
        if pd.isna(v):
            return "Other"
        s = str(v).upper()
        for tag, sp in lookup.items():
            if tag in s:
                return sp
        return "Other"

    return df[col].apply(detect)


def protein_upload_page():
    st.title("Protein Upload")

    st.markdown(
        "1. Upload your wide‑format protein table.\n"
        "2. Map Protein Group Identifier and species.\n"
        "3. Rename quantitative columns for downstream use."
    )

    # ---------- Upload ----------
    uploaded = st.file_uploader(
        "Upload wide-format protein table (CSV/TSV/TXT)",
        type=["csv", "tsv", "txt"],
        key="protein_uploader",
    )
    if not uploaded:
        st.info("Upload a protein file to continue.")
        return

    df_raw = read_table(uploaded.getvalue())
    st.write(f"Detected {df_raw.shape[0]:,} rows × {df_raw.shape[1]:,} columns")

    # ---------- Column type overview ----------
    numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()  # auto-quant
    non_numeric_cols = df_raw.columns.difference(numeric_cols).tolist()

    with st.expander("Column overview", expanded=False):
        st.markdown("**Numeric (auto‑detected quantitative candidates):**")
        st.write(numeric_cols or "None")
        st.markdown("**Non‑numeric (ID / metadata candidates):**")
        st.write(non_numeric_cols or "None")

    # ---------- Protein Group Identifier ----------
    st.subheader("1. Protein Group Identifier")

    if not non_numeric_cols:
        st.error("No non‑numeric columns available to use as Protein Group Identifier.")
        return

    default_pg_idx = 0
    for i, c in enumerate(non_numeric_cols):
        cl = c.lower()
        if "pg." in cl or "protein group" in cl or "protein ids" in cl:
            default_pg_idx = i
            break

    protein_group_col = st.selectbox(
        "Protein Group Identifier column",
        options=non_numeric_cols,
        index=default_pg_idx,
        help="Column that uniquely identifies a protein group (e.g. PG.ProteinGroups).",
        key="protein_group_col",
    )

    is_pg = st.checkbox(
        "This column is the Protein Group identifier",
        value=True,
        help="Uncheck only if you know this is not the true protein group column.",
        key="protein_group_is_pg",
    )
    if not is_pg:
        st.warning(
            "You unchecked the Protein Group flag; downstream modules may expect this column."
        )

    # ---------- Species mapping ----------
    st.subheader("2. Species mapping")

    species_source_col = st.selectbox(
        "Column to search for species tags (e.g. IDs or names containing _HUMAN)",
        options=non_numeric_cols,
        index=default_pg_idx,
        key="species_source_col",
    )

    species_series = detect_species_from_tags(df_raw, species_source_col, key_suffix="protein")
    st.markdown("Species counts:")
    st.write(species_series.value_counts())

    # ---------- Quantitative columns and renaming ----------
    st.subheader("3. Quantitative columns and renaming")

    if not numeric_cols:
        st.error("No numeric columns detected; cannot define quantitative data.")
        return

    st.markdown("**Select quantitative columns (numeric auto‑detected)**")
    quant_cols = st.multiselect(
        "Quantitative columns",
        options=numeric_cols,
        default=numeric_cols,
        help="Deselect numeric columns that are not intensities.",
        key="quant_cols_protein",
    )
    if not quant_cols:
        st.error("Select at least one quantitative column.")
        return

    # Two-column mapping: original header + new name
    st.markdown("**Rename quantitative columns for downstream ease of use**")
    rename_rows = []
    for col in quant_cols:
        default_short = col  # user can overwrite with e.g. A1, B1, ...
        new_name = st.text_input(
            f"New name for `{col}`",
            value=default_short,
            key=f"rename_protein_{col}",
        )
        rename_rows.append({"original": col, "new_name": new_name or col})

    mapping_df = pd.DataFrame(rename_rows)
    st.markdown("Preview of quant column name mapping:")
    st.dataframe(mapping_df, use_container_width=True)

    # Build rename dict and apply
    rename_map = {row["original"]: row["new_name"] for _, row in mapping_df.iterrows()}
    df = df_raw.rename(columns=rename_map)
    quant_cols_renamed = [rename_map[c] for c in quant_cols]

    # ---------- Final transformed frame ----------
    st.subheader("4. Preview transformed protein table")

    df_indexed = df.set_index(protein_group_col)
    species_aligned = species_series.reindex(df_indexed.index)

    preview = df_indexed[quant_cols_renamed].copy()
    preview["Species"] = species_aligned
    st.write(
        f"Proteins: {df_indexed.shape[0]:,}, "
        f"Quant columns: {len(quant_cols_renamed)}, "
        f"Unique species: {species_aligned.nunique()}"
    )
    st.dataframe(preview.head(10), use_container_width=True)

    # ---------- Store for downstream use ----------
    st.session_state["protein_upload"] = {
        "df_raw": df_raw,
        "df": df_indexed,               # indexed by Protein Group
        "quant_cols": quant_cols_renamed,
        "protein_group_col": protein_group_col,
        "species": species_aligned,
        "species_source_col": species_source_col,
        "quant_rename_table": mapping_df,
        "level": "protein",
    }

    st.success("Protein data successfully configured and stored for downstream analysis.")


# Ensure this runs when used as a page
protein_upload_page()
