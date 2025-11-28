import io
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Single Table Header Mapping", layout="wide")


@st.cache_data(show_spinner="Loading file...")
def read_table(b: bytes) -> pd.DataFrame:
    txt = b.decode("utf-8", errors="replace")
    if txt.startswith("\ufeff"):
        txt = txt[1:]
    return pd.read_csv(io.StringIO(txt), sep=None, engine="python")


def main():
    st.title("Single-Table Protein Header Mapping")

    uploaded = st.file_uploader(
        "Upload wide-format protein table (CSV/TSV/TXT)",
        type=["csv", "tsv", "txt"],
    )
    if not uploaded:
        st.info("Upload a file to start.")
        return

    df_raw = read_table(uploaded.getvalue())
    st.write(f"Table: {df_raw.shape[0]:,} rows × {df_raw.shape[1]:,} columns")

    numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = df_raw.columns.difference(numeric_cols).tolist()

    # ---------- build initial single mapping table ----------
    rows = []
    for col in df_raw.columns:
        is_numeric = col in numeric_cols
        dtype = "numeric" if is_numeric else "string"

        if not is_numeric:
            # metadata rows: default widget is Drop, unless we detect PG/Name
            cl = col.lower()
            if "pg.proteingroups" in cl or "protein group" in cl or "protein ids" in cl:
                widget = "Protein Group"   # radio choice
            elif "name" in cl:
                widget = "Species"         # radio choice
            else:
                widget = "Drop"
        else:
            # quant rows: free-text rename field, start as empty
            widget = ""

        rows.append({"Header": col, "Widget": widget, "DataType": dtype})

    mapping_df = pd.DataFrame(rows)

    st.subheader("1. Configure mapping in a single table")
    st.markdown(
        "- For **string rows**: set `Widget` to `Protein Group`, `Species`, or `Drop`.\n"
        "- For **numeric rows**: type a short name in `Widget` (e.g. A1) for downstream use."
    )

    edited = st.data_editor(
        mapping_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Header": st.column_config.TextColumn("Header Information", disabled=True),
            "Widget": st.column_config.TextColumn(
                "Widget / Role or New Name",
                help=(
                    "string rows: one of Protein Group, Species, Drop; "
                    "numeric rows: free text, e.g. A1, B1."
                ),
            ),
            "DataType": st.column_config.TextColumn("Data Type", disabled=True),
        },
        hide_index=True,
        key="single_mapping_editor",
    )

    # ---------- interpret edited table ----------
    # meta rows: where DataType == "string"
    meta_rows = edited[edited["DataType"] == "string"]
    quant_rows = edited[edited["DataType"] == "numeric"]

    # roles: Protein Group / Species / Drop
    protein_group_rows = meta_rows[meta_rows["Widget"].str.upper() == "PROTEIN GROUP"]
    species_rows = meta_rows[meta_rows["Widget"].str.upper() == "SPECIES"]

    if len(protein_group_rows) != 1:
        st.error("In the table, set **exactly one** string row's Widget to 'Protein Group'.")
        return
    if len(species_rows) != 1:
        st.error("In the table, set **exactly one** string row's Widget to 'Species'.")
        return

    protein_group_col = protein_group_rows.iloc[0]["Header"]
    species_source_col = species_rows.iloc[0]["Header"]

    # species tags from the species row's Widget cell
    # (e.g. user can type 'HUMAN;ECOLI;YEAST' there)
    species_tag_cell = species_rows.iloc[0]["Widget"]
    # default tags if empty
    if isinstance(species_tag_cell, str) and species_tag_cell.strip():
        tag_list = [t.strip().upper() for t in species_tag_cell.split(";")]
    else:
        tag_list = ["HUMAN", "ECOLI", "YEAST"]

    st.subheader("2. Interpreted species tags")
    st.write(f"Using tags from Species row: {tag_list}")

    def detect_species(val):
        if pd.isna(val):
            return "Other"
        s = str(val).upper()
        for t in tag_list:
            if t in s:
                return t
        return "Other"

    species_series = df_raw[species_source_col].apply(detect_species)
    st.write("Species counts:")
    st.write(species_series.value_counts())

    # quantitative columns: all numeric headers, renamed from Widget if non-empty
    rename_map = {}
    for _, row in quant_rows.iterrows():
        old = row["Header"]
        new = row["Widget"].strip() if isinstance(row["Widget"], str) else ""
        if not new:
            new = old
        rename_map[old] = new

    df = df_raw.rename(columns=rename_map)
    quant_cols_renamed = [rename_map[c] for c in numeric_cols]

    # ---------- final transformed DataFrame ----------
    st.subheader("3. Preview transformed DataFrame")

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

    st.session_state["protein_upload"] = {
        "df_raw": df_raw,
        "df": df_indexed,
        "mapping_table": edited,            # single table as you requested
        "protein_group_col": protein_group_col,
        "species_source_col": species_source_col,
        "species_tags": tag_list,
        "quant_cols": quant_cols_renamed,
        "species": species_aligned,
    }
    st.success("Protein data mapped and stored for downstream use.")


if __name__ == "__main__":
    main()
