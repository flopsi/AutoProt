import io
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Header Mapping Table", layout="wide")


@st.cache_data(show_spinner="Loading file...")
def read_table(b: bytes) -> pd.DataFrame:
    txt = b.decode("utf-8", errors="replace")
    if txt.startswith("\ufeff"):
        txt = txt[1:]
    return pd.read_csv(io.StringIO(txt), sep=None, engine="python")


def main():
    st.title("Header Mapping (Headers + Widget + DataType)")

    uploaded = st.file_uploader(
        "Upload wide-format protein table (CSV/TSV/TXT)",
        type=["csv", "tsv", "txt"],
    )
    if not uploaded:
        st.info("Upload a file to configure headers.")
        return

    df_raw = read_table(uploaded.getvalue())
    st.write(f"Columns detected: {len(df_raw.columns)}")

    numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = df_raw.columns.difference(numeric_cols).tolist()

    # ----- Build initial mapping: Header, Role, DataType -----
    rows = []
    for col in df_raw.columns:
        is_numeric = col in numeric_cols
        dtype = "numeric" if is_numeric else "string"

        if not is_numeric:
            cl = col.lower()
            if "pg.proteingroups" in cl or "protein group" in cl or "protein ids" in cl:
                role = "Protein Group"
            elif "name" in cl:
                role = "Species"
            else:
                role = "Drop"
        else:
            # quant: empty text rename by default
            role = ""

        rows.append({"Header": col, "RoleOrName": role, "DataType": dtype})

    mapping_df = pd.DataFrame(rows)

    st.subheader("Header configuration table")

    edited = st.data_editor(
        mapping_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Header": st.column_config.TextColumn(
                "Header",
                disabled=True,
                help="Original column name from the file.",
            ),
            "RoleOrName": st.column_config.TextColumn(
                "Widget / Role or New Name",
                help=(
                    "For string rows: type 'Protein Group', 'Species', or 'Drop'. "
                    "For numeric rows: type the short name to use downstream (e.g. A1)."
                ),
            ),
            "DataType": st.column_config.TextColumn(
                "Data type",
                disabled=True,
            ),
        },
        hide_index=True,
        key="header_mapping_editor",
    )

    # ----- Interpret edited table -----
    meta_rows = edited[edited["DataType"] == "string"]
    quant_rows = edited[edited["DataType"] == "numeric"]

    protein_group_rows = meta_rows[meta_rows["RoleOrName"].str.upper() == "PROTEIN GROUP"]
    species_rows = meta_rows[meta_rows["RoleOrName"].str.upper().str.startswith("SPECIES")]

    if len(protein_group_rows) != 1:
        st.error("Set RoleOrName='Protein Group' for exactly one string header.")
        return
    if len(species_rows) != 1:
        st.error("Set RoleOrName='Species' for exactly one string header (you can append tags).")
        return

    protein_group_col = protein_group_rows.iloc[0]["Header"]
    species_source_col = species_rows.iloc[0]["Header"]

    # Species tags: anything after 'Species' in that cell, or defaults if empty
    species_cell = str(species_rows.iloc[0]["RoleOrName"])
    parts = species_cell.split(None, 1)
    tag_part = parts[1] if len(parts) > 1 else ""
    if tag_part:
        tags = [t.strip().upper() for t in tag_part.split(";")]
    else:
        tags = ["HUMAN", "ECOLI", "YEAST"]

    st.subheader("Interpreted roles")
    st.write(f"Protein Group column: `{protein_group_col}`")
    st.write(f"Species source column: `{species_source_col}`")
    st.write(f"Species tags: {tags}")

    # Quantitative rename map
    rename_map = {}
    for _, row in quant_rows.iterrows():
        old = row["Header"]
        new = str(row["RoleOrName"]).strip() or old
        rename_map[old] = new
    quant_cols_renamed = [rename_map[c] for c in numeric_cols]

    st.subheader("Quantitative column renames")
    st.dataframe(
        pd.DataFrame(
            {"Original": numeric_cols, "NewName": quant_cols_renamed}
        ),
        use_container_width=True,
    )

    # At this stage you have:
    # - edited: the single table as you wanted
    # - protein_group_col, species_source_col, tags
    # - rename_map / quant_cols_renamed
    # You can now build transformed dataframes in downstream pages.

    st.success("Header configuration captured. Downstream processing can now use this mapping.")


if __name__ == "__main__":
    main()
