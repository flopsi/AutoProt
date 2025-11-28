import io
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Header Mapping Multi-Column", layout="wide")


@st.cache_data(show_spinner="Loading file...")
def read_table(b: bytes) -> pd.DataFrame:
    txt = b.decode("utf-8", errors="replace")
    if txt.startswith("\ufeff"):
        txt = txt[1:]
    return pd.read_csv(io.StringIO(txt), sep=None, engine="python")


def main():
    st.title("Header Mapping (Multi-column with embedded widgets)")

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

    # ----- Build mapping table with separate columns for meta/quant -----
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
            new_name = ""
        else:
            role = ""
            new_name = ""

        rows.append(
            {
                "Header": col,
                "DataType": dtype,
                "Role": role,               # for string rows: selectbox
                "NewName": new_name,        # for numeric rows: text input
            }
        )

    mapping_df = pd.DataFrame(rows)

    st.subheader("Header configuration")

    edited = st.data_editor(
        mapping_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Header": st.column_config.TextColumn(
                "Header",
                disabled=True,
            ),
            "DataType": st.column_config.TextColumn(
                "Data Type",
                disabled=True,
            ),
            "Role": st.column_config.SelectboxColumn(
                "Role (for string columns)",
                options=["Protein Group", "Species", "Drop", ""],
                help="Leave empty for numeric columns.",
            ),
            "NewName": st.column_config.TextColumn(
                "New Name (for numeric columns)",
                help="Rename quantitative columns, e.g. A1, B1. Leave empty for string columns.",
            ),
        },
        hide_index=True,
        key="header_mapping_editor",
    )

    # ----- Interpret edited table -----
    meta_rows = edited[edited["DataType"] == "string"]
    quant_rows = edited[edited["DataType"] == "numeric"]

    protein_group_rows = meta_rows[meta_rows["Role"] == "Protein Group"]
    species_rows = meta_rows[meta_rows["Role"] == "Species"]

    if len(protein_group_rows) != 1:
        st.error("Set Role='Protein Group' for exactly one string header.")
        return
    if len(species_rows) != 1:
        st.error("Set Role='Species' for exactly one string header.")
        return

    protein_group_col = protein_group_rows.iloc[0]["Header"]
    species_source_col = species_rows.iloc[0]["Header"]

    st.subheader("Interpreted configuration")
    st.write(f"✓ Protein Group column: `{protein_group_col}`")
    st.write(f"✓ Species source column: `{species_source_col}`")

    # Quantitative rename map
    rename_map = {}
    for _, row in quant_rows.iterrows():
        old = row["Header"]
        new = str(row["NewName"]).strip() or old
        rename_map[old] = new
    quant_cols_renamed = [rename_map[c] for c in numeric_cols]

    st.write(f"✓ Quantitative columns renamed: {len(quant_cols_renamed)}")
    st.dataframe(
        pd.DataFrame({"Original": numeric_cols, "NewName": quant_cols_renamed}),
        use_container_width=True,
        hide_index=True,
    )

    st.success("Header configuration ready for downstream processing.")

    st.session_state["protein_mapping"] = {
        "editing_table": edited,
        "protein_group_col": protein_group_col,
        "species_source_col": species_source_col,
        "quant_rename_map": rename_map,
        "quant_cols_renamed": quant_cols_renamed,
    }


if __name__ == "__main__":
    main()
